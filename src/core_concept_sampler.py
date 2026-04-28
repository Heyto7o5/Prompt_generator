"""
核心概念驱动采样器
以未覆盖的 level3 为核心，LLM 批量选兼容二级，系统遍历三级保证覆盖
"""
import random
from typing import Dict, List, Optional, Set, Tuple
from .concept_loader import ConceptLoader, ConceptNode
from .coverage_tracker import CoverageTracker
from .models import SampledConcept
from .selection_prompt import ConceptSelectionPromptBuilder


class CoreConceptDrivenSampler:
    """核心概念驱动的 LLM 辅助采样器"""

    EXCLUDE_KEYWORDS = ['其他', '其它', 'Other', 'other']

    def __init__(
        self,
        loader: ConceptLoader,
        llm_provider,
        coverage_tracker: CoverageTracker,
        dimensions_config: List[Dict],
        batch_size: int = 10,
        num_dimensions: int = 3,
        max_retries: int = 2,
        level3_mode: str = 'traverse',
    ):
        """
        Args:
            loader: 概念加载器
            llm_provider: LLM 提供者 (需要有 generate 方法)
            coverage_tracker: 覆盖追踪器
            dimensions_config: 维度配置列表
            batch_size: 每次 LLM 调用的核心概念数量
            num_dimensions: 阶段一每个组合选几个维度 (3 or 4)
            max_retries: LLM 调用失败最大重试次数
            level3_mode: 三级展开模式 "traverse" or "llm_select"
        """
        self.loader = loader
        self.llm = llm_provider
        self.coverage = coverage_tracker
        self.batch_size = batch_size
        self.num_dimensions = num_dimensions
        self.max_retries = max_retries
        self.level3_mode = level3_mode

        # 维度配置
        self.dimensions_config = sorted(dimensions_config, key=lambda d: d.get('core_priority', 99))
        self.key_to_sheet = {d['key']: d['sheet'] for d in dimensions_config}
        self.sheet_to_key = {d['sheet']: d['key'] for d in dimensions_config}

        # Prompt 构建器
        self.prompt_builder = ConceptSelectionPromptBuilder(loader, dimensions_config)

        # 选择缓冲区：num_dimensions -> [{core_dim_key, core_concept, selections}]
        self._combination_buffers: Dict[int, List[Dict]] = {}

        # 当前核心维度
        self._current_core_dim = self._get_next_core_dim()

        # 统计
        self.stats = {
            'llm_calls': 0,
            'llm_retries': 0,
            'combinations_generated': 0,
            'combinations_from_backfill': 0,
            'combinations_with_companion_reuse': 0,
        }

    def _get_next_core_dim(self) -> Optional[str]:
        """找到还有未覆盖 level3 的、优先级最高的维度"""
        for dim in self.dimensions_config:
            sheet = dim['sheet']
            if self.coverage.get_uncovered_count(sheet) > 0:
                return dim['key']
        return None

    def sample_combination(
        self, num_dimensions: Optional[int] = None
    ) -> Optional[Dict[str, SampledConcept]]:
        """获取下一个组合。缓冲区空时自动通过 LLM 补充。

        Returns:
            Dict[str, SampledConcept] 或 None（全部覆盖完成）
        """
        requested_num_dimensions = self._normalize_num_dimensions(num_dimensions)
        buffer = self._combination_buffers.setdefault(requested_num_dimensions, [])

        if not buffer:
            self._refill_buffer(num_dimensions=requested_num_dimensions)
            buffer = self._combination_buffers.setdefault(requested_num_dimensions, [])

        if not buffer:
            return None

        item = buffer.pop(0)
        return self._to_sampled_concepts(item)

    def _refill_buffer(self, num_dimensions: Optional[int] = None):
        """通过 LLM 批量选二级，然后遍历三级填充缓冲区"""
        requested_num_dimensions = self._normalize_num_dimensions(num_dimensions)
        buffer = self._combination_buffers.setdefault(requested_num_dimensions, [])

        # 确定核心维度
        self._current_core_dim = self._get_next_core_dim()
        if not self._current_core_dim:
            return  # 全部覆盖完成

        core_sheet = self.key_to_sheet[self._current_core_dim]

        # 获取未覆盖的核心概念
        core_concepts = self.coverage.get_next_core_concepts(core_sheet, self.batch_size)
        if not core_concepts:
            return

        # 确定目标维度（排除核心维度，取 num_dimensions-1 个同伴维度）
        target_dims = self._get_target_dimensions(num_dimensions=num_dimensions)
        if not target_dims:
            self._append_core_only(core_concepts, core_sheet, buffer)
            return

        # 构建动态过滤后的可用二级列表
        available_level2, covered_fallback_dims = self._build_level2_candidates(target_dims)
        if not any(available_level2.values()):
            self._append_core_only(core_concepts, core_sheet, buffer)
            return

        # 构建并发 LLM prompt
        prompt = self.prompt_builder.build_batch_prompt(
            core_dim_key=self._current_core_dim,
            core_concepts=core_concepts,
            available_level2=available_level2,
            target_dimensions=target_dims,
        )

        # 调用 LLM（带重试）
        selections = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.generate(prompt)
                self.stats['llm_calls'] += 1

                selections = self.prompt_builder.parse_batch_response(
                    response, core_concepts, target_dims
                )

                if selections:
                    break
            except Exception as e:
                self.stats['llm_retries'] += 1
                print(f"  [Sampler] LLM call failed (attempt {attempt+1}): {e}")
                continue

        if not selections:
            # 全部重试失败，为每个核心概念创建只有核心维度的组合
            self._append_core_only(core_concepts, core_sheet, buffer)
            self.stats['combinations_from_backfill'] += len(core_concepts)
            return

        # 补选缺失维度
        selections = self.prompt_builder.fill_missing_selections(
            selections, available_level2, target_dims
        )

        # 遍历二级下未覆盖的三级，展开为组合
        for sel in selections:
            core_name = sel['core_level3']
            core_node = self._find_level3_node(core_sheet, core_name)
            if not core_node:
                continue

            # 阶段一：仅更新内存覆盖，用于当前运行内避免重复采样。
            self.coverage.mark_covered(core_sheet, core_name, as_core=True)

            if self.level3_mode == 'traverse':
                # 遍历每个同伴维度下选定二级的未覆盖三级
                companion_combos = self._traverse_companions(
                    sel,
                    fallback_dims=covered_fallback_dims,
                )

                if companion_combos:
                    for combo in companion_combos:
                        buffer.append({
                            'core_dim_key': self._current_core_dim,
                            'core_concept': core_node,
                            'selections': combo,
                        })
                        self.stats['combinations_generated'] += 1
                        if any(dim in covered_fallback_dims for dim in combo):
                            self.stats['combinations_with_companion_reuse'] += 1
                else:
                    # 没有可遍历的三级，只有核心概念
                    buffer.append({
                        'core_dim_key': self._current_core_dim,
                        'core_concept': core_node,
                        'selections': {},
                    })
                    self.stats['combinations_generated'] += 1
            else:
                # llm_select 模式：直接用 LLM 选的二级，随机取一个三级
                combo = self._pick_random_level3_from_selections(sel)
                buffer.append({
                    'core_dim_key': self._current_core_dim,
                    'core_concept': core_node,
                    'selections': combo,
                })
                self.stats['combinations_generated'] += 1

    def _get_target_dimensions(self, num_dimensions: Optional[int] = None) -> List[str]:
        """获取目标维度列表（排除核心维度）"""
        total_dimensions = self._normalize_num_dimensions(num_dimensions)
        companion_dims = [
            d['key'] for d in self.dimensions_config
            if d['key'] != self._current_core_dim and d.get('companion', True)
        ]
        # 限制维度数
        return companion_dims[:total_dimensions - 1]

    def _normalize_num_dimensions(self, num_dimensions: Optional[int] = None) -> int:
        total_dimensions = num_dimensions if num_dimensions is not None else self.num_dimensions
        return max(1, min(total_dimensions, len(self.dimensions_config)))

    def _build_available_level2(
        self,
        target_dims: List[str],
        include_fully_covered: bool = False,
    ) -> Dict[str, List[str]]:
        """构建动态过滤后的可用二级列表。

        默认只返回仍有未覆盖三级的二级；当某个同伴维度已经全覆盖时，
        include_fully_covered=True 允许把已覆盖二级作为上下文复用。
        """
        available = {}
        for dim_key in target_dims:
            sheet = self.key_to_sheet[dim_key]
            l2_structure = self.prompt_builder.level2_structure.get(dim_key, {})
            available[dim_key] = []

            for l2_name, l2_info in l2_structure.items():
                l2_path = l2_info['path']
                if (
                    include_fully_covered
                    or not self.coverage.is_level2_fully_covered(sheet, l2_path)
                ):
                    available[dim_key].append(l2_name)

        return available

    def _build_level2_candidates(
        self,
        target_dims: List[str],
    ) -> Tuple[Dict[str, List[str]], Set[str]]:
        """优先使用未覆盖二级；维度已全覆盖时复用该维度二级作为上下文。"""
        uncovered_level2 = self._build_available_level2(target_dims, include_fully_covered=False)
        all_level2 = None
        available: Dict[str, List[str]] = {}
        covered_fallback_dims: Set[str] = set()

        for dim_key in target_dims:
            if uncovered_level2.get(dim_key):
                available[dim_key] = uncovered_level2[dim_key]
                continue

            if all_level2 is None:
                all_level2 = self._build_available_level2(target_dims, include_fully_covered=True)

            reusable = all_level2.get(dim_key, []) if all_level2 else []
            available[dim_key] = reusable
            if reusable:
                covered_fallback_dims.add(dim_key)

        return available, covered_fallback_dims

    def _append_core_only(
        self,
        core_concepts: List[ConceptNode],
        core_sheet: str,
        buffer: List[Dict],
    ):
        """没有可用同伴时仍输出核心概念，避免覆盖推进被同伴维度卡住。"""
        for concept in core_concepts:
            self.coverage.mark_covered(core_sheet, concept.name, as_core=True)
            buffer.append({
                'core_dim_key': self._current_core_dim,
                'core_concept': concept,
                'selections': {},
            })
            self.stats['combinations_generated'] += 1

    def _traverse_companions(
        self,
        sel: Dict,
        fallback_dims: Optional[Set[str]] = None,
    ) -> List[Dict[str, ConceptNode]]:
        """遍历选定二级下所有未覆盖的三级，返回组合列表

        每个组合: {dim_key: ConceptNode(三级)}
        """
        fallback_dims = fallback_dims or set()
        selections = sel.get('selections', {})
        if not selections:
            return []

        # 收集每个维度下未覆盖的三级节点
        dim_level3_options: Dict[str, List[ConceptNode]] = {}

        for dim_key, l2_info in selections.items():
            sheet = self.key_to_sheet[dim_key]
            l2_path = l2_info['path']

            # 获取该二级下未覆盖的三级
            uncovered_names = self.coverage.get_uncovered_level3_under_level2(sheet, l2_path)
            l3_nodes = self._get_level3_nodes_under_level2(
                sheet,
                l2_path,
                allowed_names=set(uncovered_names),
            )

            # 该同伴维度已全覆盖时，复用已覆盖三级作为上下文，不让核心维度覆盖停滞。
            if not l3_nodes and dim_key in fallback_dims:
                l3_nodes = self._get_level3_nodes_under_level2(sheet, l2_path)

            if l3_nodes:
                dim_level3_options[dim_key] = l3_nodes

        if not dim_level3_options:
            return []

        # 阶段一优先让核心概念去重：每个 core level3 只展开一条 prompt。
        combos = self._cartesian_product(dim_level3_options, max_combos=1)

        # 仅更新内存覆盖，用于当前运行内避免重复采样。
        records = []
        for combo in combos:
            for dim_key, node in combo.items():
                sheet = self.key_to_sheet[dim_key]
                records.append({
                    'sheet_name': sheet,
                    'level3_name': node.name,
                    'as_core': False,
                })
        if records:
            self.coverage.mark_batch_covered(records, save=False)

        return combos

    def _get_level3_nodes_under_level2(
        self,
        sheet_name: str,
        l2_path: List[str],
        allowed_names: Optional[Set[str]] = None,
    ) -> List[ConceptNode]:
        """取二级下的三级节点，按当前覆盖次数升序，降低同伴重复。"""
        nodes = []
        for node in self.loader.get_level3_categories(sheet_name):
            if len(node.path) < 2 or len(l2_path) < 2:
                continue
            if node.path[0] != l2_path[0] or node.path[1] != l2_path[1]:
                continue
            if any(kw in node.name for kw in self.EXCLUDE_KEYWORDS):
                continue
            if allowed_names is not None and node.name not in allowed_names:
                continue
            nodes.append(node)

        random.shuffle(nodes)
        nodes.sort(key=lambda n: self.coverage.get_times_covered(sheet_name, n.name))
        return nodes

    def _cartesian_product(
        self, dim_options: Dict[str, List[ConceptNode]], max_combos: int = 20
    ) -> List[Dict[str, ConceptNode]]:
        """生成维度间的笛卡尔积组合，限制总数"""
        if not dim_options:
            return []

        dims = list(dim_options.keys())

        # 如果某个维度选项太多，截断
        truncated = {}
        for dim in dims:
            options = dim_options[dim]
            truncated[dim] = options[:max_combos]

        result = []
        self._cartesian_recursive(truncated, dims, 0, {}, result, max_combos)
        return result

    def _cartesian_recursive(
        self, dim_options, dims, idx, current, result, max_combos
    ):
        if len(result) >= max_combos:
            return
        if idx >= len(dims):
            result.append(dict(current))
            return

        dim = dims[idx]
        for option in dim_options[dim]:
            current[dim] = option
            self._cartesian_recursive(dim_options, dims, idx + 1, current, result, max_combos)
            if len(result) >= max_combos:
                return

    def _pick_random_level3_from_selections(self, sel: Dict) -> Dict[str, ConceptNode]:
        """llm_select 模式下，从选定二级中随机取一个三级"""
        result = {}
        for dim_key, l2_info in sel.get('selections', {}).items():
            sheet = self.key_to_sheet[dim_key]
            l2_path = l2_info['path']

            all_level3 = self.loader.get_level3_categories(sheet)
            candidates = [
                n for n in all_level3
                if len(n.path) >= 2
                and n.path[0] == l2_path[0]
                and n.path[1] == l2_path[1]
                and not any(kw in n.name for kw in self.EXCLUDE_KEYWORDS)
            ]
            if candidates:
                chosen = random.choice(candidates)
                result[dim_key] = chosen
                sheet_name = self.key_to_sheet[dim_key]
                self.coverage.mark_covered(sheet_name, chosen.name, as_core=False)

        return result

    def _find_level3_node(self, sheet_name: str, level3_name: str) -> Optional[ConceptNode]:
        """在 ConceptLoader 中查找 level3 节点"""
        for node in self.loader.get_level3_categories(sheet_name):
            if node.name == level3_name:
                return node
        return None

    def _to_sampled_concepts(self, item: Dict) -> Dict[str, SampledConcept]:
        """将内部组合转为 Dict[str, SampledConcept] 格式"""
        result = {}
        core_dim = item['core_dim_key']
        core_node = item['core_concept']
        core_sheet = self.key_to_sheet[core_dim]

        # 核心概念
        result[core_dim] = SampledConcept(
            sheet_name=core_sheet,
            level3_category=core_node.name,
            level3_path=list(core_node.path),
            leaf=self._pick_leaf(core_sheet, core_node.name),
        )

        # 同伴概念
        for dim_key, l3_node in item.get('selections', {}).items():
            sheet = self.key_to_sheet[dim_key]
            result[dim_key] = SampledConcept(
                sheet_name=sheet,
                level3_category=l3_node.name,
                level3_path=list(l3_node.path),
                leaf=self._pick_leaf(sheet, l3_node.name),
            )

        return result

    def _pick_leaf(self, sheet_name: str, level3_name: str) -> Optional[str]:
        """从三级类目下随机选一个叶子节点"""
        leaves = self.loader.get_leaves_under_level3(sheet_name, level3_name)
        valid = [l for l in leaves if not any(kw in l.name for kw in self.EXCLUDE_KEYWORDS)]
        if valid:
            return random.choice(valid).name
        return None

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'coverage': self.coverage.get_report(),
            'buffer_size': sum(len(buffer) for buffer in self._combination_buffers.values()),
        }
