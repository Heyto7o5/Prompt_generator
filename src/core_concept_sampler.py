"""
核心概念驱动采样器
以未覆盖的 level3 为核心，系统采样 companion 维度，LLM 选择兼容二级，
系统在选中二级下覆盖优先采样真实三级。
"""
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from .concept_loader import ConceptLoader, ConceptNode
from .coverage_tracker import CoverageTracker
from .models import SampledCombination, SampledConcept
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
        challenge_elements: Optional[List[Dict[str, Any]]] = None,
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
        self.challenge_elements = challenge_elements or []

        # 维度配置
        self.dimensions_config = sorted(dimensions_config, key=lambda d: d.get('core_priority', 99))
        self.key_to_sheet = {d['key']: d['sheet'] for d in dimensions_config}
        self.sheet_to_key = {d['sheet']: d['key'] for d in dimensions_config}

        # Prompt 构建器
        self.prompt_builder = ConceptSelectionPromptBuilder(loader, dimensions_config)

        # 选择缓冲区：(num_dimensions, target_challenge_count) -> items
        self._combination_buffers: Dict[Tuple[int, int], List[Dict]] = {}

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
        self,
        num_dimensions: Optional[int] = None,
        target_challenge_count: int = 0,
    ) -> Optional[SampledCombination]:
        """获取下一个组合。缓冲区空时自动通过 LLM 补充。

        Returns:
            SampledCombination 或 None（全部覆盖完成）
        """
        requested_num_dimensions = self._normalize_num_dimensions(num_dimensions)
        buffer_key = (requested_num_dimensions, max(0, target_challenge_count))
        buffer = self._combination_buffers.setdefault(buffer_key, [])

        if not buffer:
            self._refill_buffer(
                num_dimensions=requested_num_dimensions,
                target_challenge_count=target_challenge_count,
            )
            buffer = self._combination_buffers.setdefault(buffer_key, [])

        if not buffer:
            return None

        item = buffer.pop(0)
        return self._to_sampled_concepts(item)

    def sample_combination_from_anchor_pool(
        self,
        anchor_pool: Dict[str, List[ConceptNode]],
        num_dimensions: Optional[int] = None,
        target_challenge_count: int = 0,
    ) -> Optional[SampledCombination]:
        """阶段二：从已覆盖概念池选 anchor，再让 LLM 选择兼容 companion。"""
        available_dims = [dim for dim, nodes in anchor_pool.items() if nodes]
        if not available_dims:
            return None

        anchor_dim = random.choice(available_dims)
        anchor_node = random.choice(anchor_pool[anchor_dim])
        requested_num_dimensions = self._normalize_num_dimensions(num_dimensions)
        return self._sample_from_core_nodes(
            core_dim_key=anchor_dim,
            core_concepts=[anchor_node],
            num_dimensions=requested_num_dimensions,
            target_challenge_count=target_challenge_count,
            phase='phase2',
            mark_core_covered=True,
            include_fully_covered_candidates=True,
        )

    def _refill_buffer(
        self,
        num_dimensions: Optional[int] = None,
        target_challenge_count: int = 0,
    ):
        """通过 LLM 批量选二级，然后遍历三级填充缓冲区"""
        requested_num_dimensions = self._normalize_num_dimensions(num_dimensions)
        buffer_key = (requested_num_dimensions, max(0, target_challenge_count))
        buffer = self._combination_buffers.setdefault(buffer_key, [])

        # 确定核心维度
        self._current_core_dim = self._get_next_core_dim()
        if not self._current_core_dim:
            return  # 全部覆盖完成

        core_sheet = self.key_to_sheet[self._current_core_dim]

        # 获取未覆盖的核心概念
        core_concepts = self.coverage.get_next_core_concepts(core_sheet, self.batch_size)
        if not core_concepts:
            return

        items = self._build_items_from_core_nodes(
            core_dim_key=self._current_core_dim,
            core_concepts=core_concepts,
            num_dimensions=requested_num_dimensions,
            target_challenge_count=target_challenge_count,
            phase='phase1',
            mark_core_covered=True,
            include_fully_covered_candidates=False,
        )
        buffer.extend(items)

    def _sample_from_core_nodes(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        num_dimensions: int,
        target_challenge_count: int,
        phase: str,
        mark_core_covered: bool,
        include_fully_covered_candidates: bool,
    ) -> Optional[SampledCombination]:
        items = self._build_items_from_core_nodes(
            core_dim_key=core_dim_key,
            core_concepts=core_concepts,
            num_dimensions=num_dimensions,
            target_challenge_count=target_challenge_count,
            phase=phase,
            mark_core_covered=mark_core_covered,
            include_fully_covered_candidates=include_fully_covered_candidates,
        )
        if not items:
            return None
        return self._to_sampled_concepts(items[0])

    def _build_items_from_core_nodes(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        num_dimensions: int,
        target_challenge_count: int,
        phase: str,
        mark_core_covered: bool,
        include_fully_covered_candidates: bool,
    ) -> List[Dict]:
        """系统采样 companion 维度，LLM 基于指定维度选择二级类目和 challenge。"""
        requested_num_dimensions = self._normalize_num_dimensions(num_dimensions)
        target_companion_count = max(0, requested_num_dimensions - 1)
        core_sheet = self.key_to_sheet[core_dim_key]
        items: List[Dict] = []

        candidate_dims = self._get_candidate_dimensions(core_dim_key)
        if target_companion_count == 0 or not candidate_dims:
            self._append_core_only(
                core_concepts,
                core_sheet,
                items,
                core_dim_key=core_dim_key,
                phase=phase,
                target_total_dimensions=requested_num_dimensions,
            )
            return items

        # 构建动态过滤后的可用二级列表
        available_level2, covered_fallback_dims = self._build_level2_candidates(
            candidate_dims,
            include_fully_covered_candidates=include_fully_covered_candidates,
        )
        if not any(available_level2.values()):
            self._append_core_only(
                core_concepts,
                core_sheet,
                items,
                core_dim_key=core_dim_key,
                phase=phase,
                target_total_dimensions=requested_num_dimensions,
            )
            return items

        selected_candidate_dims = self._sample_companion_dimensions(
            candidate_dims,
            available_level2,
            target_companion_count,
        )
        if not selected_candidate_dims:
            self._append_core_only(
                core_concepts,
                core_sheet,
                items,
                core_dim_key=core_dim_key,
                phase=phase,
                target_total_dimensions=requested_num_dimensions,
            )
            return items

        selected_available_level2 = {
            dim_key: available_level2.get(dim_key, [])
            for dim_key in selected_candidate_dims
        }
        actual_total_dimensions = 1 + len(selected_candidate_dims)

        # 构建并发 LLM prompt
        prompt = self.prompt_builder.build_batch_prompt(
            core_dim_key=core_dim_key,
            core_concepts=core_concepts,
            available_level2=selected_available_level2,
            candidate_dimensions=selected_candidate_dims,
            target_total_dimensions=actual_total_dimensions,
            challenge_elements=self.challenge_elements,
            target_challenge_count=target_challenge_count,
        )

        # 调用 LLM（带重试）
        selections = None
        response = ''
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.generate(prompt)
                self.stats['llm_calls'] += 1

                selections = self.prompt_builder.parse_batch_response(
                    response,
                    core_concepts,
                    selected_candidate_dims,
                    len(selected_candidate_dims),
                    challenge_elements=self.challenge_elements,
                    target_challenge_count=target_challenge_count,
                )
                selections = self.prompt_builder.filter_complete_selections(
                    selections,
                    selected_candidate_dims,
                )

                if selections:
                    break
            except Exception as e:
                self.stats['llm_retries'] += 1
                print(f"  [Sampler] LLM call failed (attempt {attempt+1}): {e}")
                continue

        if not selections:
            # 全部重试失败，为每个核心概念创建只有核心维度的组合
            self._append_core_only(
                core_concepts,
                core_sheet,
                items,
                core_dim_key=core_dim_key,
                phase=phase,
                target_total_dimensions=requested_num_dimensions,
            )
            self.stats['combinations_from_backfill'] += len(core_concepts)
            return items

        # 遍历二级下未覆盖的三级，展开为组合
        for sel in selections:
            core_name = sel['core_level3']
            core_node = self._find_level3_node(core_sheet, core_name)
            if not core_node:
                continue

            # 阶段一：仅更新内存覆盖，用于当前运行内避免重复采样。
            if mark_core_covered:
                self.coverage.mark_covered(core_sheet, core_name, as_core=True)

            selection_trace = self._build_selection_trace(
                core_dim_key=core_dim_key,
                core_node=core_node,
                target_total_dimensions=actual_total_dimensions,
                target_challenge_count=target_challenge_count,
                candidate_dimensions=selected_candidate_dims,
                available_level2=selected_available_level2,
                prompt=prompt,
                raw_response=response,
                parsed_selection=sel,
                phase=phase,
            )
            selection_trace['level3_selection'] = {
                'policy': 'system_samples_real_level3_under_llm_selected_level2',
                'reason': 'LLM 只选择整体兼容的二级组合；真实三级/叶子由系统在选中二级下覆盖优先采样。',
            }
            selected_challenges = sel.get('selected_challenges', [])

            if self.level3_mode == 'traverse':
                companion_combos = self._traverse_companions(
                    sel,
                    fallback_dims=covered_fallback_dims,
                )

                if companion_combos:
                    for combo in companion_combos:
                        items.append({
                            'core_dim_key': core_dim_key,
                            'core_concept': core_node,
                            'selections': combo,
                            'selected_challenges': selected_challenges,
                            'selection_trace': selection_trace,
                            'phase': phase,
                        })
                        self.stats['combinations_generated'] += 1
                        if any(dim in covered_fallback_dims for dim in combo):
                            self.stats['combinations_with_companion_reuse'] += 1
                else:
                    # 没有可遍历的三级，只有核心概念
                    items.append({
                        'core_dim_key': core_dim_key,
                        'core_concept': core_node,
                        'selections': {},
                        'selected_challenges': selected_challenges,
                        'selection_trace': selection_trace,
                        'phase': phase,
                    })
                    self.stats['combinations_generated'] += 1
            else:
                # llm_select 模式：直接用 LLM 选的二级，随机取一个三级
                combo = self._pick_random_level3_from_selections(sel)
                items.append({
                    'core_dim_key': core_dim_key,
                    'core_concept': core_node,
                    'selections': combo,
                    'selected_challenges': selected_challenges,
                    'selection_trace': selection_trace,
                    'phase': phase,
                })
                self.stats['combinations_generated'] += 1

        return items

    def _get_target_dimensions(self, num_dimensions: Optional[int] = None) -> List[str]:
        """兼容旧接口：返回候选 companion 维度的前 N 个。"""
        total_dimensions = self._normalize_num_dimensions(num_dimensions)
        companion_dims = self._get_candidate_dimensions(self._current_core_dim)
        return companion_dims[:total_dimensions - 1]

    def _get_candidate_dimensions(self, core_dim_key: str) -> List[str]:
        """获取所有候选 companion 维度（不再按难度提前截断）。"""
        return [
            d['key'] for d in self.dimensions_config
            if d['key'] != core_dim_key and d.get('companion', True)
        ]

    def _sample_companion_dimensions(
        self,
        candidate_dims: List[str],
        available_level2: Dict[str, List[str]],
        target_companion_count: int,
    ) -> List[str]:
        """随机选择本条样本要使用的 companion 维度。"""
        usable_dims = [
            dim_key for dim_key in candidate_dims
            if available_level2.get(dim_key)
        ]
        if target_companion_count <= 0 or not usable_dims:
            return []

        sampled_count = min(target_companion_count, len(usable_dims))
        return random.sample(usable_dims, sampled_count)

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
        include_fully_covered_candidates: bool = False,
    ) -> Tuple[Dict[str, List[str]], Set[str]]:
        """优先使用未覆盖二级；维度已全覆盖时复用该维度二级作为上下文。"""
        if include_fully_covered_candidates:
            all_level2 = self._build_available_level2(target_dims, include_fully_covered=True)
            return all_level2, {dim for dim, names in all_level2.items() if names}

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
        core_dim_key: Optional[str] = None,
        phase: str = 'phase1',
        target_total_dimensions: int = 1,
    ):
        """没有可用同伴时仍输出核心概念，避免覆盖推进被同伴维度卡住。"""
        core_dim_key = core_dim_key or self._current_core_dim
        for concept in core_concepts:
            self.coverage.mark_covered(core_sheet, concept.name, as_core=True)
            buffer.append({
                'core_dim_key': core_dim_key,
                'core_concept': concept,
                'selections': {},
                'selected_challenges': [],
                'phase': phase,
                'selection_trace': {
                    'phase': phase,
                    'core': self._core_trace(core_dim_key, concept),
                    'target_total_dimensions': target_total_dimensions,
                    'target_challenge_count': 0,
                    'llm_selection': None,
                    'level3_selection': {'policy': 'core_only'},
                    'system_expansion': {},
                },
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

    def _core_trace(self, core_dim_key: str, core_node: ConceptNode) -> Dict[str, Any]:
        return {
            'dimension': core_dim_key,
            'sheet_name': self.key_to_sheet.get(core_dim_key, core_dim_key),
            'level3': core_node.name,
            'full_path': list(core_node.path),
        }

    def _build_selection_trace(
        self,
        core_dim_key: str,
        core_node: ConceptNode,
        target_total_dimensions: int,
        target_challenge_count: int,
        candidate_dimensions: List[str],
        available_level2: Dict[str, List[str]],
        prompt: str,
        raw_response: str,
        parsed_selection: Dict,
        phase: str,
    ) -> Dict[str, Any]:
        """Build trace metadata for auditing LLM compatibility selection."""
        return {
            'phase': phase,
            'core': self._core_trace(core_dim_key, core_node),
            'target_total_dimensions': target_total_dimensions,
            'target_challenge_count': target_challenge_count,
            'candidate_dimensions': list(candidate_dimensions),
            'selection_mode': 'system_sampled_dimensions_llm_selected_level2',
            'available_level2_counts': {
                dim_key: len(names)
                for dim_key, names in available_level2.items()
            },
            'llm_selection': {
                'provider': self.llm.__class__.__name__,
                'model': getattr(self.llm, 'model', ''),
                'prompt_char_length': len(prompt),
                'raw_response': raw_response,
                'selected_level2': {
                    dim_key: payload.get('level2_name')
                    for dim_key, payload in parsed_selection.get('selections', {}).items()
                },
                'selected_companions': parsed_selection.get('selected_companions', []),
                'selected_challenges': parsed_selection.get('selected_challenges', []),
                'combination_reason': parsed_selection.get('combination_reason', ''),
                'raw_combination_reason': parsed_selection.get('raw_combination_reason', ''),
                'reason_warnings': parsed_selection.get('reason_warnings', []),
                'risk_flags': parsed_selection.get('risk_flags', []),
                'confidence': parsed_selection.get('confidence', 0.5),
                'note': parsed_selection.get('note', ''),
                'validation': parsed_selection.get('validation', {}),
            },
            'level3_selection': {},
            'system_expansion': {},
        }

    def _to_sampled_concepts(self, item: Dict) -> SampledCombination:
        """将内部组合转为 SampledCombination 格式"""
        result: Dict[str, SampledConcept] = {}
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

        selection_trace = item.get('selection_trace', {}) or {}

        # 同伴概念
        system_expansion = {}
        for dim_key, l3_node in item.get('selections', {}).items():
            sheet = self.key_to_sheet[dim_key]
            leaf = self._pick_leaf(sheet, l3_node.name)
            result[dim_key] = SampledConcept(
                sheet_name=sheet,
                level3_category=l3_node.name,
                level3_path=list(l3_node.path),
                leaf=leaf,
            )
            system_expansion[dim_key] = {
                'selected_level3': l3_node.name,
                'selected_leaf': leaf,
                'level3_path': list(l3_node.path),
                'policy': 'coverage_priority_random_under_llm_selected_level2'
                if self.level3_mode == 'traverse'
                else 'random_under_selected_level2',
            }

        selection_trace['system_expansion'] = system_expansion

        return SampledCombination(
            concepts=result,
            challenge_elements=item.get('selected_challenges', []),
            selection_trace=selection_trace,
            phase=item.get('phase', 'phase1'),
        )

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
