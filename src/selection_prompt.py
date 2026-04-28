"""
LLM 概念选择 Prompt 构建器
构建批量二级类目选择 prompt，解析 LLM 返回并校验名称匹配
"""
import json
from typing import Dict, List, Optional, Set, Tuple
from .concept_loader import ConceptLoader, ConceptNode
from .coverage_tracker import CoverageTracker


class ConceptSelectionPromptBuilder:
    """为 LLM 构建二级类目选择的 prompt，并解析校验返回结果"""

    EXCLUDE_KEYWORDS = ['其他', '其它', 'Other', 'other']

    def __init__(self, loader: ConceptLoader, dimensions_config: List[Dict]):
        """
        Args:
            loader: 概念加载器
            dimensions_config: 维度配置列表, 每项 {key, sheet, core_priority, companion}
        """
        self.loader = loader
        self.dimensions_config = dimensions_config

        # key → sheet 映射
        self.key_to_sheet = {d['key']: d['sheet'] for d in dimensions_config}
        self.sheet_to_key = {d['sheet']: d['key'] for d in dimensions_config}

        # 预构建二级结构：key → {level2_display_name: {path, level3_names}}
        self.level2_structure: Dict[str, Dict[str, Dict]] = {}
        self._build_level2_structure()

        # 精确名称集合，用于校验
        self.known_level2_names: Dict[str, Set[str]] = {}
        for dim_key, l2_map in self.level2_structure.items():
            self.known_level2_names[dim_key] = set(l2_map.keys())

    def _build_level2_structure(self):
        """从 ConceptLoader 构建 level2 结构"""
        for sheet_name, category in self.loader.categories.items():
            dim_key = self.sheet_to_key.get(sheet_name, sheet_name)
            self.level2_structure[dim_key] = {}

            for level1_node in category.level1_categories:
                for level2_node in level1_node.children:
                    if level2_node.level != 2:
                        continue
                    display_name = f"{level1_node.name}-{level2_node.name}"
                    level3_names = [
                        child.name for child in level2_node.children
                        if child.level == 3
                        and not any(kw in child.name for kw in self.EXCLUDE_KEYWORDS)
                    ]
                    if level3_names:
                        self.level2_structure[dim_key][display_name] = {
                            'path': list(level2_node.path),
                            'level3_names': level3_names,
                        }

    def build_batch_prompt(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        available_level2: Dict[str, List[str]],
        target_dimensions: List[str],
    ) -> str:
        """构建批量选择 prompt

        Args:
            core_dim_key: 核心维度 key (如 'subject')
            core_concepts: 核心概念列表 (未覆盖的 level3 节点)
            available_level2: {dim_key: [level2_display_names]} 动态过滤后的可用二级
            target_dimensions: LLM 需要为哪些维度选二级
        """
        core_sheet = self.key_to_sheet[core_dim_key]
        core_dim_display = core_sheet

        lines = []
        lines.append("你是一个视频概念兼容性专家。给定核心概念，为每个核心概念从其他维度选取最兼容的二级类目。")
        lines.append("")
        lines.append("## 选取规则")
        lines.append("1. 物理合理性优先：主体和能力/环境必须匹配（如鱼不能飞）")
        lines.append("2. 使用下表中**精确的二级类目名称**，不要修改或缩写")
        lines.append("3. 尽量让不同的核心概念选择不同的二级类目，增加多样性")
        lines.append("4. 如果某个维度没有合适的选项，该维度可以留空")
        lines.append("")

        # 输出格式
        target_fields = [f'"{d}_level2"' for d in target_dimensions]
        lines.append("## 输出格式")
        lines.append("严格输出 JSON 数组，不要输出其他内容：")
        lines.append("```json")
        lines.append("[")
        lines.append('  {')
        lines.append(f'    "core_level3": "核心概念名称",')
        for d in target_dimensions:
            lines.append(f'    "{d}_level2": "选中的二级类目名称",')
        lines.append('    "confidence": 0.9,')
        lines.append('    "note": "简要说明选择理由"')
        lines.append('  }')
        lines.append("]")
        lines.append("```")
        lines.append("")

        # 可用二级列表
        lines.append("## 可选二级类目")
        for dim_key in target_dimensions:
            dim_display = self.key_to_sheet.get(dim_key, dim_key)
            names = available_level2.get(dim_key, [])
            if not names:
                continue
            lines.append(f"")
            lines.append(f"### {dim_display} ({dim_key})")
            for name in names:
                lines.append(f"- {name}")

        # 核心概念任务
        lines.append("")
        lines.append("## 核心概念")
        lines.append(f"维度: {core_dim_display} ({core_dim_key})")
        lines.append("")
        for i, concept in enumerate(core_concepts, 1):
            path_str = ' > '.join(concept.path)
            lines.append(f"{i}. {concept.name} ({path_str})")

        return '\n'.join(lines)

    def parse_batch_response(
        self,
        response: str,
        core_concepts: List[ConceptNode],
        target_dimensions: List[str],
    ) -> List[Dict]:
        """解析 LLM 返回的 JSON，校验名称匹配

        Returns:
            选中结果列表，每项:
            {
                "core_level3": str,
                "core_path": List[str],
                "selections": {dim_key: {"level2_name": str, "path": List[str]}},
                "confidence": float,
                "note": str,
                "validation": {dim_key: "exact"|"fuzzy"|"missing"}
            }
        """
        # 提取 JSON
        json_str = self._extract_json(response)
        if not json_str:
            return []

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        if not isinstance(items, list):
            items = [items]

        results = []
        core_names = {c.name for c in core_concepts}
        core_map = {c.name: c for c in core_concepts}

        for item in items:
            if not isinstance(item, dict):
                continue

            core_name = item.get('core_level3', '')
            if core_name not in core_map:
                # 尝试模糊匹配核心名称
                matched = self._fuzzy_find(core_name, core_names)
                if matched:
                    core_name = matched
                else:
                    continue

            core_node = core_map[core_name]
            result = {
                'core_level3': core_name,
                'core_path': list(core_node.path),
                'selections': {},
                'confidence': item.get('confidence', 0.5),
                'note': item.get('note', ''),
                'validation': {},
            }

            for dim_key in target_dimensions:
                field_name = f'{dim_key}_level2'
                raw_name = item.get(field_name, '')
                if not raw_name:
                    result['validation'][dim_key] = 'missing'
                    continue

                known = self.known_level2_names.get(dim_key, set())
                matched, match_type = self._match_level2_name(raw_name, known)

                if matched:
                    l2_info = self.level2_structure[dim_key][matched]
                    result['selections'][dim_key] = {
                        'level2_name': matched,
                        'path': l2_info['path'],
                    }
                    result['validation'][dim_key] = match_type
                else:
                    result['validation'][dim_key] = 'missing'

            results.append(result)

        return results

    def fill_missing_selections(
        self,
        selections: List[Dict],
        available_level2: Dict[str, List[str]],
        target_dimensions: List[str],
    ) -> List[Dict]:
        """对缺失的维度从未覆盖二级中随机补选"""
        import random

        for sel in selections:
            for dim_key in target_dimensions:
                if sel['validation'].get(dim_key) == 'missing':
                    available = available_level2.get(dim_key, [])
                    if available:
                        chosen = random.choice(available)
                        l2_info = self.level2_structure[dim_key].get(chosen)
                        if l2_info:
                            sel['selections'][dim_key] = {
                                'level2_name': chosen,
                                'path': l2_info['path'],
                            }
                            sel['validation'][dim_key] = 'backfill'
        return selections

    # ── 内部方法 ─────────────────────────────────────

    def _extract_json(self, text: str) -> Optional[str]:
        """从 LLM 响应中提取 JSON，支持截断修复"""
        # 尝试直接解析
        text = text.strip()
        if text.startswith('[') or text.startswith('{'):
            # 先尝试原样解析
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass

            # 尝试修复截断的 JSON 数组
            if text.startswith('[') and ']' not in text:
                # 找到最后一个完整的 }
                last_brace = text.rfind('}')
                if last_brace > 0:
                    repaired = text[:last_brace + 1] + ']'
                    try:
                        json.loads(repaired)
                        return repaired
                    except json.JSONDecodeError:
                        pass

                # 尝试逐步截断到上一个完整对象
                repaired = text + ']'
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass

            return text

        # 提取 ```json ... ``` 块
        if '```json' in text:
            start = text.index('```json') + 7
            end = text.find('```', start)
            if end > start:
                return text[start:end].strip()

        if '```' in text:
            start = text.index('```') + 3
            end = text.find('```', start)
            if end > start:
                return text[start:end].strip()

        # 尝试找第一个 [ 到最后一个 ]
        bracket_start = text.find('[')
        bracket_end = text.rfind(']')
        if bracket_start >= 0 and bracket_end > bracket_start:
            return text[bracket_start:bracket_end + 1]

        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            return text[brace_start:brace_end + 1]

        return None

    def _match_level2_name(self, raw: str, known: Set[str]) -> Tuple[Optional[str], str]:
        """匹配二级名称，返回 (匹配结果, 匹配类型)"""
        # 精确匹配
        if raw in known:
            return raw, 'exact'

        # 去除空格后精确匹配
        raw_stripped = raw.strip()
        if raw_stripped in known:
            return raw_stripped, 'exact'

        # 包含匹配：known 中包含 raw，或 raw 包含 known
        for name in known:
            if raw in name or name in raw:
                return name, 'fuzzy'

        # 去除分隔符后匹配
        raw_clean = raw.lower().replace('-', '').replace('_', '').replace(' ', '')
        for name in known:
            name_clean = name.lower().replace('-', '').replace('_', '').replace(' ', '')
            if raw_clean == name_clean:
                return name, 'fuzzy'
            if raw_clean in name_clean or name_clean in raw_clean:
                return name, 'fuzzy'

        return None, 'missing'

    def _fuzzy_find(self, name: str, candidates: Set[str]) -> Optional[str]:
        """在候选集合中模糊查找"""
        if name in candidates:
            return name
        for c in candidates:
            if name in c or c in name:
                return c
        return None
