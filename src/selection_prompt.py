"""
LLM 概念选择 Prompt 构建器
构建批量二级类目选择 prompt，解析 LLM 返回并校验名称匹配
"""
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from .concept_loader import ConceptLoader, ConceptNode


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
        self.known_level3_names: Set[str] = set()
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
                        self.known_level3_names.update(level3_names)
                        self.level2_structure[dim_key][display_name] = {
                            'path': list(level2_node.path),
                            'level3_names': level3_names,
                        }

    def build_batch_prompt(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        available_level2: Dict[str, List[str]],
        candidate_dimensions: List[str],
        target_total_dimensions: int,
        challenge_elements: Optional[List[Dict[str, Any]]] = None,
        target_challenge_count: int = 0,
    ) -> str:
        """构建批量选择 prompt

        Args:
            core_dim_key: 核心维度 key (如 'subject')
            core_concepts: 核心概念列表 (未覆盖的 level3 节点)
            available_level2: {dim_key: [level2_display_names]} 动态过滤后的可用二级
            candidate_dimensions: LLM 可从哪些 companion 维度中选择
            target_total_dimensions: 最终组合包含的总维度数，含核心维度
            challenge_elements: 可选挑战性要素列表
            target_challenge_count: 建议选择的挑战性要素数量
        """
        core_sheet = self.key_to_sheet[core_dim_key]
        core_dim_display = core_sheet
        companion_count = max(0, target_total_dimensions - 1)
        challenge_elements = challenge_elements or []

        lines = []
        lines.append("你是一个视频概念组合规划专家。你的任务不是生成最终视频 prompt，而是基于给定的核心概念、指定 companion 维度，以及每个 companion 维度下的二级类目，选择一个整体兼容的二级类目组合。")
        lines.append("")
        lines.append("## 选取规则")
        lines.append("1. 当前阶段只展示二级类目，系统稍后才会从选中的二级类目下随机或覆盖优先采样真实三级/叶子概念。")
        lines.append("2. 你不能假设、猜测、引用或依赖任何未展示的三级概念、叶子概念、名人专长、外部事实或百科知识。")
        lines.append("3. combination_reason 必须解释“核心概念 + 所有选中二级类目”在二级语义层面为什么能共同构成一个视频场面，而不是分别解释两两关系。")
        lines.append("4. 不允许出现类似“张继科擅长乒乓球，所以选择球类运动”的理由，因为系统后续可能从球类运动下采样到排球、篮球、足球等任意真实三级。")
        lines.append("5. 合法理由应停留在二级整体语义，例如“体育人物、球类运动和训练场馆可以形成运动训练或挑战场面，不依赖具体球类项目。”")
        lines.append("6. 必须为每个指定 companion 维度选择一个二级类目，二级名称必须从该维度候选列表中逐字复制，不能自行创造、改写、组合、缩写、翻译、泛化或补充不存在的二级条目。")
        lines.append("7. 如果候选列表中没有你理想的二级类目，也必须从已有候选中选择最接近且整体兼容的一项；绝对不要输出候选列表之外的二级名称。")
        lines.append("8. 输出前逐项自查：selected_level2 中每个 value 必须与对应维度候选列表里的某一行完全一致；否则该输出无效。")
        lines.append("9. 挑战性要素只有在未来最终 prompt 中能转化为可观察的视觉/听觉证据时才选择；如果无法想象具体落地方式，可以少选或不选。")
        lines.append("10. challenge reason 必须说明该挑战会通过什么可观察动作、声音、空间关系、物理变化或因果过程体现；不能只写“增强表现力”“更有动态感”等抽象理由。")
        lines.append("11. 挑战性要素必须基于“核心概念 + 选中二级类目组合”的整体语义选择和解释；reason 不允许引用、期待或绑定未展示的具体三级或叶子概念。")
        lines.append("12. 保持中文语义，不要把中文概念替换成英文解释。")
        lines.append("13. 输出严格 JSON，不要输出解释、Markdown 或代码块外文本。")
        lines.append("")

        # 输出格式
        lines.append("## 输出格式")
        lines.append("严格输出 JSON 数组，不要输出其他内容：")
        lines.append("```json")
        lines.append("[")
        lines.append('  {')
        lines.append(f'    "core_level3": "核心概念名称",')
        lines.append(f'    "target_total_dimensions": {target_total_dimensions},')
        lines.append('    "selected_level2": {')
        lines.append('      "指定维度key": "从该维度候选列表逐字复制的二级类目名称"')
        lines.append('    },')
        lines.append('    "combination_reason": "只基于核心概念与所有选中二级类目的整体兼容理由，不引用未展示三级、叶子或外部事实",')
        lines.append('    "selected_challenges": [')
        lines.append('      {')
        lines.append('        "id": "挑战性要素id",')
        lines.append('        "reason": "只基于二级组合整体语义的选择理由，并说明可观察落地方式，不绑定具体三级或叶子"')
        lines.append('      }')
        lines.append('    ],')
        lines.append('    "risk_flags": [],')
        lines.append('    "confidence": 0.9')
        lines.append('  }')
        lines.append("]")
        lines.append("```")
        lines.append("")

        # 可用二级列表
        lines.append("## 指定 companion 维度与可选二级类目")
        lines.append(f"本条样本最终必须包含 {target_total_dimensions} 个维度：核心维度 1 个，指定 companion 维度 {companion_count} 个。")
        for dim_key in candidate_dimensions:
            dim_display = self.key_to_sheet.get(dim_key, dim_key)
            names = available_level2.get(dim_key, [])
            if not names:
                continue
            lines.append(f"")
            lines.append(f"### {dim_display} ({dim_key})")
            for name in names:
                lines.append(f"- {name}")

        if challenge_elements:
            lines.append("")
            lines.append("## 可选挑战性要素")
            lines.append(f"建议选择数量: {target_challenge_count}。如果没有自然兼容的挑战性要素，可以少选或不选。")
            for elem in challenge_elements:
                elem_id = elem.get('id', '')
                elem_name = elem.get('name', elem_id)
                desc = elem.get('description', '')
                lines.append(f"- {elem_id} | {elem_name}: {desc}")

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
        candidate_dimensions: List[str],
        target_companion_count: int,
        challenge_elements: Optional[List[Dict[str, Any]]] = None,
        target_challenge_count: int = 0,
    ) -> List[Dict]:
        """解析 LLM 返回的 JSON，校验名称匹配

        Returns:
            选中结果列表，每项:
            {
                "core_level3": str,
                "core_path": List[str],
                "selections": {dim_key: {"level2_name": str, "path": List[str]}},
                "selected_challenges": List[Dict],
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
        candidate_set = set(candidate_dimensions)
        challenge_elements = challenge_elements or []

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
            raw_combination_reason = str(
                item.get('combination_reason') or item.get('note', '')
            ).strip()
            combination_reason, combination_warnings = self._inspect_level2_reason(
                raw_combination_reason,
                allowed_level3_names={core_name},
            )
            result = {
                'core_level3': core_name,
                'core_path': list(core_node.path),
                'selections': {},
                'selected_companions': [],
                'selected_challenges': [],
                'risk_flags': item.get('risk_flags', []) if isinstance(item.get('risk_flags', []), list) else [],
                'confidence': item.get('confidence', 0.5),
                'combination_reason': combination_reason,
                'raw_combination_reason': raw_combination_reason,
                'reason_warnings': combination_warnings,
                'note': item.get('note', combination_reason),
                'validation': {},
            }

            companion_items = []

            selected_level2 = item.get('selected_level2', {})
            if isinstance(selected_level2, dict):
                for dim_key, level2_name in selected_level2.items():
                    companion_items.append({
                        'dimension': dim_key,
                        'level2_name': level2_name,
                        'reason': combination_reason,
                    })

            raw_companion_items = item.get('selected_companions', [])
            if isinstance(raw_companion_items, list):
                companion_items.extend(raw_companion_items)

            # Backward compatibility with the old {motion_level2: "..."} schema.
            for dim_key in candidate_dimensions:
                legacy_name = item.get(f'{dim_key}_level2')
                if legacy_name:
                    companion_items.append({
                        'dimension': dim_key,
                        'level2_name': legacy_name,
                        'reason': item.get('note', ''),
                    })

            used_dims = set()
            for companion in companion_items:
                if not isinstance(companion, dict):
                    continue

                dim_key = self._match_dimension_key(
                    str(companion.get('dimension', '')).strip(),
                    candidate_set,
                )
                if dim_key not in candidate_set or dim_key in used_dims:
                    continue

                raw_name = str(companion.get('level2_name', '')).strip()
                if not raw_name:
                    result['validation'][dim_key] = 'missing'
                    continue

                known = self.known_level2_names.get(dim_key, set())
                matched, match_type = self._match_level2_name(raw_name, known)

                if matched:
                    l2_info = self.level2_structure[dim_key][matched]
                    companion_reason, companion_warnings = self._inspect_level2_reason(
                        str(companion.get('reason', combination_reason)).strip(),
                        allowed_level3_names={core_name},
                    )
                    result['selections'][dim_key] = {
                        'level2_name': matched,
                        'path': l2_info['path'],
                        'reason': companion_reason,
                        'reason_warnings': companion_warnings,
                    }
                    result['selected_companions'].append({
                        'dimension': dim_key,
                        'level2_name': matched,
                        'path': l2_info['path'],
                    })
                    result['validation'][dim_key] = match_type
                    used_dims.add(dim_key)
                else:
                    result['validation'][dim_key] = 'missing'

                if len(result['selections']) >= target_companion_count:
                    break

            raw_challenges = item.get('selected_challenges', [])
            if not isinstance(raw_challenges, list):
                raw_challenges = []
            for raw_challenge in raw_challenges:
                matched = self._match_challenge(raw_challenge, challenge_elements)
                if matched:
                    challenge_reason, challenge_warnings = self._inspect_level2_reason(
                        str(matched.get('reason', '')).strip(),
                        allowed_level3_names={core_name},
                    )
                    if challenge_reason:
                        matched['reason'] = challenge_reason
                    else:
                        matched.pop('reason', None)
                    if challenge_warnings:
                        matched['reason_warnings'] = challenge_warnings
                    result['selected_challenges'].append(matched)
                if len(result['selected_challenges']) >= target_challenge_count:
                    break

            results.append(result)

        return results

    def filter_complete_selections(
        self,
        selections: List[Dict],
        required_dimensions: List[str],
    ) -> List[Dict]:
        """Keep only selections where LLM picked every system-specified dimension."""
        required = set(required_dimensions)
        complete = []

        for sel in selections:
            selected = set(sel.get('selections', {}).keys())
            missing = sorted(required - selected)
            if missing:
                sel.setdefault('validation', {})['missing_required_dimensions'] = missing
                continue
            complete.append(sel)

        return complete

    # ── 内部方法 ─────────────────────────────────────

    def _normalize_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _inspect_level2_reason(
        self,
        reason: str,
        allowed_level3_names: Optional[Set[str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Keep the raw reason and report likely Level-3 leaks for audit."""
        reason = reason.strip()
        if not reason:
            return reason, []

        allowed = allowed_level3_names or set()
        leaked_terms = []
        for level3_name in self.known_level3_names:
            if level3_name in allowed:
                continue
            if len(level3_name) < 2:
                continue
            if level3_name and level3_name in reason:
                leaked_terms.append(level3_name)

        if not leaked_terms:
            return reason, []

        leaked_terms = sorted(set(leaked_terms), key=lambda item: (-len(item), item))
        return reason, [{
            'type': 'reason_mentions_unshown_level3_or_leaf',
            'terms': leaked_terms[:20],
            'message': '理由中疑似引用了当前选择阶段未展示的三级/叶子概念；已保留原文用于审查。',
        }]

    def _match_dimension_key(self, raw: str, candidates: Set[str]) -> str:
        """Accept exact keys plus common model outputs like '运动' or '运动 (motion)'."""
        if raw in candidates:
            return raw

        raw_clean = raw.strip()
        raw_lower = raw_clean.lower()
        for dim_key in candidates:
            display = self.key_to_sheet.get(dim_key, '')
            if raw_clean == display:
                return dim_key
            if f"({dim_key})" in raw_lower:
                return dim_key
            if raw_lower == dim_key.lower():
                return dim_key
            if dim_key.lower() in raw_lower:
                return dim_key
            if display and raw_clean.startswith(display):
                return dim_key
        return raw

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

    def _match_challenge(
        self,
        raw_challenge: Any,
        challenge_elements: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Match a challenge by id or name and preserve the LLM reason."""
        if not challenge_elements:
            return None

        if isinstance(raw_challenge, dict):
            raw_id = str(raw_challenge.get('id', '')).strip()
            raw_name = str(raw_challenge.get('name', '')).strip()
            reason = raw_challenge.get('reason', '')
        else:
            raw_id = str(raw_challenge).strip()
            raw_name = raw_id
            reason = ''

        for elem in challenge_elements:
            elem_id = str(elem.get('id', '')).strip()
            elem_name = str(elem.get('name', '')).strip()
            if raw_id == elem_id or raw_id == elem_name or raw_name == elem_id or raw_name == elem_name:
                matched = dict(elem)
                matched['reason'] = reason
                return matched

        return None
