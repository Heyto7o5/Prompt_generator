"""
Semantic top-k concept sampler.

This sampler samples companion dimensions in system code, then lets the LLM
choose compatible level-2 and level-3 concepts from a compact top-k candidate
pool.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .concept_loader import ConceptLoader, ConceptNode
from .coverage_tracker import CoverageTracker
from .models import SampledCombination, SampledConcept
from .selection_prompt import ConceptSelectionPromptBuilder


EXCLUDE_KEYWORDS = ["其他", "其它", "Other", "other"]


@dataclass
class SemanticTopKParams:
    """Candidate-pool parameters derived from the taxonomy shape."""

    level2_candidate_count_by_dim: Dict[str, int]
    level3_candidate_count_per_level2: int = 10
    stats_by_dim: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def derive_semantic_topk_params(
    loader: ConceptLoader,
    dimensions_config: List[Dict[str, Any]],
    level3_candidate_count_per_level2: int = 10,
) -> SemanticTopKParams:
    """Derive conservative top-M/top-K values from the loaded taxonomy."""
    sheet_to_key = {item["sheet"]: item["key"] for item in dimensions_config}
    level2_counts: Dict[str, int] = {}
    stats_by_dim: Dict[str, Dict[str, Any]] = {}

    for sheet_name, category in loader.categories.items():
        dim_key = sheet_to_key.get(sheet_name, sheet_name)
        l3_counts: List[int] = []
        for level1 in category.level1_categories:
            for level2 in level1.children:
                if level2.level != 2:
                    continue
                level3_nodes = [
                    child for child in level2.children
                    if child.level == 3 and not _has_excluded_name(child.name)
                ]
                if level3_nodes:
                    l3_counts.append(len(level3_nodes))

        level2_count = len(l3_counts)
        level2_counts[dim_key] = level2_count
        stats_by_dim[dim_key] = {
            "sheet_name": sheet_name,
            "level2_count": level2_count,
            "level3_count": len(category.level3_categories),
            "l3_per_level2_min": min(l3_counts) if l3_counts else 0,
            "l3_per_level2_median": _percentile(l3_counts, 0.5),
            "l3_per_level2_p75": _percentile(l3_counts, 0.75),
            "l3_per_level2_p90": _percentile(l3_counts, 0.9),
            "l3_per_level2_max": max(l3_counts) if l3_counts else 0,
        }

    level2_candidate_count_by_dim = {
        dim_key: min(
            level2_count,
            max(8, min(24, int(round(math.sqrt(max(level2_count, 1)) * 2.2)))),
        )
        for dim_key, level2_count in level2_counts.items()
    }

    return SemanticTopKParams(
        level2_candidate_count_by_dim=level2_candidate_count_by_dim,
        level3_candidate_count_per_level2=level3_candidate_count_per_level2,
        stats_by_dim=stats_by_dim,
    )


class SemanticTopKSampler:
    """LLM-assisted sampler that selects level-2 and level-3 concepts together."""

    def __init__(
        self,
        loader: ConceptLoader,
        llm_provider,
        coverage_tracker: CoverageTracker,
        dimensions_config: List[Dict[str, Any]],
        params: SemanticTopKParams,
        batch_size: int = 5,
        max_retries: int = 2,
        challenge_elements: Optional[List[Dict[str, Any]]] = None,
    ):
        self.loader = loader
        self.llm = llm_provider
        self.coverage = coverage_tracker
        self.dimensions_config = sorted(dimensions_config, key=lambda d: d.get("core_priority", 99))
        self.params = params
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.challenge_elements = challenge_elements or []
        self.prompt_builder = ConceptSelectionPromptBuilder(loader, dimensions_config)
        self.key_to_sheet = {item["key"]: item["sheet"] for item in dimensions_config}
        self.sheet_to_key = {item["sheet"]: item["key"] for item in dimensions_config}
        self._buffers: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self.stats = {
            "llm_calls": 0,
            "llm_retries": 0,
            "combinations_generated": 0,
            "combinations_from_backfill": 0,
            "candidate_pool_stats": params.stats_by_dim,
            "level2_candidate_count_by_dim": params.level2_candidate_count_by_dim,
            "level3_candidate_count_per_level2": params.level3_candidate_count_per_level2,
            "sampling_mode": "semantic_topk_level2_level3",
        }

    def sample_combination(
        self,
        num_dimensions: int,
        target_challenge_count: int = 0,
        phase: str = "phase1",
    ) -> Optional[SampledCombination]:
        buffer_key = (self._normalize_num_dimensions(num_dimensions), max(0, target_challenge_count))
        buffer = self._buffers.setdefault(buffer_key, [])
        if not buffer:
            buffer.extend(self._refill_buffer(buffer_key[0], target_challenge_count, phase))
        if not buffer:
            return None
        return self._to_sampled_combination(buffer.pop(0))

    def _refill_buffer(
        self,
        num_dimensions: int,
        target_challenge_count: int,
        phase: str,
    ) -> List[Dict[str, Any]]:
        core_dim_key = self._next_core_dimension()
        if not core_dim_key:
            return []

        core_sheet = self.key_to_sheet[core_dim_key]
        core_concepts = self.coverage.get_next_core_concepts(core_sheet, self.batch_size)
        if not core_concepts:
            return []

        return self._build_items_for_core_nodes(
            core_dim_key=core_dim_key,
            core_concepts=core_concepts,
            num_dimensions=num_dimensions,
            target_challenge_count=target_challenge_count,
            phase=phase,
            include_fully_covered_candidates=False,
            mark_core_covered=True,
        )

    def sample_combination_from_anchor_pool(
        self,
        anchor_pool: Dict[str, List[ConceptNode]],
        num_dimensions: int,
        target_challenge_count: int = 0,
    ) -> Optional[SampledCombination]:
        """Phase 2: reuse covered concepts as anchors while preserving dimension count."""
        available_dims = [dim_key for dim_key, nodes in anchor_pool.items() if nodes]
        if not available_dims:
            return None

        anchor_dim = random.choice(available_dims)
        anchor_node = random.choice(anchor_pool[anchor_dim])
        items = self._build_items_for_core_nodes(
            core_dim_key=anchor_dim,
            core_concepts=[anchor_node],
            num_dimensions=num_dimensions,
            target_challenge_count=target_challenge_count,
            phase="phase2",
            include_fully_covered_candidates=True,
            mark_core_covered=True,
        )
        if not items:
            return None
        return self._to_sampled_combination(items[0])

    def _build_items_for_core_nodes(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        num_dimensions: int,
        target_challenge_count: int,
        phase: str,
        include_fully_covered_candidates: bool,
        mark_core_covered: bool,
    ) -> List[Dict[str, Any]]:
        core_sheet = self.key_to_sheet[core_dim_key]
        target_companion_count = max(0, self._normalize_num_dimensions(num_dimensions) - 1)
        candidate_dims = self._candidate_dimensions(core_dim_key)
        candidate_pool_by_dim = {
            dim_key: self._build_candidate_pool_for_dim(
                dim_key,
                include_fully_covered=include_fully_covered_candidates,
            )
            for dim_key in candidate_dims
        }
        usable_dims = [dim for dim, candidates in candidate_pool_by_dim.items() if candidates]
        if target_companion_count <= 0:
            return self._core_only_items(
                core_dim_key,
                core_concepts,
                phase,
                num_dimensions,
                mark_core_covered=mark_core_covered,
            )
        if len(usable_dims) < target_companion_count:
            return []

        selected_dims = random.sample(usable_dims, target_companion_count)
        candidate_pool = {dim: candidate_pool_by_dim[dim] for dim in selected_dims}
        actual_total_dimensions = 1 + len(selected_dims)

        prompt = self._build_planning_prompt(
            core_dim_key=core_dim_key,
            core_concepts=core_concepts,
            candidate_pool=candidate_pool,
            target_total_dimensions=actual_total_dimensions,
            target_challenge_count=target_challenge_count,
        )

        parsed_selections: List[Dict[str, Any]] = []
        raw_response = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self.llm.generate(prompt)
                self.stats["llm_calls"] += 1
                parsed_selections = self._parse_planning_response(
                    raw_response,
                    core_concepts,
                    selected_dims,
                    candidate_pool,
                    target_challenge_count,
                )
                if parsed_selections:
                    break
            except Exception as exc:
                self.stats["llm_retries"] += 1
                print(f"  [SemanticTopK] LLM call failed (attempt {attempt + 1}): {exc}")

        if not parsed_selections:
            self.stats["combinations_from_backfill"] += len(core_concepts)
            return []

        items: List[Dict[str, Any]] = []
        for selection in parsed_selections:
            core_node = self._find_level3_node(core_sheet, selection["core_level3"])
            if not core_node:
                continue

            if mark_core_covered:
                self.coverage.mark_covered(core_sheet, core_node.name, as_core=True)
            selected_nodes: Dict[str, ConceptNode] = {}
            expansion: Dict[str, Dict[str, Any]] = {}
            for dim_key, payload in selection["selections"].items():
                sheet_name = self.key_to_sheet[dim_key]
                node = self._find_level3_under_level2(
                    sheet_name,
                    payload["level2_path"],
                    payload["level3_name"],
                )
                if not node:
                    continue
                selected_nodes[dim_key] = node
                self.coverage.mark_covered(sheet_name, node.name, as_core=False)
                expansion[dim_key] = {
                    "selected_level2": payload["level2_name"],
                    "selected_level2_path": payload["level2_path"],
                    "selected_level3": node.name,
                    "selected_level3_path": list(node.path),
                    "reason": payload.get("reason", ""),
                    "policy": "llm_selected_level3_from_system_topk_candidates",
                }

            if len(selected_nodes) != len(selected_dims):
                continue

            selection_trace = {
                "phase": phase,
                "selection_mode": "semantic_topk_llm_selected_level2_and_level3",
                "core": self._core_trace(core_dim_key, core_node),
                "target_total_dimensions": actual_total_dimensions,
                "target_challenge_count": target_challenge_count,
                "candidate_dimensions": selected_dims,
                "candidate_pool_policy": {
                    "level2_candidate_count_by_dim": self.params.level2_candidate_count_by_dim,
                    "level3_candidate_count_per_level2": self.params.level3_candidate_count_per_level2,
                },
                "candidate_pool_counts": self._candidate_pool_counts(candidate_pool),
                "llm_selection": {
                    "provider": self.llm.__class__.__name__,
                    "model": getattr(self.llm, "model", ""),
                    "prompt_char_length": len(prompt),
                    "raw_response": raw_response,
                    "selected_level2": {
                        dim: payload["level2_name"]
                        for dim, payload in selection["selections"].items()
                    },
                    "selected_level3": {
                        dim: payload["level3_name"]
                        for dim, payload in selection["selections"].items()
                    },
                    "selected_challenges": selection["selected_challenges"],
                    "combination_reason": selection.get("combination_reason", ""),
                    "confidence": selection.get("confidence", 0.5),
                },
                "system_expansion": expansion,
            }

            items.append({
                "core_dim_key": core_dim_key,
                "core_concept": core_node,
                "selections": selected_nodes,
                "selected_challenges": selection["selected_challenges"],
                "selection_trace": selection_trace,
                "phase": phase,
            })
            self.stats["combinations_generated"] += 1

        return items

    def _build_candidate_pool_for_dim(
        self,
        dim_key: str,
        include_fully_covered: bool,
    ) -> List[Dict[str, Any]]:
        sheet_name = self.key_to_sheet[dim_key]
        l2_structure = self.prompt_builder.level2_structure.get(dim_key, {})
        entries: List[Dict[str, Any]] = []

        for level2_name, level2_info in l2_structure.items():
            level2_path = level2_info["path"]
            nodes = self._level3_nodes_under_level2(sheet_name, level2_path)
            if not nodes:
                continue
            uncovered = [
                node for node in nodes
                if self.coverage.get_times_covered(sheet_name, node.name) == 0
            ]
            if not include_fully_covered and not uncovered:
                continue
            shuffled_nodes = list(nodes)
            random.shuffle(shuffled_nodes)
            shuffled_nodes.sort(key=lambda node: self.coverage.get_times_covered(sheet_name, node.name))
            top_nodes = shuffled_nodes[:self.params.level3_candidate_count_per_level2]
            entries.append({
                "level2_name": level2_name,
                "level2_path": list(level2_path),
                "uncovered_level3_count": len(uncovered),
                "min_coverage": min(self.coverage.get_times_covered(sheet_name, node.name) for node in nodes),
                "level3_candidates": [
                    {
                        "name": node.name,
                        "path": list(node.path),
                        "times_covered": self.coverage.get_times_covered(sheet_name, node.name),
                    }
                    for node in top_nodes
                ],
                "_random": random.random(),
            })

        if not entries and not include_fully_covered:
            return self._build_candidate_pool_for_dim(dim_key, include_fully_covered=True)

        entries.sort(
            key=lambda item: (
                item["min_coverage"],
                -item["uncovered_level3_count"],
                item["_random"],
            )
        )
        limit = self.params.level2_candidate_count_by_dim.get(dim_key, 12)
        trimmed = entries[:limit]
        for item in trimmed:
            item.pop("_random", None)
        return trimmed

    def _build_planning_prompt(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        candidate_pool: Dict[str, List[Dict[str, Any]]],
        target_total_dimensions: int,
        target_challenge_count: int,
    ) -> str:
        payload = {
            "core_dimension": {
                "key": core_dim_key,
                "sheet_name": self.key_to_sheet[core_dim_key],
            },
            "target_total_dimensions": target_total_dimensions,
            "core_concepts": [
                {"name": node.name, "path": list(node.path)}
                for node in core_concepts
            ],
            "candidate_pool": candidate_pool,
            "challenge_target_count": target_challenge_count,
            "challenge_elements": self.challenge_elements,
        }
        return (
            "你是一个视频概念组合规划专家。当前任务不是生成最终视频 prompt，"
            "而是从真实类目树候选中选择语义兼容的概念组合。\n\n"
            "## 核心规则\n"
            "1. 系统只指定 companion 大类，没有预先固定二级类目。\n"
            "2. 你必须同时选择二级类目和该二级下的真实三级类目，二者都必须逐字来自候选池。\n"
            "3. 对每个 core_concepts 中的核心概念，只有在能为所有 companion dimension 选出完整兼容组合时才输出 object；找不到完整兼容组合时跳过该核心概念，不要输出空 object、空 selected_concepts 或缺少维度的对象。\n"
            "4. 每个 object 的 selected_concepts 必须且只能包含 candidate_pool 中列出的 companion dimension key，不能少选，也不能新增 candidate_pool、extra、other 或任何非维度字段。\n"
            "5. 不要输出候选池之外的二级、三级、叶子或外部百科事实。\n"
            "6. 必须按 core_concepts 的完整 path 理解核心概念，不能只根据名称局部联想；如果 path 表示食品、食材、材料、物品、身体部位或场所，就不能把它当成活体、人物或能自主行动的主体。\n"
            "7. 对每个候选组合必须做承载者兼容检查：运动、声音、交互和挑战性要素必须能由已选概念本身，或该概念语义直接允许的来源承担。\n"
            "8. 声音/说话类概念可以使用其语义直接包含的画外声源或说话者，但不能让这个声源变成新增画面主体，也不能用它承担未选中的视觉动作。\n"
            "9. 不要通过比喻、拟人、广告创意、外部常识或相似质感强行连接概念；例如食材的柔软、弹性或形状，不能作为选择体操、舞蹈、武术等自主运动的理由。\n"
            "10. combination_reason 必须解释核心概念与所有选中三级概念如何共同构成一个自然视频场面，只能引用 core_level3、已选 level2_name/level3_name 和已选挑战性要素；如果理由需要提到未选承载者，说明组合不应选择。\n"
            "11. 如果存在多个自然度接近的可行选择，先比较候选池中 level3_candidates 的 times_covered 字段，优先选择 times_covered 更低的三级概念；但不要为了覆盖率牺牲语义兼容性。\n"
            "12. 挑战性要素只有在最终 prompt 中能转化为可观察的视觉/听觉证据时才选择；如果无法想象具体落地方式，可以少选或不选。\n"
            "13. challenge reason 必须说明该挑战会通过什么可观察动作、声音、空间关系、物理变化或因果过程体现；不能只写“增强表现力”“更有动态感”等抽象理由。\n"
            "14. 挑战性要素必须基于 core_level3 与已选 level2_name/level3_name，不要绑定候选池外概念，也不要引用未被选中的候选概念。\n"
            "15. 保持中文语义，不要翻译、改写或泛化类目名称。\n"
            "16. 严格输出 JSON 数组，不要输出 Markdown、解释或代码块。\n\n"
            "## 输出格式\n"
            "[\n"
            "  {\n"
            "    \"core_level3\": \"核心概念名称\",\n"
            "    \"selected_concepts\": {\n"
            "      \"dimension_key\": {\n"
            "        \"level2_name\": \"候选池中的二级名称\",\n"
            "        \"level3_name\": \"该二级下候选池中的三级名称\",\n"
            "        \"reason\": \"为什么该三级与整体组合兼容，并说明其承载者不依赖未选概念\"\n"
            "      }\n"
            "    },\n"
            "    \"combination_reason\": \"核心概念 + 所有选中三级概念的整体兼容理由，不引用未选承载者\",\n"
            "    \"selected_challenges\": [\n"
            "      {\"id\": \"挑战性要素id\", \"reason\": \"基于已选概念的理由，并说明可观察落地方式\"}\n"
            "    ],\n"
            "    \"confidence\": 0.9\n"
            "  }\n"
            "]\n\n"
            "## 候选池 JSON\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _parse_planning_response(
        self,
        response: str,
        core_concepts: List[ConceptNode],
        selected_dims: List[str],
        candidate_pool: Dict[str, List[Dict[str, Any]]],
        target_challenge_count: int,
    ) -> List[Dict[str, Any]]:
        data = _extract_json_payload(response)
        if not isinstance(data, list):
            data = [data]

        core_map = {node.name: node for node in core_concepts}
        candidate_maps = self._candidate_maps(candidate_pool)
        results: List[Dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            core_name = str(item.get("core_level3", "")).strip()
            if core_name not in core_map:
                continue

            raw_selected = item.get("selected_concepts") or {}
            if not isinstance(raw_selected, dict):
                continue

            parsed_dims: Dict[str, Dict[str, Any]] = {}
            for dim_key in selected_dims:
                raw_payload = raw_selected.get(dim_key)
                if not isinstance(raw_payload, dict):
                    continue
                level2_name = str(raw_payload.get("level2_name", "")).strip()
                level3_name = str(raw_payload.get("level3_name", "")).strip()
                dim_map = candidate_maps.get(dim_key, {})
                level2_info = dim_map.get(level2_name)
                if not level2_info:
                    continue
                if level3_name not in level2_info["level3_names"]:
                    continue
                parsed_dims[dim_key] = {
                    "level2_name": level2_name,
                    "level2_path": list(level2_info["level2_path"]),
                    "level3_name": level3_name,
                    "reason": str(raw_payload.get("reason", "")).strip(),
                }

            if len(parsed_dims) != len(selected_dims):
                continue

            results.append({
                "core_level3": core_name,
                "selections": parsed_dims,
                "selected_challenges": self._match_challenges(
                    item.get("selected_challenges", []),
                    target_challenge_count,
                ),
                "combination_reason": str(item.get("combination_reason", "")).strip(),
                "confidence": item.get("confidence", 0.5),
            })

        return results

    def _match_challenges(self, raw_challenges: Any, target_count: int) -> List[Dict[str, Any]]:
        if not isinstance(raw_challenges, list) or target_count <= 0:
            return []
        by_id = {str(elem.get("id")): elem for elem in self.challenge_elements}
        by_name = {str(elem.get("name")): elem for elem in self.challenge_elements}
        selected: List[Dict[str, Any]] = []
        for raw in raw_challenges:
            if not isinstance(raw, dict):
                raw = {"id": str(raw)}
            elem = by_id.get(str(raw.get("id", "")).strip()) or by_name.get(str(raw.get("name", "")).strip())
            if not elem:
                continue
            matched = dict(elem)
            matched["reason"] = str(raw.get("reason", "")).strip()
            selected.append(matched)
            if len(selected) >= target_count:
                break
        return selected

    def _candidate_maps(self, candidate_pool: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for dim_key, entries in candidate_pool.items():
            result[dim_key] = {}
            for entry in entries:
                result[dim_key][entry["level2_name"]] = {
                    "level2_path": entry["level2_path"],
                    "level3_names": {
                        candidate["name"]
                        for candidate in entry.get("level3_candidates", [])
                    },
                }
        return result

    def _core_only_items(
        self,
        core_dim_key: str,
        core_concepts: List[ConceptNode],
        phase: str,
        target_total_dimensions: int,
        mark_core_covered: bool = True,
    ) -> List[Dict[str, Any]]:
        sheet_name = self.key_to_sheet[core_dim_key]
        items = []
        for core_node in core_concepts:
            if mark_core_covered:
                self.coverage.mark_covered(sheet_name, core_node.name, as_core=True)
            items.append({
                "core_dim_key": core_dim_key,
                "core_concept": core_node,
                "selections": {},
                "selected_challenges": [],
                "selection_trace": {
                    "phase": phase,
                    "selection_mode": "semantic_topk_core_only",
                    "core": self._core_trace(core_dim_key, core_node),
                    "target_total_dimensions": target_total_dimensions,
                },
                "phase": phase,
            })
            self.stats["combinations_generated"] += 1
        return items

    def _to_sampled_combination(self, item: Dict[str, Any]) -> SampledCombination:
        concepts: Dict[str, SampledConcept] = {}
        core_dim = item["core_dim_key"]
        core_node = item["core_concept"]
        core_sheet = self.key_to_sheet[core_dim]
        concepts[core_dim] = SampledConcept(
            sheet_name=core_sheet,
            level3_category=core_node.name,
            level3_path=list(core_node.path),
            leaf=self._pick_leaf(core_sheet, core_node.name),
        )
        for dim_key, node in item.get("selections", {}).items():
            sheet_name = self.key_to_sheet[dim_key]
            concepts[dim_key] = SampledConcept(
                sheet_name=sheet_name,
                level3_category=node.name,
                level3_path=list(node.path),
                leaf=self._pick_leaf(sheet_name, node.name),
            )
        return SampledCombination(
            concepts=concepts,
            challenge_elements=item.get("selected_challenges", []),
            selection_trace=item.get("selection_trace", {}),
            phase=item.get("phase", "phase1"),
        )

    def _next_core_dimension(self) -> Optional[str]:
        for dim in self.dimensions_config:
            sheet_name = dim["sheet"]
            if self.coverage.get_uncovered_count(sheet_name) > 0:
                return dim["key"]
        return None

    def _candidate_dimensions(self, core_dim_key: str) -> List[str]:
        return [
            item["key"] for item in self.dimensions_config
            if item["key"] != core_dim_key and item.get("companion", True)
        ]

    def _normalize_num_dimensions(self, num_dimensions: int) -> int:
        return max(1, min(num_dimensions, len(self.dimensions_config)))

    def _level3_nodes_under_level2(self, sheet_name: str, level2_path: List[str]) -> List[ConceptNode]:
        nodes = []
        for node in self.loader.get_level3_categories(sheet_name):
            if len(node.path) < 2 or len(level2_path) < 2:
                continue
            if node.path[0] != level2_path[0] or node.path[1] != level2_path[1]:
                continue
            if _has_excluded_name(node.name):
                continue
            nodes.append(node)
        return nodes

    def _find_level3_under_level2(
        self,
        sheet_name: str,
        level2_path: List[str],
        level3_name: str,
    ) -> Optional[ConceptNode]:
        for node in self._level3_nodes_under_level2(sheet_name, level2_path):
            if node.name == level3_name:
                return node
        return None

    def _find_level3_node(self, sheet_name: str, level3_name: str) -> Optional[ConceptNode]:
        for node in self.loader.get_level3_categories(sheet_name):
            if node.name == level3_name:
                return node
        return None

    def _pick_leaf(self, sheet_name: str, level3_name: str) -> Optional[str]:
        leaves = self.loader.get_leaves_under_level3(sheet_name, level3_name)
        valid = [leaf for leaf in leaves if not _has_excluded_name(leaf.name)]
        return random.choice(valid).name if valid else None

    def _core_trace(self, core_dim_key: str, core_node: ConceptNode) -> Dict[str, Any]:
        return {
            "dimension": core_dim_key,
            "sheet_name": self.key_to_sheet.get(core_dim_key, core_dim_key),
            "level3": core_node.name,
            "full_path": list(core_node.path),
        }

    def _candidate_pool_counts(self, candidate_pool: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
        return {
            dim_key: {
                "level2_count": len(entries),
                "level3_count": sum(len(entry.get("level3_candidates", [])) for entry in entries),
            }
            for dim_key, entries in candidate_pool.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "coverage": self.coverage.get_report(),
            "buffer_size": sum(len(buffer) for buffer in self._buffers.values()),
        }


def _extract_json_payload(text: str) -> Any:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", stripped, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _percentile(values: List[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = int(round((len(values) - 1) * fraction))
    return values[index]


def _has_excluded_name(name: str) -> bool:
    return any(keyword in str(name) for keyword in EXCLUDE_KEYWORDS)
