#!/usr/bin/env python3
"""Regenerate prompts that failed LLM-as-judge review.

Default mode is text-only repair: keep the original concept combination and
regenerate only the prompt text. Concept-repair mode is available for hard
cases that still fail after text-only regeneration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combiner import DifficultyManager, PromptCombination
from src.config import load_config
from src.generator import PromptGenerator
from src.judge import (
    OpenAICompatibleJudgeProvider,
    parse_json_object,
    write_judge_report,
)
from src.models import SampledConcept
from src.output import OutputWriter


PASS_DECISIONS = {"PASS", "PASS_WITH_MINOR_ISSUES"}
FAIL_DECISIONS = {"FAIL"}
DIMENSION_SHEETS = {
    "subject": "主体",
    "motion": "运动",
    "scene": "场景",
    "audio": "音频类型",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate judge-failed prompts.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config YAML path")
    parser.add_argument("--source", required=True, help="Source prompt JSON file")
    parser.add_argument("--judge", required=True, help="Judge report JSON file for the source prompts")
    parser.add_argument("--repair-mode", choices=["text", "concept"], default="text")
    parser.add_argument("--regen-llm", default=None, help="LLM provider for regeneration; default inferred from source")
    parser.add_argument("--round", type=int, default=1, help="Revision round number")
    parser.add_argument("--regen-output", default=None, help="Output JSON for regenerated prompts only")
    parser.add_argument("--merged-output", default=None, help="Output JSON for pass/minor prompts plus regenerated prompts")
    parser.add_argument("--carryover-judge-output", default=None, help="Initial judge report with carried-over pass/minor results")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not call LLM or write outputs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of failed prompts to regenerate")
    parser.add_argument("--pool-size-per-category", type=int, default=160, help="Concept repair candidate cap per category")
    args = parser.parse_args()

    config = load_config(args.config)
    source_path = resolve_path(args.source)
    judge_path = resolve_path(args.judge)
    source_data = load_json(source_path)
    judge_data = load_json(judge_path)

    source_prompts = source_data.get("prompts", [])
    prompt_by_id = {prompt.get("prompt_id"): prompt for prompt in source_prompts}
    judge_results = judge_data.get("results", [])
    judge_by_id = {
        result.get("prompt_id"): result
        for result in judge_results
        if result.get("prompt_id")
    }

    failed_ids = [
        prompt_id
        for prompt_id, result in judge_by_id.items()
        if get_decision(result) in FAIL_DECISIONS and prompt_id in prompt_by_id
    ]
    if args.limit > 0:
        failed_ids = failed_ids[:args.limit]

    carryover_ids = [
        prompt.get("prompt_id")
        for prompt in source_prompts
        if get_decision(judge_by_id.get(prompt.get("prompt_id"), {})) in PASS_DECISIONS
    ]
    unjudged_ids = [
        prompt.get("prompt_id")
        for prompt in source_prompts
        if prompt.get("prompt_id") not in judge_by_id
    ]

    regen_llm = args.regen_llm or infer_regen_llm(source_prompts, source_path)
    default_paths = build_default_paths(source_path, judge_path, config.get("judge", {}), args.repair_mode, args.round)
    regen_output = resolve_path(args.regen_output) if args.regen_output else default_paths["regen_output"]
    merged_output = resolve_path(args.merged_output) if args.merged_output else default_paths["merged_output"]
    carryover_judge_output = (
        resolve_path(args.carryover_judge_output)
        if args.carryover_judge_output
        else default_paths["carryover_judge_output"]
    )

    print("[Regenerate failed prompts]")
    print(f"  source: {source_path}")
    print(f"  judge: {judge_path}")
    print(f"  repair_mode: {args.repair_mode}")
    print(f"  regen_llm: {regen_llm}")
    print(f"  total_source_prompts: {len(source_prompts)}")
    print(f"  pass_or_minor_carryover: {len(carryover_ids)}")
    print(f"  unjudged_kept_for_later: {len(unjudged_ids)}")
    print(f"  failed_to_regenerate: {len(failed_ids)}")
    print(f"  regen_output: {regen_output}")
    print(f"  merged_output: {merged_output}")
    print(f"  carryover_judge_output: {carryover_judge_output}")

    if args.dry_run:
        print(f"  preview_failed_ids: {failed_ids[:20]}")
        return

    if not failed_ids:
        print("  No failed prompts found; nothing to regenerate.")
        return

    generator = PromptGenerator(config.llm_providers, [regen_llm])
    if regen_llm not in generator.providers:
        provider_cfg = config.llm_providers.get(regen_llm, {})
        api_key_env = provider_cfg.get("api_key_env", "")
        raise ValueError(f"Regeneration provider {regen_llm} unavailable. Check {api_key_env}.")

    difficulty_manager = DifficultyManager(config.difficulty_distribution, config.difficulty_params)
    concept_pool = build_failed_concept_pool([prompt_by_id[prompt_id] for prompt_id in failed_ids])
    selector = None
    if args.repair_mode == "concept":
        selector = OpenAICompatibleJudgeProvider(config.get("judge", {}))
        if not selector.is_available():
            raise ValueError(
                f"Concept repair selector unavailable. Please set {selector.api_key_env}."
            )

    regenerated_prompts = []
    for index, prompt_id in enumerate(failed_ids, start=1):
        source_prompt = prompt_by_id[prompt_id]
        judge_result = judge_by_id[prompt_id]
        if args.repair_mode == "text":
            repaired_prompt = regenerate_text_only(
                source_prompt,
                judge_result,
                generator,
                difficulty_manager,
                regen_llm,
                args.round,
                source_path,
                judge_path,
            )
        else:
            repaired_prompt = regenerate_with_concept_repair(
                source_prompt,
                judge_result,
                generator,
                difficulty_manager,
                regen_llm,
                args.round,
                source_path,
                judge_path,
                concept_pool,
                selector,
                args.pool_size_per_category,
            )
        regenerated_prompts.append(repaired_prompt)
        if index % 10 == 0 or index == len(failed_ids):
            print(f"  regenerated {index}/{len(failed_ids)}")

    write_prompt_file(
        regen_output,
        regenerated_prompts,
        {
            "kind": "regenerated_failed_prompts",
            "source": str(source_path),
            "judge": str(judge_path),
            "repair_mode": args.repair_mode,
            "revision_round": args.round,
            "regen_llm": regen_llm,
        },
    )

    merged_prompts = []
    regenerated_by_source_id = {
        prompt["revision"]["source_prompt_id"]: prompt for prompt in regenerated_prompts
    }
    for prompt in source_prompts:
        prompt_id = prompt.get("prompt_id")
        if prompt_id in regenerated_by_source_id:
            merged_prompts.append(regenerated_by_source_id[prompt_id])
        elif prompt_id not in regenerated_by_source_id:
            merged_prompts.append(prompt)

    write_prompt_file(
        merged_output,
        merged_prompts,
        {
            "kind": "merged_after_regeneration",
            "source": str(source_path),
            "judge": str(judge_path),
            "repair_mode": args.repair_mode,
            "revision_round": args.round,
            "regen_llm": regen_llm,
            "replacement_policy": "replace_failed_prompts",
            "carried_over_prompts": len(carryover_ids),
            "unjudged_prompts_kept": len(unjudged_ids),
            "regenerated_prompts": len(regenerated_prompts),
        },
    )

    carryover_results = []
    for prompt_id in carryover_ids:
        result = dict(judge_by_id[prompt_id])
        result["judge_result_origin"] = "carried_over"
        result["carried_from"] = str(judge_path)
        carryover_results.append(result)

    write_judge_report(
        carryover_judge_output,
        carryover_results,
        {
            "path": str(merged_output),
            "base_source": str(source_path),
            "base_judge": str(judge_path),
            "selected_prompts": len(merged_prompts),
            "carried_over_results": len(carryover_results),
            "pending_regenerated_results": len(regenerated_prompts),
        },
        {
            "provider": config.get("judge.provider", "glm"),
            "model": config.get("judge.model", "glm-5.1"),
            "base_url": config.get("judge.base_url", ""),
            "judge_version": config.get("judge.judge_version", "v1"),
            "initialized_from": str(judge_path),
            "initialization": "carry_over_pass_and_minor_only",
        },
    )

    print("  wrote regenerated prompts, merged prompts, and carry-over judge report")


def regenerate_text_only(
    source_prompt: Dict[str, Any],
    judge_result: Dict[str, Any],
    generator: PromptGenerator,
    difficulty_manager: DifficultyManager,
    regen_llm: str,
    revision_round: int,
    source_path: Path,
    judge_path: Path,
) -> Dict[str, Any]:
    combination = reconstruct_combination(source_prompt, difficulty_manager)
    generated = generate_one(generator, combination, regen_llm)
    return build_revised_prompt(
        source_prompt,
        judge_result,
        generated,
        revision_round,
        "text_regen",
        source_path,
        judge_path,
        new_concepts=combination.concepts,
        concept_selection_reason="Text-only regeneration keeps the original mandatory concepts.",
    )


def regenerate_with_concept_repair(
    source_prompt: Dict[str, Any],
    judge_result: Dict[str, Any],
    generator: PromptGenerator,
    difficulty_manager: DifficultyManager,
    regen_llm: str,
    revision_round: int,
    source_path: Path,
    judge_path: Path,
    concept_pool: Dict[str, List[Dict[str, Any]]],
    selector: OpenAICompatibleJudgeProvider,
    pool_size_per_category: int,
) -> Dict[str, Any]:
    anchor_category, anchor_concept = choose_anchor_concept(source_prompt, judge_result)
    selected_concepts, reason = select_repaired_concepts(
        selector,
        source_prompt,
        judge_result,
        anchor_category,
        anchor_concept,
        concept_pool,
        pool_size_per_category,
    )
    combination = reconstruct_combination(
        {
            **source_prompt,
            "combination_id": f"{source_prompt.get('combination_id')}-C{revision_round}",
            "sampling": {
                **(source_prompt.get("sampling") or {}),
                "categories_selected": list(selected_concepts.keys()),
                "concepts": selected_concepts,
            },
        },
        difficulty_manager,
    )
    generated = generate_one(generator, combination, regen_llm)
    return build_revised_prompt(
        source_prompt,
        judge_result,
        generated,
        revision_round,
        "concept_repair_regen",
        source_path,
        judge_path,
        new_concepts=combination.concepts,
        anchor={
            "category": anchor_category,
            "concept": anchor_concept,
        },
        concept_selection_reason=reason,
    )


def reconstruct_combination(prompt: Dict[str, Any], difficulty_manager: DifficultyManager) -> PromptCombination:
    sampling = prompt.get("sampling") or {}
    concepts = {
        category: sampled_concept_from_dict(category, concept)
        for category, concept in (sampling.get("concepts") or {}).items()
    }
    difficulty_level = str((prompt.get("difficulty") or {}).get("level", "MEDIUM")).lower()
    return PromptCombination(
        combination_id=prompt.get("combination_id") or sampling.get("combination_id", ""),
        concepts=concepts,
        difficulty_level=difficulty_level,
        difficulty_params=difficulty_manager.get_params(difficulty_level),
        challenge_elements=sampling.get("challenge_elements") or [],
    )


def sampled_concept_from_dict(category: str, data: Dict[str, Any]) -> SampledConcept:
    raw_path = data.get("path")
    full_path = data.get("full_path") or ""
    if isinstance(raw_path, list) and raw_path:
        level3_path = [str(part).strip() for part in raw_path]
    else:
        level3_path = []
    if not level3_path and full_path:
        level3_path = [part.strip() for part in full_path.split(">")]
    if not level3_path:
        level3_path = [
            data.get("level1_category", ""),
            data.get("level2_category", ""),
            data.get("level3_category", ""),
        ]
    level3_path = [part for part in level3_path if part]
    if not level3_path and data.get("level3_category"):
        level3_path = [data["level3_category"]]
    return SampledConcept(
        sheet_name=DIMENSION_SHEETS.get(category, category),
        level3_category=data.get("level3_category", ""),
        level3_path=level3_path,
        leaf=data.get("leaf"),
    )


def generate_one(generator: PromptGenerator, combination: PromptCombination, regen_llm: str):
    generated = generator.generate(combination)
    selected = [item for item in generated if item.llm_provider == regen_llm]
    if not selected:
        raise RuntimeError(f"No prompt generated by {regen_llm} for {combination.combination_id}")
    return selected[0]


def build_revised_prompt(
    source_prompt: Dict[str, Any],
    judge_result: Dict[str, Any],
    generated,
    revision_round: int,
    repair_type: str,
    source_path: Path,
    judge_path: Path,
    new_concepts: Dict[str, SampledConcept],
    anchor: Optional[Dict[str, Any]] = None,
    concept_selection_reason: str = "",
) -> Dict[str, Any]:
    source_prompt_id = source_prompt.get("prompt_id")
    suffix = f"R{revision_round}" if repair_type == "text_regen" else f"C{revision_round}"
    revised = generated.to_dict()
    revised["prompt_id"] = f"{source_prompt_id}-{suffix}"
    revised["combination_id"] = generated.combination_id
    revised["revision"] = {
        "is_revision": True,
        "revision_round": revision_round,
        "repair_type": repair_type,
        "source_prompt_id": source_prompt_id,
        "source_combination_id": source_prompt.get("combination_id"),
        "source_llm": source_prompt.get("llm", {}),
        "source_judge_file": str(judge_path),
        "source_judge_version": get_nested(judge_result, ["llm_judge", "judge_version"]),
        "source_judge_decision": get_decision(judge_result),
        "source_judge_issue_types": get_issue_types(judge_result),
        "created_from": str(source_path),
        "anchor_concept": anchor,
        "old_concepts": (source_prompt.get("sampling") or {}).get("concepts", {}),
        "new_concepts": {key: value.to_dict() for key, value in new_concepts.items()},
        "concept_selection_reason": concept_selection_reason,
    }
    return revised


def build_failed_concept_pool(failed_prompts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pool = {category: OrderedDict() for category in DIMENSION_SHEETS}
    for prompt in failed_prompts:
        concepts = (prompt.get("sampling") or {}).get("concepts") or {}
        for category, concept in concepts.items():
            key = concept_key(concept)
            pool.setdefault(category, OrderedDict())[key] = concept
    return {
        category: list(items.values())
        for category, items in pool.items()
    }


def choose_anchor_concept(prompt: Dict[str, Any], judge_result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    concepts = (prompt.get("sampling") or {}).get("concepts") or {}
    checks = (judge_result.get("llm_judge") or {}).get("concept_checks") or []
    for status in ["missing", "wrong", "weak"]:
        for check in checks:
            category = check.get("category")
            if check.get("status") == status and category in concepts:
                return category, concepts[category]

    for issue in (judge_result.get("llm_judge") or {}).get("issues") or []:
        location = str(issue.get("location", "")).lower()
        issue_type = str(issue.get("type", "")).lower()
        for category in concepts:
            if category in location or category in issue_type:
                return category, concepts[category]

    first_category = next(iter(concepts))
    return first_category, concepts[first_category]


def select_repaired_concepts(
    selector: OpenAICompatibleJudgeProvider,
    source_prompt: Dict[str, Any],
    judge_result: Dict[str, Any],
    anchor_category: str,
    anchor_concept: Dict[str, Any],
    concept_pool: Dict[str, List[Dict[str, Any]]],
    pool_size_per_category: int,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    difficulty = str((source_prompt.get("difficulty") or {}).get("level", "MEDIUM")).upper()
    target_categories = target_categories_for_difficulty(
        difficulty,
        list(((source_prompt.get("sampling") or {}).get("concepts") or {}).keys()),
        anchor_category,
    )
    candidate_pool = {
        category: trim_pool_with_anchor(concepts, anchor_concept, pool_size_per_category)
        for category, concepts in concept_pool.items()
        if category in target_categories
    }
    candidate_pool[anchor_category] = trim_pool_with_anchor(
        candidate_pool.get(anchor_category, []),
        anchor_concept,
        pool_size_per_category,
    )

    system_prompt = (
        "You are a strict taxonomy-constrained concept repair selector. "
        "Select a more compatible concept combination without inventing concepts."
    )
    user_prompt = f"""Repair the failed video prompt by selecting concepts from the provided failed-concept pool.

Rules:
- Preserve the anchor concept exactly.
- Select only from candidate_pool. Do not invent, rename, merge, or generalize concepts.
- Use exactly these categories: {target_categories}.
- Choose companion concepts that make the anchor concept easier to express explicitly.
- Avoid the exact original failed combination if a more coherent combination exists.
- Output valid JSON only.

Input:
{json.dumps({
    "prompt_id": source_prompt.get("prompt_id"),
    "difficulty": difficulty,
    "target_categories": target_categories,
    "anchor_category": anchor_category,
    "anchor_concept": anchor_concept,
    "original_concepts": (source_prompt.get("sampling") or {}).get("concepts", {}),
    "judge_issues": (judge_result.get("llm_judge") or {}).get("issues", []),
    "candidate_pool": candidate_pool,
}, ensure_ascii=False, indent=2)}

Required output schema:
{{
  "selected_concepts": {{
    "subject": {{"level1_category": "...", "level2_category": "...", "level3_category": "...", "path": ["...", "...", "..."], "leaf": null}},
    "motion": {{"level1_category": "...", "level2_category": "...", "level3_category": "...", "path": ["...", "...", "..."], "leaf": "..."}}
  }},
  "reason": "brief reason"
}}
"""
    raw = selector.generate(system_prompt, user_prompt)
    parsed = parse_json_object(raw)
    selected = parsed.get("selected_concepts") or {}
    reason = parsed.get("reason", "")
    validated = validate_selected_concepts(
        selected,
        candidate_pool,
        target_categories,
        anchor_category,
        anchor_concept,
    )
    return validated, reason


def target_categories_for_difficulty(difficulty: str, original_categories: List[str], anchor_category: str) -> List[str]:
    if difficulty == "LOW":
        target_count = min(max(len(original_categories), 1), 2)
    elif difficulty == "MEDIUM":
        target_count = min(max(len(original_categories), 3), 3)
    else:
        target_count = min(max(len(original_categories), 4), 4)

    ordered = [anchor_category] + [cat for cat in original_categories if cat != anchor_category]
    for category in ["subject", "motion", "scene", "audio"]:
        if category not in ordered:
            ordered.append(category)
    return ordered[:target_count]


def trim_pool_with_anchor(
    concepts: List[Dict[str, Any]],
    anchor_concept: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    seen = OrderedDict()
    seen[concept_key(anchor_concept)] = anchor_concept
    for concept in concepts:
        seen[concept_key(concept)] = concept
        if limit > 0 and len(seen) >= limit:
            break
    return list(seen.values())


def validate_selected_concepts(
    selected: Dict[str, Dict[str, Any]],
    candidate_pool: Dict[str, List[Dict[str, Any]]],
    target_categories: List[str],
    anchor_category: str,
    anchor_concept: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    validated = {}
    for category in target_categories:
        concept = selected.get(category)
        if not concept:
            raise ValueError(f"Concept repair missing required category: {category}")
        candidates = {
            concept_key(candidate): candidate
            for candidate in candidate_pool.get(category, [])
        }
        key = concept_key(concept)
        if key not in candidates:
            raise ValueError(f"Concept repair selected taxonomy-outside concept for {category}: {concept}")
        if category == anchor_category and key != concept_key(anchor_concept):
            raise ValueError(f"Concept repair changed the anchor concept for {anchor_category}.")
        validated[category] = candidates[key]
    return validated


def get_decision(judge_result: Dict[str, Any]) -> str:
    return str((judge_result.get("llm_judge") or {}).get("overall_decision", "")).upper()


def get_issue_types(judge_result: Dict[str, Any]) -> List[str]:
    return [
        issue.get("type", "unknown")
        for issue in (judge_result.get("llm_judge") or {}).get("issues", [])
    ]


def infer_regen_llm(source_prompts: List[Dict[str, Any]], source_path: Path) -> str:
    for prompt in source_prompts:
        provider = (prompt.get("llm") or {}).get("provider")
        if provider:
            return provider
    stem = source_path.stem.lower()
    if "gpt4o" in stem:
        return "gpt"
    if "dpsk" in stem:
        return "dpsk"
    return "gemini"


def build_default_paths(
    source_path: Path,
    judge_path: Path,
    judge_cfg: Dict[str, Any],
    repair_mode: str,
    revision_round: int,
) -> Dict[str, Path]:
    revisions_dir = ROOT / "output" / "revisions"
    judge_dir = ROOT / "reports" / "judge"
    mode_slug = "text" if repair_mode == "text" else "concept"
    source_stem = source_path.stem
    regen_output = revisions_dir / f"{source_stem}_regen_{mode_slug}_r{revision_round}.json"
    merged_output = revisions_dir / f"{source_stem}_merged_after_{mode_slug}_r{revision_round}.json"
    judge_provider = judge_cfg.get("provider", "glm")
    judge_version = judge_cfg.get("judge_version", "v1")
    judge_model_label = merged_output.stem.replace("_prompts", "")
    carryover = judge_dir / f"{judge_model_label}_{judge_provider}_{judge_version}" / "full.json"
    return {
        "regen_output": regen_output,
        "merged_output": merged_output,
        "carryover_judge_output": carryover,
    }


def write_prompt_file(path: Path, prompts: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    writer = OutputWriter(str(path))
    for prompt in prompts:
        writer.add_prompt(prompt)
    writer.set_stats({"metadata": metadata})
    writer.write()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def concept_key(concept: Dict[str, Any]) -> str:
    path = concept.get("path")
    if isinstance(path, list):
        path_key = " > ".join(str(part) for part in path)
    else:
        path_key = str(concept.get("full_path", ""))
    return "|".join([
        path_key,
        str(concept.get("level3_category", "")),
        str(concept.get("leaf", "")),
    ])


def get_nested(data: Dict[str, Any], keys: List[str]) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    main()
