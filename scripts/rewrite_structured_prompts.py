#!/usr/bin/env python3
"""Rewrite a random 30% subset of existing prompts into structured template families.

This is a post-processing step on top of the existing prompt-generation pipeline.
It keeps the original concepts and challenge elements, but rewrites the text into
one of the extracted structured families:
shot_timeline, split_screen_explanation, field_spec, script_timeline.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.generator import PromptGenerator
from src.judge import load_prompt_file
from src.output import OutputWriter
from src.structured_templates import (
    STRUCTURED_TEMPLATE_IDS,
    build_structured_rewrite_prompt,
    choose_structured_family,
    get_prompt_text,
    validate_structured_output,
)
from src.text_metrics import count_chinese_chars


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a subset of prompts into structured families.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config YAML path")
    parser.add_argument("--source", required=True, help="Source prompt JSON file")
    parser.add_argument("--output", default=None, help="Output JSON file for structured rewrite")
    parser.add_argument("--ratio", type=float, default=None, help="Rewrite ratio, default 0.3")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--rewrite-llm", default=None, help="Preferred LLM provider for rewrite")
    parser.add_argument("--round", type=int, default=1, help="Rewrite round number")
    parser.add_argument("--limit", type=int, default=0, help="Cap the number of prompts to rewrite")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for each rewrite prompt")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not call LLM or write outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    source_path = resolve_path(args.source)
    source_prompts = load_prompt_file(source_path)

    ratio = args.ratio if args.ratio is not None else config.get("structured_rewrite.ratio", 0.3)
    ratio = max(0.0, min(1.0, float(ratio)))
    seed = args.seed if args.seed is not None else config.get("structured_rewrite.seed", 42)
    preferred_provider = args.rewrite_llm or config.get("structured_rewrite.provider", "")

    output_path = resolve_path(args.output) if args.output else default_output_path(source_path, args.round)

    eligible_indices = [
        idx for idx, prompt in enumerate(source_prompts)
        if normalize_template_family(prompt) == "plain_paragraph"
    ]
    target_count = int(round(len(eligible_indices) * ratio))
    if ratio > 0 and eligible_indices and target_count <= 0:
        target_count = 1
    if args.limit > 0:
        target_count = min(target_count, args.limit)
    target_count = min(target_count, len(eligible_indices))

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(eligible_indices, target_count)) if target_count > 0 else []

    provider_bank = PromptGenerator(
        config.llm_providers,
        list(config.llm_providers.keys()),
        dimensions_config=config.dimensions,
    )
    available_provider_names = list(provider_bank.providers.keys())
    if not available_provider_names:
        raise ValueError("No available LLM providers found for structured rewrite.")

    print("[Structured rewrite]")
    print(f"  source: {source_path}")
    print(f"  output: {output_path}")
    print(f"  source_prompts: {len(source_prompts)}")
    print(f"  eligible_prompts: {len(eligible_indices)}")
    print(f"  target_ratio: {ratio:.2f}")
    print(f"  target_count: {target_count}")
    print(f"  seed: {seed}")
    print(f"  preferred_provider: {preferred_provider or '(auto)'}")
    print(f"  dry_run: {args.dry_run}")

    if args.dry_run:
        preview = [source_prompts[idx].get("prompt_id") for idx in selected_indices[:20]]
        print(f"  selected_prompt_ids_preview: {preview}")
        return

    rewritten_prompts: List[Dict[str, Any]] = []
    rewrite_stats = {
        "kind": "structured_rewrite",
        "source": str(source_path),
        "output": str(output_path),
        "rewrite_round": args.round,
        "rewrite_ratio": ratio,
        "seed": seed,
        "requested_rewrite_count": target_count,
        "eligible_prompt_count": len(eligible_indices),
        "selected_prompt_ids": [],
        "rewritten_prompt_ids": [],
        "failed_prompt_ids": [],
        "validation_failed_prompt_ids": [],
        "target_family_distribution": {family_id: 0 for family_id in STRUCTURED_TEMPLATE_IDS},
        "family_distribution": {family_id: 0 for family_id in STRUCTURED_TEMPLATE_IDS},
        "provider_distribution": {},
    }

    selected_set = set(selected_indices)
    for idx, prompt in enumerate(source_prompts):
        prompt_record = copy.deepcopy(prompt)
        source_text = get_prompt_text(prompt_record)
        original_length = count_chinese_chars(source_text)

        if idx not in selected_set:
            prompt_record["template_family"] = normalize_template_family(prompt_record)
            rewritten_prompts.append(prompt_record)
            continue

        family_id, scores, reason = choose_structured_family(prompt_record)
        rewrite_stats["selected_prompt_ids"].append(prompt_record.get("prompt_id"))
        rewrite_stats["target_family_distribution"][family_id] = (
            rewrite_stats["target_family_distribution"].get(family_id, 0) + 1
        )

        provider_name = resolve_provider_name(
            prompt_record,
            provider_bank.providers,
            preferred_provider,
            available_provider_names,
        )
        provider = provider_bank.providers[provider_name]
        rewrite_stats["provider_distribution"][provider_name] = rewrite_stats["provider_distribution"].get(provider_name, 0) + 1

        target_length_hint = build_target_length_hint(original_length, prompt_record)
        rewrite_prompt = build_structured_rewrite_prompt(prompt_record, family_id, target_length_hint)
        current_rewrite_prompt = rewrite_prompt

        rewritten_text = None
        last_error: Optional[Exception] = None
        validation_errors: List[str] = []
        invalid_output = ""
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                response = provider.generate(current_rewrite_prompt)
                candidate_text = clean_rewrite_output(response)
                is_valid, validation_errors = validate_structured_output(candidate_text, family_id)
                if is_valid:
                    rewritten_text = candidate_text
                    break

                invalid_output = candidate_text
                last_error = ValueError("; ".join(validation_errors))
                if attempt < max(1, args.max_retries):
                    current_rewrite_prompt = build_validation_retry_prompt(
                        rewrite_prompt,
                        family_id,
                        validation_errors,
                        invalid_output,
                    )
                    time.sleep(min(2 ** (attempt - 1), 4))
            except Exception as exc:
                last_error = exc
                if attempt < max(1, args.max_retries):
                    time.sleep(min(2 ** (attempt - 1), 4))

        if not rewritten_text:
            prompt_record["template_family"] = "plain_paragraph"
            prompt_record["rewrite"] = {
                "is_revision": False,
                "repair_type": "structured_rewrite",
                "status": "failed",
                "source_prompt_id": prompt_record.get("prompt_id"),
                "source_text": source_text,
                "target_template_family": family_id,
                "template_family_reason": reason,
                "family_scores": scores,
                "rewrite_llm": {"provider": provider_name, "model": provider.model},
                "rewrite_round": args.round,
                "rewrite_ratio": ratio,
                "validation_errors": validation_errors,
                "invalid_output": invalid_output,
                "error": str(last_error) if last_error else "",
            }
            rewritten_prompts.append(prompt_record)
            rewrite_stats["failed_prompt_ids"].append(prompt_record.get("prompt_id"))
            if validation_errors:
                rewrite_stats["validation_failed_prompt_ids"].append(prompt_record.get("prompt_id"))
            continue

        prompt_record["text"] = rewritten_text
        prompt_record.pop("prompt", None)
        prompt_record["text_length"] = count_chinese_chars(rewritten_text)
        prompt_record["template_family"] = family_id
        prompt_record["revision"] = {
            "is_revision": True,
            "revision_round": args.round,
            "repair_type": "structured_rewrite",
            "source_prompt_id": prompt_record.get("prompt_id"),
            "source_llm": prompt_record.get("llm", {}),
            "source_text": source_text,
            "source_text_length": original_length,
            "target_template_family": family_id,
            "template_family_reason": reason,
            "family_scores": scores,
            "created_from": str(source_path),
            "rewrite_llm": {"provider": provider_name, "model": provider.model},
            "rewrite_ratio": ratio,
        }
        rewritten_prompts.append(prompt_record)
        rewrite_stats["rewritten_prompt_ids"].append(prompt_record.get("prompt_id"))
        rewrite_stats["family_distribution"][family_id] = rewrite_stats["family_distribution"].get(family_id, 0) + 1

    output_writer = OutputWriter(str(output_path))
    for prompt in rewritten_prompts:
        output_writer.add_prompt(prompt)
    output_writer.set_stats(rewrite_stats)
    output_writer.write()

    print(
        f"  rewritten: {len(rewrite_stats['rewritten_prompt_ids'])} / {len(selected_indices)} "
        f"(failed: {len(rewrite_stats['failed_prompt_ids'])})"
    )
    print(f"  output_written: {output_path}")


def build_validation_retry_prompt(
    base_prompt: str,
    family_id: str,
    validation_errors: List[str],
    invalid_output: str,
) -> str:
    """Build a stricter retry prompt after deterministic structure validation fails."""
    errors = "\n".join(f"- {error}" for error in validation_errors) or "- 结构化格式不合格"
    return "\n".join([
        base_prompt,
        "",
        "上一次输出未通过结构化格式校验，不能接受为结构化 prompt。",
        f"目标模板族：{family_id}",
        "校验失败原因：",
        errors,
        "",
        "上一次不合格输出：",
        invalid_output,
        "",
        "请重新输出。必须修正上述结构问题，只输出最终改写后的 prompt 文本：",
    ])


def normalize_template_family(prompt_record: Dict[str, Any]) -> str:
    family = prompt_record.get("template_family")
    if isinstance(family, str) and family in STRUCTURED_TEMPLATE_IDS:
        return family
    revision = prompt_record.get("revision") or {}
    family = revision.get("target_template_family") or revision.get("template_family")
    if isinstance(family, str) and family in STRUCTURED_TEMPLATE_IDS:
        return family
    rewrite = prompt_record.get("rewrite") or {}
    family = rewrite.get("target_template_family") or rewrite.get("template_family")
    if isinstance(family, str) and family in STRUCTURED_TEMPLATE_IDS:
        return family
    return "plain_paragraph"


def resolve_provider_name(
    prompt_record: Dict[str, Any],
    providers: Dict[str, Any],
    preferred_provider: str,
    available_provider_names: List[str],
) -> str:
    source_provider = ((prompt_record.get("llm") or {}).get("provider") or "").strip()
    candidates = [
        preferred_provider.strip() if preferred_provider else "",
        source_provider,
    ]
    for candidate in candidates:
        if candidate and candidate in providers:
            return candidate

    for candidate in ("gpt", "gemini", "dpsk", "qwen"):
        if candidate in providers:
            return candidate

    return available_provider_names[0]


def build_target_length_hint(original_length: int, prompt_record: Dict[str, Any]) -> str:
    if original_length <= 0:
        return "与原文接近"
    low = max(1, int(original_length * 0.8))
    high = max(low, int(original_length * 1.25))
    return f"约 {original_length} 个中文汉字，建议控制在 {low}-{high} 之间（按汉字数估计）"


def clean_rewrite_output(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("“") and cleaned.endswith("”") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned.strip()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_output_path(source_path: Path, rewrite_round: int) -> Path:
    suffix = source_path.suffix or ".json"
    stem = source_path.stem
    if stem.endswith("_review"):
        stem = stem[: -len("_review")]
    return source_path.with_name(f"{stem}_structured_r{rewrite_round}{suffix}")


if __name__ == "__main__":
    main()
