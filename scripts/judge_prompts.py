#!/usr/bin/env python3
"""Run GLM LLM-as-judge review on generated prompt JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.judge import (
    OpenAICompatibleJudgeProvider,
    build_token_budget_batches,
    judge_prompt_record,
    judge_prompt_batch,
    load_existing_results,
    load_prompt_file,
    stratified_sample_prompts,
    write_judge_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GLM judge over generated prompt files.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config YAML path")
    parser.add_argument("--input", action="append", help="Prompt JSON file to judge; can be repeated")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of prompts per input file; 0 means all")
    parser.add_argument("--all", action="store_true", help="Judge all prompts from each input file")
    parser.add_argument("--output-dir", default=None, help="Directory for judge reports")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing report results")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Optional hard cap on prompts per batch")
    parser.add_argument("--dry-run", action="store_true", help="Only show sampling plan and provider config")
    args = parser.parse_args()

    config = load_config(args.config)
    judge_cfg = config.get("judge", {})
    if not judge_cfg:
        raise ValueError("Missing judge config in config.yaml")

    input_specs = _resolve_input_specs(judge_cfg, args.input)
    sample_cfg = judge_cfg.get("sample", {})
    sample_size = 0 if args.all else args.sample_size
    if sample_size is None:
        sample_size = int(sample_cfg.get("size_per_file", 50))
    seed = int(sample_cfg.get("seed", 42))
    output_dir = _resolve_path(args.output_dir or judge_cfg.get("output_dir", "reports/judge"))
    resume = bool(judge_cfg.get("resume", True)) and not args.no_resume
    max_context_tokens = int(judge_cfg.get("max_context_tokens", 200000))
    target_fraction = float(judge_cfg.get("target_context_fraction", 0.4))
    target_input_tokens = int(judge_cfg.get("target_input_tokens", max_context_tokens * target_fraction))
    max_batch_size = args.max_batch_size
    if max_batch_size is None:
        max_batch_size = int(judge_cfg.get("max_prompts_per_batch", 0))

    provider = OpenAICompatibleJudgeProvider(judge_cfg)
    if args.dry_run:
        print("[Dry-run] GLM judge configuration")
        print(f"  provider: {judge_cfg.get('provider', 'glm')}")
        print(f"  model: {judge_cfg.get('model')}")
        print(f"  base_url: {judge_cfg.get('base_url')}")
        print(f"  api_key_env: {judge_cfg.get('api_key_env', 'GLM_JUDGE_API_KEY')}")
        print(f"  api_key_present: {bool(provider.api_key)}")
        print(f"  target_input_tokens: {target_input_tokens}")
        print(f"  max_prompts_per_batch: {max_batch_size or 'unlimited'}")
    elif not provider.is_available():
        raise ValueError(
            f"Judge provider unavailable. Please set {provider.api_key_env} "
            "and check judge.base_url/model."
        )

    for spec in input_specs:
        input_path = _resolve_path(spec["path"])
        model_label = spec.get("model_label") or input_path.stem
        prompts = load_prompt_file(input_path)
        selected = stratified_sample_prompts(prompts, sample_size, seed)
        output_path = output_dir / f"{model_label}_{judge_cfg.get('provider', 'glm')}_{judge_cfg.get('judge_version', 'v1')}.json"

        print(
            f"[Judge] {model_label}: total={len(prompts)}, selected={len(selected)}, "
            f"output={output_path}"
        )
        existing_results, judged_ids = load_existing_results(output_path) if resume else ([], set())
        results = list(existing_results)
        pending = [item for item in selected if item.get("prompt_id") not in judged_ids]
        batches = build_token_budget_batches(
            pending,
            difficulty_params=config.difficulty_params,
            target_input_tokens=target_input_tokens,
            max_prompts_per_batch=max_batch_size,
        )
        print(
            f"  resume={resume}, existing={len(existing_results)}, pending={len(pending)}, "
            f"batches={len(batches)}, target_input_tokens={target_input_tokens}"
        )

        if args.dry_run:
            _print_batch_plan(batches)
            continue

        processed = 0
        for batch_index, batch in enumerate(batches, start=1):
            batch_records = batch["records"]
            try:
                batch_results = judge_prompt_batch(
                    provider=provider,
                    prompt_records=batch_records,
                    difficulty_params=config.difficulty_params,
                    save_raw_response=bool(judge_cfg.get("save_raw_response", True)),
                )
            except Exception as exc:
                if len(batch_records) == 1:
                    batch_results = [_error_result(batch_records[0], exc)]
                else:
                    print(
                        f"  batch {batch_index}/{len(batches)} failed, "
                        f"falling back to single-item judge: {exc}"
                    )
                    batch_results = _judge_records_one_by_one(
                        provider,
                        batch_records,
                        config.difficulty_params,
                        bool(judge_cfg.get("save_raw_response", True)),
                    )
            results.extend(batch_results)
            processed += len(batch_records)

            if batch_index % 2 == 0 or batch_index == len(batches):
                _write_report(output_path, results, input_path, model_label, prompts, selected, judge_cfg)
                print(
                    f"  checkpoint: {len(results)} judged "
                    f"(processed {processed}/{len(pending)}, batch {batch_index}/{len(batches)})"
                )

        _write_report(output_path, results, input_path, model_label, prompts, selected, judge_cfg)


def _resolve_input_specs(judge_cfg: Dict[str, Any], cli_inputs: List[str] | None) -> List[Dict[str, str]]:
    if cli_inputs:
        return [{"path": item, "model_label": Path(item).stem.replace("_prompts", "")} for item in cli_inputs]
    specs = judge_cfg.get("input_files", [])
    if not specs:
        raise ValueError("No judge input files configured.")
    return specs


def _write_report(
    output_path: Path,
    results: List[Dict[str, Any]],
    input_path: Path,
    model_label: str,
    prompts: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    judge_cfg: Dict[str, Any],
) -> None:
    judge_info = {
        "provider": judge_cfg.get("provider", "glm"),
        "model": judge_cfg.get("model"),
        "base_url": judge_cfg.get("base_url"),
        "judge_version": judge_cfg.get("judge_version", "v1"),
        "temperature": judge_cfg.get("temperature", 0),
        "max_tokens": judge_cfg.get("max_tokens", 4096),
        "max_context_tokens": judge_cfg.get("max_context_tokens", 200000),
        "target_context_fraction": judge_cfg.get("target_context_fraction", 0.4),
        "target_input_tokens": judge_cfg.get("target_input_tokens"),
        "max_prompts_per_batch": judge_cfg.get("max_prompts_per_batch", 0),
    }
    source_info = {
        "path": str(input_path),
        "model_label": model_label,
        "total_prompts": len(prompts),
        "selected_prompts": len(selected),
    }
    write_judge_report(output_path, results, source_info, judge_info)


def _print_batch_plan(batches: List[Dict[str, Any]]) -> None:
    if not batches:
        print("  batch_plan: no pending prompts")
        return
    counts = [batch["prompt_count"] for batch in batches]
    tokens = [batch["estimated_input_tokens"] for batch in batches]
    print(
        "  batch_plan: "
        f"count={len(batches)}, prompts_per_batch=min{min(counts)} max{max(counts)} "
        f"avg{sum(counts) / len(counts):.1f}, "
        f"estimated_input_tokens=min{min(tokens)} max{max(tokens)} avg{sum(tokens) / len(tokens):.0f}"
    )
    for idx, batch in enumerate(batches[:10], start=1):
        ids = [item.get("prompt_id") for item in batch["records"][:3]]
        print(
            f"    batch {idx}: prompts={batch['prompt_count']}, "
            f"estimated_input_tokens={batch['estimated_input_tokens']}, preview={ids}"
        )
    if len(batches) > 10:
        print(f"    ... {len(batches) - 10} more batches")


def _judge_records_one_by_one(
    provider: OpenAICompatibleJudgeProvider,
    records: List[Dict[str, Any]],
    difficulty_params: Dict[str, Any],
    save_raw_response: bool,
) -> List[Dict[str, Any]]:
    results = []
    for record in records:
        try:
            results.append(
                judge_prompt_record(
                    provider=provider,
                    prompt_record=record,
                    difficulty_params=difficulty_params,
                    save_raw_response=save_raw_response,
                )
            )
        except Exception as exc:
            results.append(_error_result(record, exc))
    return results


def _error_result(prompt_record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    return {
        "prompt_id": prompt_record.get("prompt_id"),
        "combination_id": prompt_record.get("combination_id"),
        "source_llm": prompt_record.get("llm", {}),
        "difficulty": prompt_record.get("difficulty", {}),
        "text": prompt_record.get("text", ""),
        "sampling": prompt_record.get("sampling", {}),
        "rule_precheck": {},
        "llm_judge": {
            "overall_decision": "FAIL",
            "overall_score": 1,
            "dimension_scores": {
                "concept_fidelity": 1,
                "clarity": 1,
                "internal_consistency": 1,
                "language_quality": 1,
                "video_prompt_usability": 1,
                "difficulty_alignment": 1,
            },
            "concept_checks": [],
            "issues": [{
                "severity": "fatal",
                "type": "judge_error",
                "location": "judge",
                "description": str(exc),
                "suggested_fix": "Inspect raw provider error and retry.",
            }],
            "short_summary": "Judge call failed.",
        },
        "judge_error": str(exc),
    }


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
