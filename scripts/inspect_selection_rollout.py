#!/usr/bin/env python3
"""Roll out raw LLM Level-2 companion selections for inspection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.concept_loader import load_concepts
from src.generator import DpskProvider, GeminiProvider, GPTProvider, QwenProvider
from src.selection_prompt import ConceptSelectionPromptBuilder


PROVIDER_CLASSES = {
    "dpsk": DpskProvider,
    "gemini": GeminiProvider,
    "gpt": GPTProvider,
    "qwen": QwenProvider,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the LLM output for Level-2 companion category selection."
    )
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--provider", default=None, help="Selection provider; default from config")
    parser.add_argument("--core-dim", default="subject", help="Core dimension key")
    parser.add_argument("--count", type=int, default=5, help="Number of core Level-3 concepts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random", action="store_true", help="Randomly sample core concepts")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path; default output/selection_rollouts/selection_rollout_<timestamp>.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    loader = load_concepts(config.concept_tree_path)
    prompt_builder = ConceptSelectionPromptBuilder(loader, config.dimensions)

    provider_name = args.provider or config.selection_llm_provider
    provider = build_provider(provider_name, config.llm_providers)

    key_to_sheet = {d["key"]: d["sheet"] for d in config.dimensions}
    if args.core_dim not in key_to_sheet:
        raise ValueError(f"Unknown core dimension: {args.core_dim}")

    core_sheet = key_to_sheet[args.core_dim]
    core_concepts = select_core_concepts(
        loader.get_level3_categories(core_sheet),
        count=args.count,
        seed=args.seed,
        random_sample=args.random,
    )
    target_dimensions = [
        d["key"]
        for d in config.dimensions
        if d["key"] != args.core_dim and d.get("companion", True)
    ]
    available_level2 = {
        dim_key: list(prompt_builder.level2_structure.get(dim_key, {}).keys())
        for dim_key in target_dimensions
    }

    prompt = prompt_builder.build_batch_prompt(
        core_dim_key=args.core_dim,
        core_concepts=core_concepts,
        available_level2=available_level2,
        target_dimensions=target_dimensions,
    )
    raw_response = provider.generate(prompt)
    parsed = prompt_builder.parse_batch_response(
        raw_response,
        core_concepts,
        target_dimensions,
    )

    output = {
        "generated_at": datetime.now().isoformat(),
        "provider": provider_name,
        "model": getattr(provider, "model", ""),
        "core_dimension": args.core_dim,
        "target_dimensions": target_dimensions,
        "core_concepts": [
            {
                "name": concept.name,
                "path": list(concept.path),
            }
            for concept in core_concepts
        ],
        "available_level2_counts": {
            dim_key: len(names)
            for dim_key, names in available_level2.items()
        },
        "raw_response": raw_response,
        "parsed_selections": parsed,
        "selection_prompt": prompt,
    }

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Selection rollout written to: {output_path}")
    print(f"Provider: {provider_name} | Model: {getattr(provider, 'model', '')}")
    print(f"Core dimension: {args.core_dim} | Count: {len(core_concepts)}")
    print("\nCore concepts:")
    for concept in core_concepts:
        print(f"- {concept.name} ({' > '.join(concept.path)})")
    print("\nRaw LLM response:")
    print(raw_response)
    print("\nParsed selections:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


def build_provider(provider_name: str, llm_providers: Dict[str, Any]):
    provider_class = PROVIDER_CLASSES.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider_name}")
    provider_config = llm_providers.get(provider_name, {})
    if not provider_config:
        raise ValueError(f"Missing llm.providers.{provider_name} config")
    provider = provider_class(provider_config)
    if not provider.is_available():
        api_key_env = provider_config.get("api_key_env", "")
        raise ValueError(f"Provider {provider_name} unavailable. Check {api_key_env}.")
    return provider


def select_core_concepts(nodes: List[Any], count: int, seed: int, random_sample: bool) -> List[Any]:
    candidates = [
        node
        for node in nodes
        if not any(keyword in node.name for keyword in ["其他", "其它", "Other", "other"])
    ]
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(candidates)
    return candidates[:count]


def resolve_output_path(path: str | None) -> Path:
    if path:
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        return output_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "output" / "selection_rollouts" / f"selection_rollout_{timestamp}.json"


if __name__ == "__main__":
    main()
