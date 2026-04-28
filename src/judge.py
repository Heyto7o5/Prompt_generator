"""
LLM-as-judge utilities for generated video prompts.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ISSUE_TYPES = [
    "missing_mandatory_concept",
    "weak_or_implicit_concept",
    "wrong_concept_substitution",
    "ambiguous_subject",
    "ambiguous_action",
    "ambiguous_scene",
    "ambiguous_audio",
    "internal_contradiction",
    "physical_or_temporal_conflict",
    "grammar_or_typo",
    "non_english_or_mixed_language",
    "format_violation",
    "too_vague_for_video_generation",
    "challenge_not_reflected",
    "difficulty_mismatch",
    "overcomplicated_low_difficulty",
    "judge_error",
]


JUDGE_DIMENSION_KEYS = [
    "concept_fidelity",
    "clarity",
    "internal_consistency",
    "language_quality",
    "video_prompt_usability",
    "difficulty_alignment",
]


class OpenAICompatibleJudgeProvider:
    """OpenAI-compatible chat-completions provider for judge calls."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get("provider", "glm")
        self.model = config.get("model", "glm-5.1")
        self.base_url = config.get("base_url", "")
        self.api_key_env = config.get("api_key_env", "GLM_JUDGE_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "")
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 3)
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("请安装 openai: pip install openai") from exc

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"Judge LLM call failed after {self.max_retries} attempts: {last_error}")


def build_judge_messages(prompt_record: Dict[str, Any], rule_precheck: Dict[str, Any]) -> Tuple[str, str]:
    """Build strict judge instructions and one prompt-review input."""
    compact_record = _compact_prompt_record(prompt_record)
    system_prompt = """You are a strict QA reviewer for English video-generation prompts. You are not the prompt writer. Your job is to judge whether a generated prompt faithfully and clearly expresses all mandatory selected concepts, is internally consistent, has no meaningful ambiguity, has no grammar/spelling/format problems, and is usable by a video generation model. Be conservative: a prompt that sounds fluent but omits a mandatory selected concept must fail."""

    user_prompt = f"""Review the generated video prompt below.

Core principle:
- The selected concepts are mandatory requirements.
- Every selected concept must be explicitly expressed in the final prompt text.
- Explicit means a reader can locate clear textual evidence for the concept or an obvious English equivalent.
- Implied, hidden, overly generic, or replaced concepts do not count as fully present.
- Audio concepts are not special exceptions: if an audio category is selected, the audio concept must be clearly audible in the prompt; if audio is not selected, do not require audio.
- Future or unknown categories should be judged according to their semantic role.
- Natural incidental details are allowed when they do not conflict with or obscure mandatory concepts.

Hard-fail rules:
- FAIL if any mandatory selected concept is missing or incorrectly substituted.
- FAIL if the prompt contains an internal contradiction that makes the described video incoherent.
- FAIL if format/language problems make the prompt unsuitable for direct video generation.
- If a challenge element intentionally asks for anti-commonsense or unusual physics, do not fail it merely for being unrealistic; only fail actual self-contradictions or unclear instructions.

Dimension scoring rubric, use integers 1-5:
- 5: Excellent, no meaningful issue.
- 4: Good, only minor imperfections.
- 3: Acceptable but has noticeable weakness.
- 2: Poor, major issue affects usefulness.
- 1: Invalid or nearly unusable.

Required JSON output schema:
{{
  "overall_decision": "PASS | PASS_WITH_MINOR_ISSUES | FAIL",
  "overall_score": 1,
  "dimension_scores": {{
    "concept_fidelity": 1,
    "clarity": 1,
    "internal_consistency": 1,
    "language_quality": 1,
    "video_prompt_usability": 1,
    "difficulty_alignment": 1
  }},
  "concept_checks": [
    {{
      "category": "subject|motion|scene|audio|other",
      "required_concept": "level3 or leaf concept",
      "status": "present | weak | missing | wrong",
      "evidence": "short evidence phrase from the prompt, or empty string if absent",
      "reason": "brief reason"
    }}
  ],
  "issues": [
    {{
      "severity": "fatal | major | minor",
      "type": "one of: {", ".join(ISSUE_TYPES)}",
      "location": "concept category, text span, or general",
      "description": "brief concrete issue",
      "suggested_fix": "brief actionable fix"
    }}
  ],
  "short_summary": "one-sentence summary"
}}

Rules for your answer:
- Output valid JSON only.
- Do not use Markdown, code fences, comments, or extra explanation.
- Do not invent issues. If there is no issue, use an empty issues list.
- For each selected concept, include exactly one concept_checks item.
- For missing concepts, evidence must be an empty string.
- Use English for all reasons and descriptions.

Rule-based precheck findings:
{json.dumps(rule_precheck, ensure_ascii=False, indent=2)}

Prompt record to review:
{json.dumps(compact_record, ensure_ascii=False, indent=2)}
"""
    return system_prompt, user_prompt


def build_batch_judge_messages(batch_items: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Build judge messages for a batch of prompt records."""
    records = [
        {
            "prompt_id": item["prompt_record"].get("prompt_id"),
            "rule_precheck": item["rule_precheck"],
            "prompt_record": _compact_prompt_record(item["prompt_record"]),
        }
        for item in batch_items
    ]

    system_prompt = """You are a strict QA reviewer for English video-generation prompts. You are not the prompt writer. Judge every prompt independently. Be conservative: a fluent prompt that omits a mandatory selected concept must fail."""

    user_prompt = f"""Review each generated video prompt in the batch.

Core principle:
- The selected concepts are mandatory requirements.
- Every selected concept must be explicitly expressed in the final prompt text.
- Explicit means a reader can locate clear textual evidence for the concept or an obvious English equivalent.
- Implied, hidden, overly generic, or replaced concepts do not count as fully present.
- If an audio category is selected, the audio concept must be clearly audible in the prompt; if audio is not selected, do not require audio.
- Future or unknown categories should be judged according to their semantic role.
- Natural incidental details are allowed when they do not conflict with or obscure mandatory concepts.

Hard-fail rules:
- FAIL if any mandatory selected concept is missing or incorrectly substituted.
- FAIL if the prompt contains an internal contradiction that makes the described video incoherent.
- FAIL if format/language problems make the prompt unsuitable for direct video generation.
- If a challenge element intentionally asks for anti-commonsense or unusual physics, do not fail it merely for being unrealistic; only fail actual self-contradictions or unclear instructions.

Scoring rubric, use integers 1-5:
- 5: Excellent, no meaningful issue.
- 4: Good, only minor imperfections.
- 3: Acceptable but has noticeable weakness.
- 2: Poor, major issue affects usefulness.
- 1: Invalid or nearly unusable.

For each prompt, produce one result object with this schema:
{{
  "prompt_id": "same prompt_id from input",
  "overall_decision": "PASS | PASS_WITH_MINOR_ISSUES | FAIL",
  "overall_score": 1,
  "dimension_scores": {{
    "concept_fidelity": 1,
    "clarity": 1,
    "internal_consistency": 1,
    "language_quality": 1,
    "video_prompt_usability": 1,
    "difficulty_alignment": 1
  }},
  "concept_checks": [
    {{
      "category": "subject|motion|scene|audio|other",
      "required_concept": "level3 or leaf concept",
      "status": "present | weak | missing | wrong",
      "evidence": "short evidence phrase, or empty string if absent",
      "reason": "brief reason"
    }}
  ],
  "issues": [
    {{
      "severity": "fatal | major | minor",
      "type": "one of: {", ".join(ISSUE_TYPES)}",
      "location": "concept category, text span, or general",
      "description": "brief concrete issue",
      "suggested_fix": "brief actionable fix"
    }}
  ],
  "short_summary": "one-sentence summary"
}}

Output rules:
- Output valid JSON only.
- The top-level JSON must be: {{"results": [ ... ]}}
- Return exactly one result object per input prompt_id.
- Preserve every prompt_id exactly.
- Do not use Markdown, code fences, comments, or extra explanation.
- Do not invent issues. If there is no issue, use an empty issues list.
- For each selected concept, include exactly one concept_checks item.
- For missing concepts, evidence must be an empty string.
- Keep each reason, description, suggested_fix, and summary concise.
- Use English for all reasons and descriptions.

Batch records:
{json.dumps(records, ensure_ascii=False, indent=2)}
"""
    return system_prompt, user_prompt


def rule_precheck(prompt_record: Dict[str, Any], difficulty_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic checks for mechanical format and length issues."""
    text = prompt_record.get("text") or ""
    issues = []

    def add_issue(issue_type: str, description: str, severity: str = "minor", location: str = "text"):
        issues.append({
            "severity": severity,
            "type": issue_type,
            "location": location,
            "description": description,
        })

    if not text.strip():
        add_issue("format_violation", "Prompt text is empty.", "fatal")
    if "\n" in text or "\r" in text:
        add_issue("format_violation", "Prompt contains line breaks.", "major")
    if re.search(r"(^|\s)(Prompt|Subject|Motion|Scene|Audio|Concepts|Description)\s*:", text):
        add_issue("format_violation", "Prompt contains field labels.", "major")
    if re.search(r"```|\*\*|^\s*[-*#]|\n\s*[-*#]", text):
        add_issue("format_violation", "Prompt contains Markdown-like formatting.", "major")
    if re.match(r"\s*[\[{]", text):
        add_issue("format_violation", "Prompt appears to be JSON or a structured object.", "major")
    if re.match(r"\s*\d+[\.)]\s+", text):
        add_issue("format_violation", "Prompt appears to start with numbered-list formatting.", "major")
    if re.search(r"[\u4e00-\u9fff]", text):
        add_issue("non_english_or_mixed_language", "Prompt contains Chinese characters.", "major")

    word_count = len(text.split())
    difficulty = str((prompt_record.get("difficulty") or {}).get("level", "")).lower()
    params = (difficulty_params or {}).get(difficulty, {})
    max_len = params.get("text_length_max")
    min_len = params.get("text_length_min")
    if max_len is not None and word_count > max_len:
        add_issue(
            "difficulty_mismatch",
            f"Prompt has {word_count} words, exceeding max length {max_len} for {difficulty.upper()}.",
            "minor",
            "difficulty",
        )
    if min_len is not None and word_count < min_len:
        add_issue(
            "difficulty_mismatch",
            f"Prompt has {word_count} words, below min length {min_len} for {difficulty.upper()}.",
            "minor",
            "difficulty",
        )

    return {
        "passed": not any(issue["severity"] in {"fatal", "major"} for issue in issues),
        "word_count": word_count,
        "issues": issues,
    }


def judge_prompt_record(
    provider: OpenAICompatibleJudgeProvider,
    prompt_record: Dict[str, Any],
    difficulty_params: Optional[Dict[str, Any]] = None,
    save_raw_response: bool = True,
) -> Dict[str, Any]:
    precheck = rule_precheck(prompt_record, difficulty_params)
    system_prompt, user_prompt = build_judge_messages(prompt_record, precheck)
    raw_response = provider.generate(system_prompt, user_prompt)
    parsed = parse_json_object(raw_response)
    normalized = normalize_judge_output(parsed)

    result = {
        "prompt_id": prompt_record.get("prompt_id"),
        "combination_id": prompt_record.get("combination_id"),
        "source_llm": prompt_record.get("llm", {}),
        "difficulty": prompt_record.get("difficulty", {}),
        "text": prompt_record.get("text", ""),
        "sampling": prompt_record.get("sampling", {}),
        "rule_precheck": precheck,
        "llm_judge": normalized,
        "judged_at": datetime.now().isoformat(),
    }
    if save_raw_response:
        result["raw_response"] = raw_response
    return result


def judge_prompt_batch(
    provider: OpenAICompatibleJudgeProvider,
    prompt_records: List[Dict[str, Any]],
    difficulty_params: Optional[Dict[str, Any]] = None,
    save_raw_response: bool = True,
) -> List[Dict[str, Any]]:
    """Judge a batch of prompt records in one LLM call."""
    batch_items = [
        {
            "prompt_record": prompt_record,
            "rule_precheck": rule_precheck(prompt_record, difficulty_params),
        }
        for prompt_record in prompt_records
    ]
    system_prompt, user_prompt = build_batch_judge_messages(batch_items)
    raw_response = provider.generate(system_prompt, user_prompt)
    parsed = parse_json_payload(raw_response)

    raw_results = parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(raw_results, list):
        raise ValueError("Batch judge response must contain a results list.")

    by_prompt_id = {
        str(item.get("prompt_id")): item
        for item in raw_results
        if isinstance(item, dict) and item.get("prompt_id") is not None
    }

    results = []
    for item in batch_items:
        prompt_record = item["prompt_record"]
        prompt_id = str(prompt_record.get("prompt_id"))
        judge_item = by_prompt_id.get(prompt_id)
        if judge_item is None:
            result = _judge_error_result(
                prompt_record,
                ValueError(f"Missing judge result for prompt_id={prompt_id}"),
            )
        else:
            normalized = normalize_judge_output(judge_item)
            result = _build_result_record(prompt_record, item["rule_precheck"], normalized)
            if save_raw_response:
                result["raw_response"] = json.dumps(judge_item, ensure_ascii=False)
        results.append(result)

    return results


def parse_json_object(text: str) -> Dict[str, Any]:
    parsed = parse_json_payload(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object.")
    return parsed


def parse_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def normalize_judge_output(data: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(data.get("overall_decision", "FAIL")).upper()
    if decision not in {"PASS", "PASS_WITH_MINOR_ISSUES", "FAIL"}:
        decision = "FAIL"

    score = _clamp_int(data.get("overall_score"), 1, 5, default=1)
    dimension_scores = data.get("dimension_scores") or {}
    normalized_dimensions = {
        key: _clamp_int(dimension_scores.get(key), 1, 5, default=1)
        for key in JUDGE_DIMENSION_KEYS
    }

    return {
        "overall_decision": decision,
        "overall_score": score,
        "dimension_scores": normalized_dimensions,
        "concept_checks": data.get("concept_checks") or [],
        "issues": data.get("issues") or [],
        "short_summary": data.get("short_summary", ""),
    }


def stratified_sample_prompts(prompts: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(prompts):
        return list(prompts)

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        difficulty = str((prompt.get("difficulty") or {}).get("level", "UNKNOWN")).upper()
        groups[difficulty].append(prompt)

    for items in groups.values():
        rng.shuffle(items)

    selected = []
    ordered_keys = ["LOW", "MEDIUM", "HIGH"] + sorted(
        key for key in groups if key not in {"LOW", "MEDIUM", "HIGH"}
    )
    active_keys = [key for key in ordered_keys if groups.get(key)]
    if not active_keys:
        return []

    per_group = sample_size // len(active_keys)
    remainder = sample_size % len(active_keys)
    for idx, key in enumerate(active_keys):
        take = per_group + (1 if idx < remainder else 0)
        selected.extend(groups[key][:take])

    if len(selected) < sample_size:
        selected_ids = {id(item) for item in selected}
        leftovers = [item for item in prompts if id(item) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:sample_size - len(selected)])

    rng.shuffle(selected)
    return selected[:sample_size]


def build_token_budget_batches(
    prompt_records: List[Dict[str, Any]],
    difficulty_params: Optional[Dict[str, Any]] = None,
    target_input_tokens: int = 80000,
    max_prompts_per_batch: int = 0,
) -> List[Dict[str, Any]]:
    """Pack prompts into batches by estimated input token budget."""
    batches = []
    current_records: List[Dict[str, Any]] = []
    current_items: List[Dict[str, Any]] = []

    for prompt_record in prompt_records:
        item = {
            "prompt_record": prompt_record,
            "rule_precheck": rule_precheck(prompt_record, difficulty_params),
        }
        tentative_items = current_items + [item]
        estimated_tokens = estimate_batch_input_tokens(tentative_items)
        would_exceed_count = max_prompts_per_batch > 0 and len(tentative_items) > max_prompts_per_batch
        would_exceed_tokens = current_items and estimated_tokens > target_input_tokens

        if would_exceed_count or would_exceed_tokens:
            batches.append(_make_batch_plan(current_records, current_items))
            current_records = [prompt_record]
            current_items = [item]
            continue

        current_records.append(prompt_record)
        current_items = tentative_items

    if current_items:
        batches.append(_make_batch_plan(current_records, current_items))

    return batches


def estimate_batch_input_tokens(batch_items: List[Dict[str, Any]]) -> int:
    system_prompt, user_prompt = build_batch_judge_messages(batch_items)
    return estimate_tokens(system_prompt) + estimate_tokens(user_prompt)


def estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed English/Chinese prompt payloads."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = max(0, len(text) - chinese_chars)
    return int(chinese_chars + other_chars / 4) + 1


def load_prompt_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])


def load_existing_results(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    judged_ids = {
        item.get("prompt_id")
        for item in results
        if item.get("prompt_id")
    }
    return results, judged_ids


def write_judge_report(
    output_path: Path,
    results: List[Dict[str, Any]],
    source_info: Dict[str, Any],
    judge_info: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "judge": judge_info,
        "source": source_info,
        "summary": summarize_results(results),
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    decision_counts = Counter()
    issue_counts = Counter()
    concept_status_counts = Counter()
    score_sums = Counter()
    score_counts = Counter()

    for result in results:
        judge = result.get("llm_judge", {})
        decision_counts[judge.get("overall_decision", "UNKNOWN")] += 1
        for issue in judge.get("issues", []):
            issue_counts[issue.get("type", "unknown")] += 1
        for check in judge.get("concept_checks", []):
            concept_status_counts[check.get("status", "unknown")] += 1
        for key, value in (judge.get("dimension_scores") or {}).items():
            score_sums[key] += value
            score_counts[key] += 1

    avg_scores = {
        key: round(score_sums[key] / score_counts[key], 3)
        for key in score_sums
        if score_counts[key]
    }
    return {
        "total_judged": len(results),
        "decision_counts": dict(decision_counts),
        "issue_counts": dict(issue_counts),
        "concept_status_counts": dict(concept_status_counts),
        "average_dimension_scores": avg_scores,
    }


def _build_result_record(
    prompt_record: Dict[str, Any],
    precheck: Dict[str, Any],
    normalized_judge: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "prompt_id": prompt_record.get("prompt_id"),
        "combination_id": prompt_record.get("combination_id"),
        "source_llm": prompt_record.get("llm", {}),
        "difficulty": prompt_record.get("difficulty", {}),
        "text": prompt_record.get("text", ""),
        "sampling": prompt_record.get("sampling", {}),
        "rule_precheck": precheck,
        "llm_judge": normalized_judge,
        "judged_at": datetime.now().isoformat(),
    }


def _judge_error_result(prompt_record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
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
            "dimension_scores": {key: 1 for key in JUDGE_DIMENSION_KEYS},
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


def _make_batch_plan(records: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "records": list(records),
        "estimated_input_tokens": estimate_batch_input_tokens(items),
        "prompt_count": len(records),
    }


def _compact_prompt_record(prompt_record: Dict[str, Any]) -> Dict[str, Any]:
    sampling = prompt_record.get("sampling") or {}
    return {
        "prompt_id": prompt_record.get("prompt_id"),
        "combination_id": prompt_record.get("combination_id"),
        "llm": prompt_record.get("llm", {}),
        "difficulty": prompt_record.get("difficulty", {}),
        "categories_selected": sampling.get("categories_selected", []),
        "concepts": sampling.get("concepts", {}),
        "challenge_elements": sampling.get("challenge_elements", []),
        "text": prompt_record.get("text", ""),
        "text_length": prompt_record.get("text_length"),
    }


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
