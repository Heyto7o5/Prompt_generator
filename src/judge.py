"""
LLM-as-judge utilities for generated video prompts.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .output import derive_review_output_path, derive_selection_trace_path
from .text_metrics import count_chinese_chars
from .structured_templates import STRUCTURED_TEMPLATE_IDS, validate_structured_output


ISSUE_TYPES = [
    "missing_mandatory_concept",
    "weak_or_implicit_concept",
    "wrong_concept_substitution",
    "parent_path_violation",
    "core_focus_lost",
    "ambiguous_subject",
    "ambiguous_action",
    "ambiguous_scene",
    "ambiguous_audio",
    "internal_contradiction",
    "physical_or_temporal_conflict",
    "grammar_or_typo",
    "mixed_language",
    "format_violation",
    "template_format_mismatch",
    "abstract_or_vague",
    "keyword_salad",
    "challenge_not_reflected",
    "challenge_incompatible",
    "difficulty_mismatch",
    "overcomplicated_low_difficulty",
    "too_static",
    "information_overload",
    "concept_combination_conflict",
    "judge_error",
]


JUDGE_DIMENSION_KEYS = [
    "concept_fidelity_focus",
    "internal_consistency",
    "clarity_concreteness",
    "language_quality",
    "video_prompt_usability",
]

DEFAULT_TARGET_INPUT_TOKENS = 10000


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
    precheck_summary = _rule_precheck_summary(rule_precheck)
    system_prompt = """你是严格的视频生成提示词质量审核员，不是提示词作者。你的任务是审核中文视频生成提示词是否准确表达所有强制概念、是否聚焦核心概念、是否内部一致、是否清晰具象、语言是否规范、是否适合直接用于视频生成。请保守判断：如果强制概念缺失、替换、父级路径语义被破坏，必须判为失败。若待审记录包含 template_family，则把它视为目标结构模板；结构化换行、镜头号、字段名、时间块、分屏标签都属于允许的表达方式，不要把这些合理结构本身误判为格式违规，但若它与标记模板明显不一致，请指出。"""

    user_prompt = f"""请审核下面的视频生成提示词。

核心原则：
- 被选中的 concepts 都是强制概念，必须在最终 prompt 中明确、准确表达。
- 不要扩写、删除、弱化、替换为其他概念；必须保持中文语义。
- concept fidelity 包含完整路径语义：不能只命中叶子词而偏离 level1/level2/level3 父级语义。
- 如果 concept 里存在 leaf，leaf 可能是一组由顿号、逗号、斜杠等分隔的具体候选项；这种 leaf 候选集合不要求全部出现在最终 prompt 中，只要自然、准确地体现其中一个或少数具体项，并且不偏离 level1/level2/level3 父级语义即可。
- 核心概念应在视频时间线中保持合理焦点，不能被极端边缘化。
- 所有大类一视同仁：被选中的大类按自身语义表达，未选中的大类不强制要求。
- 少量连贯性细节可以接受，但不能冲突、遮蔽或替代强制概念。
- challenge_elements 是已选择的挑战性要素，不是可选参考。必须逐条检查它是否与 concepts 兼容、是否在最终 prompt 中得到可观察表达、reason 是否引入未选概念。
- 如果 challenge 本身要求反常识或特殊物理，不要仅因“不现实”扣分；只惩罚自相矛盾、不可理解、与概念不兼容、完全没有体现，或 reason 绑定了未选概念。
- 若待审记录包含 template_family，则把该字段视为目标结构模板。结构化换行、字段标签、镜头号、时间块、分屏标签等都属于允许结构；只有当结构明显与 template_family 不一致或混乱时，才在 issues 中使用 template_format_mismatch。

五个评分维度，使用 1-5 整数：
- concept_fidelity_focus：概念保真与聚焦。检查概念是否准确完整、父级路径是否保真、核心概念是否保持焦点；challenge reason 引入未选概念也归入这一维度。
- internal_consistency：内部一致性。检查是否有同一时空下互斥设定、物理/时间/空间/镜头冲突。
- clarity_concreteness：清晰度与具象化。检查是否能形成明确视听画面，避免抽象堆词和关键词沙拉。
- language_quality：语言质量与基础规范。检查中文是否流畅、无严重错别字、无无序多语种混杂。
- video_prompt_usability：视频生成可用性。检查是否适合直接喂给视频模型，避免纯静态图片提示词或信息超载。

硬性失败规则：
- 任一强制概念缺失、错误替换、父级路径语义被破坏，应 FAIL。
- 核心概念被极端边缘化，应 FAIL 或 PASS_WITH_MINOR_ISSUES，取决于严重程度。
- 存在导致视频不可理解的内部矛盾，应 FAIL。
- 大篇幅抽象描述、关键词沙拉、严重语言错误导致不可用，应 FAIL。

必须输出以下 JSON schema：
{{
  "overall_decision": "PASS | PASS_WITH_MINOR_ISSUES | FAIL",
  "overall_score": 1,
  "dimension_scores": {{
    "concept_fidelity_focus": 1,
    "internal_consistency": 1,
    "clarity_concreteness": 1,
    "language_quality": 1,
    "video_prompt_usability": 1
  }},
  "concept_checks": [
    {{
      "category": "subject|motion|scene|audio|other",
      "required_concept": "level3 concept, or one/some concrete item(s) from leaf candidate set",
      "required_path": ["完整路径"],
      "status": "present | weak | missing | wrong",
      "evidence": "提示词中的短证据；缺失则为空字符串",
      "reason": "简短原因"
    }}
  ],
  "challenge_checks": [
    {{
      "challenge_id": "挑战性要素id",
      "challenge_name": "挑战性要素名称",
      "status": "reflected | weak | missing | incompatible | concept_fidelity_issue",
      "evidence": "最终 prompt 中体现该挑战的短证据；缺失则为空字符串",
      "reason": "简短说明，若选择理由引入未选概念也必须说明"
    }}
  ],
  "issues": [
    {{
      "severity": "fatal | major | minor",
      "type": "one of: {", ".join(ISSUE_TYPES)}",
      "location": "concept category, text span, or general",
      "description": "具体问题",
      "suggested_fix": "可执行修复建议"
    }}
  ],
  "combination_issue": {{
    "is_concept_combination_problem": false,
    "should_add_to_contradiction_pool": false,
    "reason": "如果问题来自概念组合本身而非生成文本，请说明"
  }},
  "short_summary": "一句话总结"
}}

回答规则：
- 只输出合法 JSON。
- 不要使用 Markdown、代码块、注释或额外解释。
- 不要编造问题；没有问题时 issues 为空数组。
- 每个强制概念都必须有且只有一个 concept_checks item。
- 每个 challenge_elements item 都必须有且只有一个 challenge_checks item；没有 challenge_elements 时 challenge_checks 为空数组。
- 缺失概念的 evidence 必须为空字符串。
- 对包含多个具体项的 leaf 候选集合，不要因为没有覆盖全部候选项而判缺失；只有完全没有体现三级概念或没有体现任何合适叶子候选时，才判 missing/weak。
- 若挑战性元素在最终 prompt 中完全没有体现，在 issues 中使用 challenge_not_reflected；若挑战性元素与概念组合语义冲突，使用 challenge_incompatible；若 challenge reason 引入未被 concepts 选中的概念，不新增独立问题类型，应作为 concept_fidelity_focus 扣分，并在 issues 中使用 wrong_concept_substitution。
- 机械规则问题由系统确定性检查负责，例如长度、普通段落格式、结构化模板格式。你不要在 issues 中复述这些机械检查结果；只判断概念语义、内部一致性、清晰具象、语言质量和视频可用性。

系统机械检查摘要（仅用于提醒，不要复述到 issues；系统会在后处理自动合并）：
{json.dumps(precheck_summary, ensure_ascii=False, indent=2)}

待审核记录：
{json.dumps(compact_record, ensure_ascii=False, indent=2)}
"""
    return system_prompt, user_prompt


def build_batch_judge_messages(batch_items: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Build judge messages for a batch of prompt records."""
    records = [
        {
            "prompt_id": item["prompt_record"].get("prompt_id"),
            "rule_precheck_summary": _rule_precheck_summary(item["rule_precheck"]),
            "prompt_record": _compact_prompt_record(item["prompt_record"]),
        }
        for item in batch_items
    ]

    system_prompt = """你是严格的视频生成提示词质量审核员，不是提示词作者。请独立审核每条中文视频生成提示词。若强制概念缺失、错误替换、父级路径语义被破坏，必须判为失败。若待审记录包含 template_family，则把它视为目标结构模板；结构化换行、字段标签、镜头号、时间块、分屏标签都属于允许结构，不要把这些合理结构本身误判为格式违规。"""

    user_prompt = f"""请批量审核下面的视频生成提示词。

核心原则：
- 被选中的 concepts 都是强制概念，必须在最终 prompt 中明确、准确表达。
- 不要扩写、删除、弱化、替换为其他概念；必须保持中文语义。
- concept fidelity 包含完整路径语义：不能只命中叶子词而偏离 level1/level2/level3 父级语义。
- 如果 concept 里存在 leaf，leaf 可能是一组由顿号、逗号、斜杠等分隔的具体候选项；这种 leaf 候选集合不要求全部出现在最终 prompt 中，只要自然、准确地体现其中一个或少数具体项，并且不偏离 level1/level2/level3 父级语义即可。
- 核心概念应在视频时间线中保持合理焦点，不能被极端边缘化。
- 所有大类一视同仁：被选中的大类按自身语义表达，未选中的大类不强制要求。
- 少量连贯性细节可以接受，但不能冲突、遮蔽或替代强制概念。
- challenge_elements 是已选择的挑战性要素，不是可选参考。必须逐条检查它是否与 concepts 兼容、是否在最终 prompt 中得到可观察表达、reason 是否引入未选概念。
- 如果 challenge 本身要求反常识或特殊物理，不要仅因“不现实”扣分；只惩罚自相矛盾、不可理解、与概念不兼容、完全没有体现，或 reason 绑定了未选概念。
- 若待审记录包含 template_family，则把该字段视为目标结构模板。结构化换行、字段标签、镜头号、时间块、分屏标签等都属于允许结构；只有当结构明显与 template_family 不一致或混乱时，才在 issues 中使用 template_format_mismatch。

五个评分维度，使用 1-5 整数：
- concept_fidelity_focus：概念保真与聚焦。检查概念是否准确完整、父级路径是否保真、核心概念是否保持焦点；challenge reason 引入未选概念也归入这一维度。
- internal_consistency：内部一致性。检查是否有同一时空下互斥设定、物理/时间/空间/镜头冲突。
- clarity_concreteness：清晰度与具象化。检查是否能形成明确视听画面，避免抽象堆词和关键词沙拉。
- language_quality：语言质量与基础规范。检查中文是否流畅、无严重错别字、无无序多语种混杂。
- video_prompt_usability：视频生成可用性。检查是否适合直接喂给视频模型，避免纯静态图片提示词或信息超载。

硬性失败规则：
- 任一强制概念缺失、错误替换、父级路径语义被破坏，应 FAIL。
- 核心概念被极端边缘化，应 FAIL 或 PASS_WITH_MINOR_ISSUES，取决于严重程度。
- 存在导致视频不可理解的内部矛盾，应 FAIL。
- 大篇幅抽象描述、关键词沙拉、严重语言错误导致不可用，应 FAIL。

每条 prompt 输出一个 result object，schema 如下：
{{
  "prompt_id": "same prompt_id from input",
  "overall_decision": "PASS | PASS_WITH_MINOR_ISSUES | FAIL",
  "overall_score": 1,
  "dimension_scores": {{
    "concept_fidelity_focus": 1,
    "internal_consistency": 1,
    "clarity_concreteness": 1,
    "language_quality": 1,
    "video_prompt_usability": 1
  }},
  "concept_checks": [
    {{
      "category": "subject|motion|scene|audio|other",
      "required_concept": "level3 concept, or one/some concrete item(s) from leaf candidate set",
      "required_path": ["完整路径"],
      "status": "present | weak | missing | wrong",
      "evidence": "提示词中的短证据；缺失则为空字符串",
      "reason": "简短原因"
    }}
  ],
  "challenge_checks": [
    {{
      "challenge_id": "挑战性要素id",
      "challenge_name": "挑战性要素名称",
      "status": "reflected | weak | missing | incompatible | concept_fidelity_issue",
      "evidence": "最终 prompt 中体现该挑战的短证据；缺失则为空字符串",
      "reason": "简短说明，若选择理由引入未选概念也必须说明"
    }}
  ],
  "issues": [
    {{
      "severity": "fatal | major | minor",
      "type": "one of: {", ".join(ISSUE_TYPES)}",
      "location": "concept category, text span, or general",
      "description": "具体问题",
      "suggested_fix": "可执行修复建议"
    }}
  ],
  "combination_issue": {{
    "is_concept_combination_problem": false,
    "should_add_to_contradiction_pool": false,
    "reason": "如果问题来自概念组合本身而非生成文本，请说明"
  }},
  "short_summary": "一句话总结"
}}

输出规则：
- 只输出合法 JSON。
- 顶层 JSON 必须是：{{"results": [ ... ]}}
- 每个输入 prompt_id 必须且只能返回一个 result。
- prompt_id 必须完全保留。
- 不要使用 Markdown、代码块、注释或额外解释。
- 不要编造问题；没有问题时 issues 为空数组。
- 每个强制概念都必须有且只有一个 concept_checks item。
- 每个 challenge_elements item 都必须有且只有一个 challenge_checks item；没有 challenge_elements 时 challenge_checks 为空数组。
- 缺失概念的 evidence 必须为空字符串。
- 对包含多个具体项的 leaf 候选集合，不要因为没有覆盖全部候选项而判缺失；只有完全没有体现三级概念或没有体现任何合适叶子候选时，才判 missing/weak。
- 若挑战性元素在最终 prompt 中完全没有体现，在 issues 中使用 challenge_not_reflected；若挑战性元素与概念组合语义冲突，使用 challenge_incompatible；若 challenge reason 引入未被 concepts 选中的概念，不新增独立问题类型，应作为 concept_fidelity_focus 扣分，并在 issues 中使用 wrong_concept_substitution。
- 机械规则问题由系统确定性检查负责，例如长度、普通段落格式、结构化模板格式。批量记录中的 rule_precheck_summary 只用于提醒，不要在 issues 中复述这些机械检查结果；系统会在后处理自动合并。

批量记录：
{json.dumps(records, ensure_ascii=False, indent=2)}
"""
    return system_prompt, user_prompt


def rule_precheck(prompt_record: Dict[str, Any], difficulty_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic checks for mechanical format and length issues."""
    text = prompt_record.get("text") or ""
    template_family = _get_template_family(prompt_record)
    is_structured = template_family in STRUCTURED_TEMPLATE_IDS
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
    if not is_structured and ("\n" in text or "\r" in text):
        add_issue("format_violation", "Prompt contains line breaks.", "major")
    if not is_structured and re.search(r"(^|\s)(Prompt|Subject|Motion|Scene|Audio|Concepts|Description)\s*:", text):
        add_issue("format_violation", "Prompt contains field labels.", "major")
    if not is_structured and re.search(r"```|\*\*|^\s*[-*#]|\n\s*[-*#]", text):
        add_issue("format_violation", "Prompt contains Markdown-like formatting.", "major")
    if re.match(r"\s*[\[{]", text):
        add_issue("format_violation", "Prompt appears to be JSON or a structured object.", "major")
    if not is_structured and re.match(r"\s*\d+[\.)]\s+", text):
        add_issue("format_violation", "Prompt appears to start with numbered-list formatting.", "major")
    if is_structured:
        is_valid_structure, structure_errors = validate_structured_output(text, template_family)
        if not is_valid_structure:
            for error in structure_errors:
                add_issue(
                    "template_format_mismatch",
                    f"{template_family}: {error}",
                    "major",
                    "template_family",
                )
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_tokens = re.findall(r"[A-Za-z]{2,}", text)
    if cjk_chars and len(latin_tokens) > max(12, len(cjk_chars) // 4):
        add_issue("mixed_language", "Prompt contains heavy uncontrolled Chinese/Latin language mixing.", "minor")

    char_count = count_chinese_chars(text)
    difficulty = str((prompt_record.get("difficulty") or {}).get("level", "")).lower()
    params = (difficulty_params or {}).get(difficulty, {})
    max_len = params.get("text_length_max")
    min_len = params.get("text_length_min")
    if max_len is not None and char_count > max_len:
        add_issue(
            "difficulty_mismatch",
            f"Prompt has {char_count} Chinese characters, exceeding max length {max_len} for {difficulty.upper()}.",
            "minor",
            "difficulty",
        )
    if min_len is not None and char_count < min_len:
        add_issue(
            "difficulty_mismatch",
            f"Prompt has {char_count} Chinese characters, below min length {min_len} for {difficulty.upper()}.",
            "minor",
            "difficulty",
        )

    return {
        "passed": not any(issue["severity"] in {"fatal", "major"} for issue in issues),
        "length_unit": "chinese_characters",
        "length_estimate": char_count,
        "issues": issues,
    }


def _rule_precheck_summary(rule_precheck_data: Dict[str, Any]) -> Dict[str, Any]:
    """Expose only compact mechanical-check status to the LLM."""
    issues = rule_precheck_data.get("issues", []) if isinstance(rule_precheck_data, dict) else []
    return {
        "passed": rule_precheck_data.get("passed", True) if isinstance(rule_precheck_data, dict) else True,
        "length_unit": rule_precheck_data.get("length_unit", "chinese_characters") if isinstance(rule_precheck_data, dict) else "chinese_characters",
        "length_estimate": rule_precheck_data.get("length_estimate") if isinstance(rule_precheck_data, dict) else None,
        "issue_types": sorted({
            str(issue.get("type"))
            for issue in issues
            if isinstance(issue, dict) and issue.get("type")
        }),
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
    normalized = enforce_rule_precheck(normalized, precheck)

    result = {
        "prompt_id": prompt_record.get("prompt_id"),
        "combination_id": prompt_record.get("combination_id"),
        "source_llm": prompt_record.get("llm", {}),
        "difficulty": prompt_record.get("difficulty", {}),
        "text": prompt_record.get("text", ""),
        "concepts": prompt_record.get("concepts", {}),
        "challenge_elements": prompt_record.get("challenge_elements", []),
        "selection_trace": prompt_record.get("selection_trace", {}),
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
            try:
                result = judge_prompt_record(
                    provider=provider,
                    prompt_record=prompt_record,
                    difficulty_params=difficulty_params,
                    save_raw_response=save_raw_response,
                )
                result["judge_retry_reason"] = "missing_from_batch_response"
            except Exception as exc:
                result = _judge_error_result(
                    prompt_record,
                    ValueError(f"Missing judge result for prompt_id={prompt_id}; single retry failed: {exc}"),
                )
        else:
            normalized = normalize_judge_output(judge_item)
            normalized = enforce_rule_precheck(normalized, item["rule_precheck"])
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
        "challenge_checks": data.get("challenge_checks") or [],
        "issues": data.get("issues") or [],
        "combination_issue": data.get("combination_issue") or {
            "is_concept_combination_problem": False,
            "should_add_to_contradiction_pool": False,
            "reason": "",
        },
        "short_summary": data.get("short_summary", ""),
    }


def enforce_rule_precheck(normalized_judge: Dict[str, Any], precheck: Dict[str, Any]) -> Dict[str, Any]:
    """Merge deterministic precheck issues and enforce their effect on final decision."""
    result = dict(normalized_judge)
    precheck_issues = precheck.get("issues", []) if isinstance(precheck, dict) else []
    precheck_issue_types = {
        str(issue.get("type"))
        for issue in precheck_issues
        if isinstance(issue, dict) and issue.get("type")
    }
    issues = [
        dict(issue)
        for issue in result.get("issues", [])
        if isinstance(issue, dict) and not _is_duplicate_mechanical_issue(issue, precheck_issue_types)
    ]
    seen = {
        (
            issue.get("type", ""),
            issue.get("location", ""),
            issue.get("description", ""),
        )
        for issue in issues
    }

    for issue in precheck_issues:
        if not isinstance(issue, dict):
            continue
        merged_issue = {
            "severity": issue.get("severity", "minor"),
            "type": issue.get("type", "format_violation"),
            "location": issue.get("location", "rule_precheck"),
            "description": issue.get("description", ""),
            "suggested_fix": issue.get(
                "suggested_fix",
                "Fix this deterministic rule-precheck issue before accepting the prompt.",
            ),
            "source": "rule_precheck",
        }
        key = (
            merged_issue["type"],
            merged_issue["location"],
            merged_issue["description"],
        )
        if key not in seen:
            issues.append(merged_issue)
            seen.add(key)

    result["issues"] = issues
    severities = {issue.get("severity") for issue in issues}
    dimensions = dict(result.get("dimension_scores") or {})

    if "fatal" in severities:
        result["overall_decision"] = "FAIL"
        result["overall_score"] = min(_clamp_int(result.get("overall_score"), 1, 5, 1), 1)
        dimensions["video_prompt_usability"] = min(dimensions.get("video_prompt_usability", 1), 1)
    elif "major" in severities:
        result["overall_decision"] = "FAIL"
        result["overall_score"] = min(_clamp_int(result.get("overall_score"), 1, 5, 1), 2)
        dimensions["video_prompt_usability"] = min(dimensions.get("video_prompt_usability", 2), 2)
    elif "minor" in severities and result.get("overall_decision") == "PASS":
        result["overall_decision"] = "PASS_WITH_MINOR_ISSUES"
        result["overall_score"] = min(_clamp_int(result.get("overall_score"), 1, 5, 1), 4)

    result["dimension_scores"] = {
        key: _clamp_int(dimensions.get(key), 1, 5, default=1)
        for key in JUDGE_DIMENSION_KEYS
    }
    if issues and result.get("short_summary"):
        result["short_summary"] = str(result["short_summary"]).strip()
    elif issues:
        result["short_summary"] = "Rule precheck found deterministic issues."
    return result


def _is_duplicate_mechanical_issue(issue: Dict[str, Any], precheck_issue_types: set) -> bool:
    """Drop LLM-restated mechanical issues when rule_precheck owns the same type."""
    issue_type = str(issue.get("type", ""))
    if issue.get("source") == "rule_precheck":
        return False
    if issue_type not in precheck_issue_types:
        return False
    return issue_type in {
        "difficulty_mismatch",
        "format_violation",
        "template_format_mismatch",
        "mixed_language",
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
    target_input_tokens: int = DEFAULT_TARGET_INPUT_TOKENS,
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
    """Rough token estimate for mixed-language prompt payloads."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = max(0, len(text) - chinese_chars)
    return int(chinese_chars + other_chars / 4) + 1


def load_prompt_file(path: Path) -> List[Dict[str, Any]]:
    load_path = _resolve_review_prompt_path(path)
    with load_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    return _merge_selection_trace_sidecar(load_path, prompts)


def _resolve_review_prompt_path(path: Path) -> Path:
    """Prefer the full review output when the user passes the readable output."""
    if path.stem.endswith("_review"):
        return path

    review_path = derive_review_output_path(path)
    if review_path.exists():
        return review_path
    return path


def _merge_selection_trace_sidecar(path: Path, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trace_path = derive_selection_trace_path(path)
    if not trace_path.exists():
        return prompts

    try:
        with trace_path.open("r", encoding="utf-8") as f:
            trace_data = json.load(f)
    except Exception:
        return prompts

    traces = trace_data.get("traces", [])
    if not isinstance(traces, list):
        return prompts

    trace_by_prompt_id = {
        item.get("prompt_id"): item
        for item in traces
        if isinstance(item, dict) and item.get("prompt_id")
    }
    if not trace_by_prompt_id:
        return prompts

    merged = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            merged.append(prompt)
            continue

        trace_record = trace_by_prompt_id.get(prompt.get("prompt_id"))
        if not trace_record:
            merged.append(prompt)
            continue

        prompt_copy = dict(prompt)
        selection_trace = trace_record.get("selection_trace", {})
        prompt_copy["selection_trace"] = selection_trace
        sampling = dict(prompt_copy.get("sampling") or {})
        if selection_trace:
            sampling["selection_trace"] = selection_trace
        prompt_copy["sampling"] = sampling
        merged.append(prompt_copy)

    return merged


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


def derive_contradiction_pool_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".json"
    if output_path.stem == "full":
        return output_path.with_name(f"contradiction_pool{suffix}")
    return output_path.with_name(f"{output_path.stem}_contradiction_pool{suffix}")


def derive_compact_report_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".json"
    if output_path.stem == "full":
        return output_path.with_name(f"compact{suffix}")
    return output_path.with_name(f"{output_path.stem}_compact{suffix}")


def derive_failed_compact_report_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".json"
    if output_path.stem == "full":
        return output_path.with_name(f"failed_compact{suffix}")
    return output_path.with_name(f"{output_path.stem}_failed_compact{suffix}")


def write_judge_report(
    output_path: Path,
    results: List[Dict[str, Any]],
    source_info: Dict[str, Any],
    judge_info: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contradiction_templates = extract_contradiction_templates(results)
    contradiction_pool_path = derive_contradiction_pool_path(output_path)
    compact_report_path = derive_compact_report_path(output_path)
    failed_compact_report_path = derive_failed_compact_report_path(output_path)
    summary = summarize_results(results)
    report = {
        "generated_at": datetime.now().isoformat(),
        "judge": judge_info,
        "source": source_info,
        "summary": summary,
        "compact_output": str(compact_report_path),
        "failed_compact_output": str(failed_compact_report_path),
        "contradiction_pool_output": str(contradiction_pool_path),
        "contradiction_templates": contradiction_templates,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    compact_report = build_compact_judge_report(
        results,
        source_info,
        judge_info,
        summary,
        source_judge_report=output_path,
        decisions=None,
    )
    with compact_report_path.open("w", encoding="utf-8") as f:
        json.dump(compact_report, f, ensure_ascii=False, indent=2)

    failed_compact_report = build_compact_judge_report(
        results,
        source_info,
        judge_info,
        summary,
        source_judge_report=output_path,
        decisions={"FAIL", "PASS_WITH_MINOR_ISSUES"},
    )
    with failed_compact_report_path.open("w", encoding="utf-8") as f:
        json.dump(failed_compact_report, f, ensure_ascii=False, indent=2)

    pool_report = {
        "generated_at": datetime.now().isoformat(),
        "source_judge_report": str(output_path),
        "source": source_info,
        "total_templates": len(contradiction_templates),
        "templates": contradiction_templates,
    }
    with contradiction_pool_path.open("w", encoding="utf-8") as f:
        json.dump(pool_report, f, ensure_ascii=False, indent=2)


def build_compact_judge_report(
    results: List[Dict[str, Any]],
    source_info: Dict[str, Any],
    judge_info: Dict[str, Any],
    summary: Dict[str, Any],
    source_judge_report: Path,
    decisions: Optional[set] = None,
) -> Dict[str, Any]:
    selected_results = [
        result for result in results
        if decisions is None or (result.get("llm_judge") or {}).get("overall_decision") in decisions
    ]
    return {
        "generated_at": datetime.now().isoformat(),
        "source_judge_report": str(source_judge_report),
        "source": source_info,
        "judge": judge_info,
        "summary": summary,
        "included_decisions": sorted(decisions) if decisions else "ALL",
        "total_prompts": len(selected_results),
        "prompts": [_compact_judge_result(result) for result in selected_results],
    }


def _compact_judge_result(result: Dict[str, Any]) -> Dict[str, Any]:
    judge = result.get("llm_judge") or {}
    concepts = result.get("concepts") or (result.get("sampling") or {}).get("concepts", {})
    challenge_elements = result.get("challenge_elements") or (result.get("sampling") or {}).get("challenge_elements", [])
    return {
        "prompt_id": result.get("prompt_id"),
        "combination_id": result.get("combination_id"),
        "difficulty": (result.get("difficulty") or {}).get("level", result.get("difficulty")),
        "decision": judge.get("overall_decision"),
        "overall_score": judge.get("overall_score"),
        "scores": _compact_scores(judge.get("dimension_scores") or {}),
        "concepts": {
            category: _format_concept_for_reading(concept)
            for category, concept in concepts.items()
        },
        "challenges": [
            str(elem.get("id") or elem.get("name") or "")
            for elem in challenge_elements
            if isinstance(elem, dict)
        ],
        "prompt": result.get("text") or result.get("prompt") or "",
        "issues": [_compact_issue(issue) for issue in judge.get("issues", [])],
        "concept_status": _count_statuses(judge.get("concept_checks", [])),
        "challenge_status": _count_statuses(judge.get("challenge_checks", [])),
        "combination_issue": judge.get("combination_issue") or {},
        "summary": judge.get("short_summary", ""),
    }


def _compact_scores(dimension_scores: Dict[str, Any]) -> Dict[str, Any]:
    score_map = {
        "concept": dimension_scores.get("concept_fidelity_focus", dimension_scores.get("concept_fidelity")),
        "consistency": dimension_scores.get("internal_consistency"),
        "clarity": dimension_scores.get("clarity_concreteness", dimension_scores.get("clarity")),
        "language": dimension_scores.get("language_quality"),
        "usability": dimension_scores.get("video_prompt_usability"),
    }
    if "difficulty_alignment" in dimension_scores:
        score_map["difficulty"] = dimension_scores.get("difficulty_alignment")
    return {key: value for key, value in score_map.items() if value is not None}


def _format_concept_for_reading(concept: Any) -> str:
    if not isinstance(concept, dict):
        return str(concept)
    path = concept.get("path")
    if isinstance(path, list) and path:
        text = " > ".join(str(item) for item in path)
    else:
        text = str(concept.get("full_path") or concept.get("level3_category") or "")
    leaf = concept.get("leaf")
    if leaf:
        text = f"{text} > 叶子候选：{leaf}" if text else f"叶子候选：{leaf}"
    return text


def _compact_issue(issue: Any) -> Dict[str, Any]:
    if not isinstance(issue, dict):
        return {"desc": str(issue)}
    compact = {
        "type": issue.get("type"),
        "severity": issue.get("severity"),
        "location": issue.get("location"),
        "desc": issue.get("description") or issue.get("desc"),
        "source": issue.get("source"),
    }
    return {
        key: value for key, value in compact.items()
        if value not in (None, "", [])
    }


def _count_statuses(checks: Any) -> Dict[str, int]:
    counts = Counter()
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, dict):
                counts[str(check.get("status", "unknown"))] += 1
    return dict(counts)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    decision_counts = Counter()
    issue_counts = Counter()
    concept_status_counts = Counter()
    challenge_status_counts = Counter()
    score_sums = Counter()
    score_counts = Counter()
    combination_issue_counts = Counter()

    for result in results:
        judge = result.get("llm_judge", {})
        decision_counts[judge.get("overall_decision", "UNKNOWN")] += 1
        combination_issue = judge.get("combination_issue") or {}
        if combination_issue.get("is_concept_combination_problem"):
            combination_issue_counts["concept_combination_problem"] += 1
        if combination_issue.get("should_add_to_contradiction_pool"):
            combination_issue_counts["should_add_to_contradiction_pool"] += 1
        for issue in judge.get("issues", []):
            issue_counts[issue.get("type", "unknown")] += 1
        for check in judge.get("concept_checks", []):
            concept_status_counts[check.get("status", "unknown")] += 1
        for check in judge.get("challenge_checks", []):
            challenge_status_counts[check.get("status", "unknown")] += 1
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
        "combination_issue_counts": dict(combination_issue_counts),
        "concept_status_counts": dict(concept_status_counts),
        "challenge_status_counts": dict(challenge_status_counts),
        "average_dimension_scores": avg_scores,
    }


def extract_contradiction_templates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract concept-combination failures that can seed a contradiction pool."""
    templates = []
    seen: Dict[str, Dict[str, Any]] = {}
    for result in results:
        judge = result.get("llm_judge", {})
        combination_issue = judge.get("combination_issue") or {}
        if not combination_issue.get("should_add_to_contradiction_pool"):
            continue

        concepts = result.get("concepts") or (result.get("sampling") or {}).get("concepts", {})
        issue_types = sorted({
            issue.get("type", "unknown")
            for issue in judge.get("issues", [])
        })
        concept_paths = {
            category: concept.get("path") or concept.get("full_path")
            for category, concept in concepts.items()
        }
        challenge_elements = result.get("challenge_elements") or (result.get("sampling") or {}).get("challenge_elements", [])
        challenge_ids = sorted(
            str(elem.get("id") or elem.get("name") or "")
            for elem in challenge_elements
            if isinstance(elem, dict)
        )
        dedupe_payload = {
            "concept_paths": concept_paths,
            "challenge_ids": challenge_ids,
            "issue_types": issue_types,
        }
        dedupe_key = hashlib.md5(
            json.dumps(dedupe_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        if dedupe_key in seen:
            existing = seen[dedupe_key]
            existing["source_prompt_ids"].append(result.get("prompt_id"))
            existing["source_combination_ids"].append(result.get("combination_id"))
            if combination_issue.get("reason"):
                existing["reasons"].append(combination_issue.get("reason"))
            continue

        template = {
            "template_id": f"T-{len(templates) + 1:05d}",
            "dedupe_key": dedupe_key,
            "source_prompt_id": result.get("prompt_id"),
            "source_prompt_ids": [result.get("prompt_id")],
            "source_combination_id": result.get("combination_id"),
            "source_combination_ids": [result.get("combination_id")],
            "issue_types": issue_types,
            "concept_paths": concept_paths,
            "challenge_elements": challenge_elements,
            "reason": combination_issue.get("reason", ""),
            "reasons": [combination_issue.get("reason", "")],
            "action": "review_or_avoid_similar",
            "created_at": datetime.now().isoformat(),
        }
        templates.append(template)
        seen[dedupe_key] = template
    return templates


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
        "concepts": prompt_record.get("concepts", {}),
        "challenge_elements": prompt_record.get("challenge_elements", []),
        "selection_trace": prompt_record.get("selection_trace", {}),
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
        "concepts": prompt_record.get("concepts", {}),
        "challenge_elements": prompt_record.get("challenge_elements", []),
        "selection_trace": prompt_record.get("selection_trace", {}),
        "sampling": prompt_record.get("sampling", {}),
        "rule_precheck": {},
        "llm_judge": {
            "overall_decision": "FAIL",
            "overall_score": 1,
            "dimension_scores": {key: 1 for key in JUDGE_DIMENSION_KEYS},
            "concept_checks": [],
            "challenge_checks": [],
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
    concepts = prompt_record.get("concepts") or sampling.get("concepts", {})
    challenge_elements = prompt_record.get("challenge_elements") or sampling.get("challenge_elements", [])
    template_family = _get_template_family(prompt_record)
    return {
        "prompt_id": prompt_record.get("prompt_id"),
        "combination_id": prompt_record.get("combination_id"),
        "llm": prompt_record.get("llm", {}),
        "difficulty": prompt_record.get("difficulty", {}),
        "template_family": template_family,
        "categories_selected": sampling.get("categories_selected") or list(concepts.keys()),
        "concepts": concepts,
        "challenge_elements": challenge_elements,
        "selection_trace": prompt_record.get("selection_trace") or sampling.get("selection_trace", {}),
        "text": prompt_record.get("text", ""),
        "text_length": prompt_record.get("text_length"),
        "revision": prompt_record.get("revision", {}),
        "rewrite": prompt_record.get("rewrite", {}),
    }


def _get_template_family(prompt_record: Dict[str, Any]) -> str:
    family = prompt_record.get("template_family")
    if isinstance(family, str) and family:
        return family

    revision = prompt_record.get("revision") or {}
    family = revision.get("target_template_family") or revision.get("template_family")
    if isinstance(family, str) and family:
        return family

    rewrite = prompt_record.get("rewrite") or {}
    family = rewrite.get("target_template_family") or rewrite.get("template_family")
    if isinstance(family, str) and family:
        return family

    return "plain_paragraph"


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
