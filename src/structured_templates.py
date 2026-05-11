"""
Structured prompt template families and rewrite helpers.

These templates are derived from the examples in docs/example.md:
- shot_timeline: multi-shot timeline
- split_screen_explanation: explanatory split-screen layout
- field_spec: field-like structured specification
- script_timeline: time-block/script format

cinematic_paragraph is kept only for backward compatibility with old outputs. It
is intentionally excluded from explicit structured rewrite targets because it is
too close to a normal paragraph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TemplateFamilySpec:
    family_id: str
    display_name: str
    summary: str
    structure_hint: str
    selection_hints: Tuple[str, ...]
    rewrite_hints: Tuple[str, ...]


TEMPLATE_FAMILY_SPECS: Dict[str, TemplateFamilySpec] = {
    "shot_timeline": TemplateFamilySpec(
        family_id="shot_timeline",
        display_name="镜头时间轴",
        summary="按镜头序列组织，每个镜头写清时长、景别、动作、镜头运动和视觉重点。",
        structure_hint="适合多阶段动作、视角切换、过程推进。推荐写法：镜头1：2秒，...；镜头2：...",
        selection_hints=("镜头", "分镜", "shot", "时长", "景别", "镜头运动", "推进"),
        rewrite_hints=(
            "用镜头号或分镜号拆开叙述。",
            "每个镜头最好包含时长、景别、动作和镜头运动。",
            "保持画面连贯，但不要变成解说词。",
        ),
    ),
    "split_screen_explanation": TemplateFamilySpec(
        family_id="split_screen_explanation",
        display_name="分屏解释",
        summary="使用左右屏、上下屏或并列画面对照说明同一主题的两个侧面。",
        structure_hint="适合对比、解释、信息呈现。推荐写法：画面分屏。左屏：... 右屏：... 作用：...",
        selection_hints=("分屏", "左屏", "右屏", "上下屏", "对照", "并列", "对比", "解释"),
        rewrite_hints=(
            "用分屏或对照结构表达两个并列画面。",
            "清楚写出左/右或上/下两侧内容，以及它们的关系。",
            "强调信息解释或视觉对照，而不是单线叙事。",
        ),
    ),
    "field_spec": TemplateFamilySpec(
        family_id="field_spec",
        display_name="字段式设定",
        summary="用若干字段组织风格、灯光、镜头、人物、场景和情绪，像创作设定卡。",
        structure_hint="适合风格设定、镜头配置、人物配置。推荐写法：整体风格：...；镜头：...；灯光：...",
        selection_hints=("风格", "灯光", "镜头", "人物", "场景", "情绪", "色调", "设定", "style", "lens"),
        rewrite_hints=(
            "用中文字段名组织内容，每个字段短句表达。",
            "字段之间保持并列，不要堆成纯 JSON。",
            "字段可以覆盖风格、镜头、人物、环境、情绪等。",
        ),
    ),
    "cinematic_paragraph": TemplateFamilySpec(
        family_id="cinematic_paragraph",
        display_name="电影感段落",
        summary="保持单段或少量段落的连续叙述，让主体、环境、动作、光影和情绪自然融合。",
        structure_hint="适合单场景、强氛围、强调画面质感的提示词。推荐写法：一段连贯的电影感叙述。",
        selection_hints=("电影级", "电影质感", "概念艺术", "超现实", "光影", "氛围", "叙述"),
        rewrite_hints=(
            "尽量保持单段连续叙述。",
            "强调画面、氛围、动作和镜头感的自然融合。",
            "避免明显的字段化或脚本化分段。",
        ),
    ),
    "script_timeline": TemplateFamilySpec(
        family_id="script_timeline",
        display_name="时间脚本",
        summary="按时间段、Hook、Pain Points、Solution、CTA 等脚本模块组织，适合广告或口播。",
        structure_hint="适合广告、口播、宣传和节奏分段表达。推荐写法：0-3秒（Hook）... 3-10秒（Pain Points）...",
        selection_hints=("脚本", "Hook", "CTA", "Pain Points", "Solution", "口播", "0-", "秒", "script"),
        rewrite_hints=(
            "按时间段或脚本模块分块写。",
            "每段要有明确功能，如钩子、问题、解决方案或收尾。",
            "保持节奏感和商业脚本感。",
        ),
    ),
}

STRUCTURED_TEMPLATE_IDS = (
    "shot_timeline",
    "split_screen_explanation",
    "field_spec",
    "script_timeline",
)
LEGACY_TEMPLATE_IDS = ("cinematic_paragraph",)
DEFAULT_STRUCTURED_TEMPLATE_ID = "field_spec"


SHOT_MARKER_RE = re.compile(r"(镜头\s*\d+|shot\s*\d+|第\s*\d+\s*镜头|分镜)", re.IGNORECASE)
TIME_BLOCK_RE = re.compile(r"(\b\d+\s*[-–~]\s*\d+\s*(?:秒|sec|s)\b|\b\d+\s*秒\b)", re.IGNORECASE)
SPLIT_SCREEN_RE = re.compile(r"(分屏|左屏|右屏|上屏|下屏|对照|并列|对比)")
SCRIPT_RE = re.compile(r"(Script:|脚本|Hook|CTA|Pain Points|Solution|0-\d+\s*秒|\d+\s*-\s*\d+\s*秒)", re.IGNORECASE)
FIELD_LABEL_RE = re.compile(r"(?:^|[\n；;])\s*([^：:\n；;]{2,14})\s*[：:]", re.MULTILINE)


def get_prompt_text(prompt_record: Dict[str, Any]) -> str:
    """Extract prompt text from either review or readable prompt records."""
    return (
        prompt_record.get("text")
        or prompt_record.get("prompt")
        or ""
    )


def get_prompt_concepts(prompt_record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract concept mapping from either review or readable prompt records."""
    concepts = prompt_record.get("concepts")
    if isinstance(concepts, dict) and concepts:
        return concepts

    sampling = prompt_record.get("sampling") or {}
    concepts = sampling.get("concepts")
    if isinstance(concepts, dict):
        return concepts
    return {}


def get_prompt_challenges(prompt_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract challenge elements from either review or readable prompt records."""
    challenges = prompt_record.get("challenge_elements")
    if isinstance(challenges, list):
        return challenges

    sampling = prompt_record.get("sampling") or {}
    challenges = sampling.get("challenge_elements")
    if isinstance(challenges, list):
        return challenges
    return []


def format_concept_lines(prompt_record: Dict[str, Any]) -> List[str]:
    """Format selected concepts into concise readable lines."""
    concepts = get_prompt_concepts(prompt_record)
    lines: List[str] = []
    for category_key, concept in concepts.items():
        if not isinstance(concept, dict):
            continue
        path = concept.get("full_path") or " > ".join(concept.get("path") or [])
        if not path:
            path = concept.get("level3_category", "")
        leaf = concept.get("leaf")
        if leaf:
            path = f"{path} > 叶子候选：{leaf}" if path else f"叶子候选：{leaf}"
        lines.append(f"- {category_key}：{path}")
    return lines


def format_challenge_lines(prompt_record: Dict[str, Any]) -> List[str]:
    """Format challenge elements into concise readable lines."""
    challenges = get_prompt_challenges(prompt_record)
    lines: List[str] = []
    for elem in challenges:
        if not isinstance(elem, dict):
            continue
        lines.append(
            f"- {elem.get('id', '')} | {elem.get('name', '')}: {elem.get('description', '')}"
        )
    return lines


def score_structured_families(prompt_text: str) -> Dict[str, int]:
    """Score each structured family for a source prompt."""
    text = prompt_text or ""
    lower = text.lower()
    colon_count = text.count("：") + text.count(":")
    newline_count = text.count("\n")
    field_label_count = len(FIELD_LABEL_RE.findall(text))

    scores = {family_id: 0 for family_id in STRUCTURED_TEMPLATE_IDS}

    # Shot timeline
    if SHOT_MARKER_RE.search(text):
        scores["shot_timeline"] += 6
    if TIME_BLOCK_RE.search(text):
        scores["shot_timeline"] += 3
    if newline_count >= 2:
        scores["shot_timeline"] += 1
    if "镜头" in text or "shot" in lower:
        scores["shot_timeline"] += 1

    # Split screen explanation
    if SPLIT_SCREEN_RE.search(text):
        scores["split_screen_explanation"] += 7
    if "分屏" in text or "对照" in text:
        scores["split_screen_explanation"] += 2
    if "左屏" in text or "右屏" in text:
        scores["split_screen_explanation"] += 2

    # Field spec
    if colon_count >= 4:
        scores["field_spec"] += 4
    if field_label_count >= 3:
        scores["field_spec"] += 6
    elif field_label_count > 0:
        scores["field_spec"] += 2
    if any(keyword in lower for keyword in ("global_style", "lighting_progression", "color_grading", "lens", "characters", "scenes")):
        scores["field_spec"] += 3

    # Script timeline
    if SCRIPT_RE.search(text):
        scores["script_timeline"] += 7
    if any(keyword in lower for keyword in ("hook", "cta", "pain points", "solution", "script")):
        scores["script_timeline"] += 4
    if "口播" in text or "脚本" in text:
        scores["script_timeline"] += 3

    return scores


def validate_structured_output(text: str, family_id: str) -> Tuple[bool, List[str]]:
    """Validate that rewritten text has visible structure for its target family."""
    stripped = (text or "").strip()
    errors: List[str] = []
    if not stripped:
        return False, ["输出为空"]

    if family_id not in STRUCTURED_TEMPLATE_IDS:
        return False, [f"{family_id} 不是显式结构化模板族"]

    if family_id == "shot_timeline":
        marker_count = len(SHOT_MARKER_RE.findall(stripped))
        if marker_count < 2:
            errors.append("镜头时间轴必须至少包含两个镜头/分镜编号，例如“镜头1：... 镜头2：...”。")
        if "：" not in stripped and ":" not in stripped:
            errors.append("镜头时间轴需要用冒号或等价结构写清每个镜头内容。")

    elif family_id == "split_screen_explanation":
        has_left_right = "左屏" in stripped and "右屏" in stripped
        has_top_bottom = "上屏" in stripped and "下屏" in stripped
        if not (has_left_right or has_top_bottom):
            errors.append("分屏解释必须明确包含左屏/右屏或上屏/下屏的成对标签。")
        if "分屏" not in stripped and "对照" not in stripped and "并列" not in stripped:
            errors.append("分屏解释需要明确说明分屏、对照或并列关系。")

    elif family_id == "field_spec":
        labels = {
            match.group(1).strip()
            for match in FIELD_LABEL_RE.finditer(stripped)
            if match.group(1).strip()
        }
        if len(labels) < 3:
            errors.append("字段式设定必须至少包含三个字段标签，例如“主体：”“场景：”“动作：”“镜头：”。")

    elif family_id == "script_timeline":
        time_block_count = len(TIME_BLOCK_RE.findall(stripped))
        has_script_marker = bool(SCRIPT_RE.search(stripped))
        if time_block_count < 2:
            errors.append("时间脚本必须至少包含两个时间段，例如“0-3秒”“3-6秒”。")
        if not has_script_marker:
            errors.append("时间脚本需要包含脚本功能标记，例如 Hook、CTA、开场、转折或收尾。")

    return not errors, errors


def structure_requirements_text(family_id: str) -> str:
    """Render deterministic format requirements used by rewrite prompts and retries."""
    if family_id == "shot_timeline":
        return (
            "- 必须显式写成至少两个镜头/分镜段落。\n"
            "- 每段必须带编号，例如“镜头1：”“镜头2：”或“分镜1：”“分镜2：”。\n"
            "- 每个镜头应写清时长/景别/动作/镜头运动中的至少两项。"
        )
    if family_id == "split_screen_explanation":
        return (
            "- 必须显式使用分屏结构。\n"
            "- 必须包含成对标签“左屏：/右屏：”或“上屏：/下屏：”。\n"
            "- 必须说明两个画面之间的对照、并列或解释关系。"
        )
    if family_id == "field_spec":
        return (
            "- 必须显式写成字段式设定。\n"
            "- 至少包含三个中文字段标签，例如“主体：”“场景：”“动作：”“镜头：”“光线：”“氛围：”。\n"
            "- 字段内容应是可用于视频生成的短句，不要输出 JSON。"
        )
    if family_id == "script_timeline":
        return (
            "- 必须显式写成时间脚本。\n"
            "- 至少包含两个时间段，例如“0-3秒：”“3-6秒：”。\n"
            "- 每段需要体现脚本功能，例如 Hook、铺垫、动作推进、转折、收尾或 CTA。"
        )
    return "- 必须使用可见的结构化标签或分段，不要输出普通自然段。"


def choose_structured_family(prompt_record: Dict[str, Any]) -> Tuple[str, Dict[str, int], str]:
    """Choose a structured template family for a prompt.

    Returns:
        (family_id, scores, reason)
    """
    prompt_text = get_prompt_text(prompt_record)
    scores = score_structured_families(prompt_text)
    prompt_id = str(prompt_record.get("prompt_id") or "")

    best_score = max(scores.values()) if scores else 0
    if best_score < 3:
        top_candidates = list(STRUCTURED_TEMPLATE_IDS)
    else:
        top_candidates = [
            family_id for family_id, score in scores.items()
            if score >= max(1, best_score - 1)
        ]
    if not top_candidates:
        top_candidates = [DEFAULT_STRUCTURED_TEMPLATE_ID]

    if len(top_candidates) == 1:
        chosen = top_candidates[0]
    else:
        salt = prompt_id or prompt_text[:64]
        digest = hashlib.md5(salt.encode("utf-8")).hexdigest()
        chosen = top_candidates[int(digest, 16) % len(top_candidates)]

    reason = (
        f"scores={scores}; top_candidates={top_candidates}; chosen={chosen}"
    )
    return chosen, scores, reason


def family_catalog_text(target_family_id: str) -> str:
    """Render a concise catalog plus the target family spec for prompts."""
    lines = ["可选模板族："]
    for family_id in STRUCTURED_TEMPLATE_IDS:
        spec = TEMPLATE_FAMILY_SPECS[family_id]
        lines.append(
            f"- {spec.display_name} ({family_id})：{spec.summary} 写法提示：{spec.structure_hint}"
        )
    target_spec = TEMPLATE_FAMILY_SPECS[target_family_id]
    lines.append("")
    lines.append(f"本次目标模板族：{target_spec.display_name} ({target_spec.family_id})")
    lines.append(f"目标说明：{target_spec.summary}")
    lines.append(f"目标写法：{target_spec.structure_hint}")
    lines.append("输出时只允许采用本次目标模板族的结构，不要混入其他模板族的组织方式。")
    lines.append("注意：普通电影感自然段不属于本次结构化 rewrite 的合格输出。")
    return "\n".join(lines)


def build_structured_rewrite_prompt(
    prompt_record: Dict[str, Any],
    target_family_id: str,
    target_length_hint: str,
) -> str:
    """Build the rewrite prompt for a structured prompt family."""
    spec = TEMPLATE_FAMILY_SPECS[target_family_id]
    concepts_lines = format_concept_lines(prompt_record)
    challenge_lines = format_challenge_lines(prompt_record)
    original_text = get_prompt_text(prompt_record)
    prompt_id = prompt_record.get("prompt_id", "")
    difficulty = ((prompt_record.get("difficulty") or {}).get("level", "") or "").upper()
    llm_provider = (prompt_record.get("llm") or {}).get("provider", "")

    sections = [
        "你是专业的中文视频提示词结构化改写器。你的任务不是创作新概念，而是把已有 prompt 改写成更清晰、更专业、更适合视频生成的结构化表达。",
        "",
        "硬性要求：",
        "- 必须保留原始 prompt 中已经选中的所有概念与挑战性要素，不要缺失、替换或新增未选中的概念。",
        "- 如果已选概念包含“叶子候选”，它表示该三级概念下的一组可选具体表达；候选中有多个并列项时，只需保留或自然体现其中一个或少数具体项，不要求全部写出。",
        "- 必须保持中文语义，不要翻译成英文，也不要混入无关语言。",
        "- 只允许使用本次指定模板族的结构，不要混用其他模板族的组织方式。",
        "- 必须让结构在文本中肉眼可见；普通连续自然段不算结构化改写成功。",
        "- 如果原 prompt 是单段文本，改写后可以变成多行或分段结构；如果原 prompt 已经包含结构化线索，只保留与目标模板族一致的部分。",
        f"- 总长度尽量与原文接近；参考目标长度：{target_length_hint}。",
        "- 只输出最终改写后的 prompt，不要解释、不要备注、不要 JSON、不要代码块。",
        "",
        "模板族说明：",
        family_catalog_text(target_family_id),
        "",
        "本模板族格式硬性要求：",
        structure_requirements_text(target_family_id),
        "",
        "源记录信息：",
        f"- prompt_id：{prompt_id}",
        f"- source_llm：{llm_provider}",
        f"- difficulty：{difficulty}",
    ]

    if concepts_lines:
        sections.append("")
        sections.append("已选概念：")
        sections.extend(concepts_lines)

    if challenge_lines:
        sections.append("")
        sections.append("挑战性要素：")
        sections.extend(challenge_lines)

    sections.extend([
        "",
        "原始提示词：",
        original_text,
        "",
        "改写要求：",
        *[f"- {hint}" for hint in spec.rewrite_hints],
        "",
        "请直接输出改写后的提示词文本：",
    ])

    return "\n".join(sections)
