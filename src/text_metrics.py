"""
Text length utilities for generated Chinese prompts.
"""
import re


def count_chinese_chars(text: str) -> int:
    """Count CJK ideographs as the canonical Chinese prompt length."""
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))
