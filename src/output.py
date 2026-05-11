"""
输出处理模块
"""
import copy
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def derive_selection_trace_path(output_path: Path) -> Path:
    """Derive the sidecar path for selection reasons and trace metadata."""
    suffix = output_path.suffix or ".json"
    return output_path.with_name(f"{output_path.stem}_selection_trace{suffix}")


def derive_review_output_path(output_path: Path) -> Path:
    """Derive the full review/audit output path from the readable output path."""
    suffix = output_path.suffix or ".json"
    return output_path.with_name(f"{output_path.stem}_review{suffix}")


class OutputWriter:
    """输出写入器"""
    
    def __init__(
        self,
        output_path: str,
        include_stats: bool = True,
        review_output_path: Optional[str] = None,
        trace_output_path: Optional[str] = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_output_path = (
            Path(review_output_path)
            if review_output_path
            else derive_review_output_path(self.output_path)
        )
        self.review_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.include_stats = include_stats
        self.prompts: List[Dict] = []  # Full review records; kept for coverage rebuild/resume.
        self.read_prompts: List[Dict] = []
        self.stats: Dict[str, Any] = {}
    
    def add_prompt(self, prompt_dict: Dict) -> None:
        """添加一条prompt"""
        review_record = copy.deepcopy(prompt_dict)
        self.prompts.append(review_record)
        self.read_prompts.append(self._build_read_prompt(review_record))

    def load_existing_traces(self) -> None:
        """Backward-compatible no-op; review records now keep trace inline."""
        return
    
    def set_stats(self, stats: Dict[str, Any]) -> None:
        """设置统计信息"""
        self.stats = stats
    
    def write(self) -> None:
        """写入输出文件"""
        output = {
            'generated_at': datetime.now().isoformat(),
            'review_output': str(self.review_output_path),
            'total_prompts': len(self.read_prompts),
            'prompts': self.read_prompts,
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self._write_review()
        
        print(f"阅读版输出已写入: {self.output_path}")
        print(f"Review输出已写入: {self.review_output_path}")
        print(f"总计生成: {len(self.prompts)} 条prompt")

    def _write_review(self) -> None:
        output = {
            'generated_at': datetime.now().isoformat(),
            'readable_output': str(self.output_path),
            'total_prompts': len(self.prompts),
            'prompts': self.prompts,
        }
        if self.include_stats and self.stats:
            output['stats'] = self.stats

        with self.review_output_path.open('w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _build_read_prompt(self, prompt_dict: Dict) -> Dict:
        """Build a compact record for manual reading."""
        sampling = prompt_dict.get('sampling') if isinstance(prompt_dict.get('sampling'), dict) else {}
        concepts = prompt_dict.get('concepts') or sampling.get('concepts', {})
        if not concepts and isinstance(sampling, dict):
            concepts = sampling.get('concepts', {})

        return {
            'prompt_id': prompt_dict.get('prompt_id'),
            'concepts': concepts,
            'prompt': prompt_dict.get('text', ''),
        }


class StatsCollector:
    """统计收集器"""
    
    def __init__(self):
        self.difficulty_count = {'low': 0, 'medium': 0, 'high': 0}
        self.category_count: Dict[str, int] = {}
        self.level3_coverage: Dict[str, Dict[str, int]] = {}
        self.challenge_count: Dict[str, int] = {}
        self.llm_count: Dict[str, int] = {}
        self.coverage_snapshot: Dict[str, Any] = {}
    
    def record(self, prompt_dict: Dict) -> None:
        """记录一条prompt的统计"""
        # 难度统计
        level = prompt_dict.get('difficulty', {}).get('level', 'unknown').lower()
        if level in self.difficulty_count:
            self.difficulty_count[level] += 1
        
        # 类目统计
        sampling = prompt_dict.get('sampling', {})
        concepts = prompt_dict.get('concepts') or sampling.get('concepts', {})
        categories = sampling.get('categories_selected') or list(concepts.keys())
        for cat in categories:
            self.category_count[cat] = self.category_count.get(cat, 0) + 1
        
        # 三级类目覆盖统计
        for cat, concept in concepts.items():
            level3 = concept.get('level3_category', 'unknown')
            if cat not in self.level3_coverage:
                self.level3_coverage[cat] = {}
            self.level3_coverage[cat][level3] = self.level3_coverage[cat].get(level3, 0) + 1
        
        # 挑战性要素统计
        challenges = prompt_dict.get('challenge_elements') or sampling.get('challenge_elements', [])
        for ch in challenges:
            ch_id = ch.get('id', 'unknown')
            self.challenge_count[ch_id] = self.challenge_count.get(ch_id, 0) + 1
        
        # LLM统计
        llm = prompt_dict.get('llm', {}).get('provider', 'unknown')
        self.llm_count[llm] = self.llm_count.get(llm, 0) + 1
    
    def record_coverage(self, coverage_report: Dict[str, Any]) -> None:
        """记录覆盖追踪器的快照"""
        self.coverage_snapshot = coverage_report

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'difficulty_distribution': self.difficulty_count,
            'category_distribution': self.category_count,
            'level3_coverage': {
                cat: {
                    'total_categories': len(covered),
                    'sample_counts_preview': list(covered.values())[:5] if covered else []
                }
                for cat, covered in self.level3_coverage.items()
            },
            'challenge_distribution': self.challenge_count,
            'llm_distribution': self.llm_count,
            'level3_coverage_detail': self.coverage_snapshot if self.coverage_snapshot else None,
        }
