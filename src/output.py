"""
输出处理模块
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class OutputWriter:
    """输出写入器"""
    
    def __init__(self, output_path: str, include_stats: bool = True):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.include_stats = include_stats
        self.prompts: List[Dict] = []
        self.stats: Dict[str, Any] = {}
    
    def add_prompt(self, prompt_dict: Dict) -> None:
        """添加一条prompt"""
        self.prompts.append(prompt_dict)
    
    def set_stats(self, stats: Dict[str, Any]) -> None:
        """设置统计信息"""
        self.stats = stats
    
    def write(self) -> None:
        """写入输出文件"""
        output = {
            'generated_at': datetime.now().isoformat(),
            'total_prompts': len(self.prompts),
            'prompts': self.prompts
        }
        
        if self.include_stats and self.stats:
            output['stats'] = self.stats
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"输出已写入: {self.output_path}")
        print(f"总计生成: {len(self.prompts)} 条prompt")


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
        categories = sampling.get('categories_selected', [])
        for cat in categories:
            self.category_count[cat] = self.category_count.get(cat, 0) + 1
        
        # 三级类目覆盖统计
        concepts = sampling.get('concepts', {})
        for cat, concept in concepts.items():
            level3 = concept.get('level3_category', 'unknown')
            if cat not in self.level3_coverage:
                self.level3_coverage[cat] = {}
            self.level3_coverage[cat][level3] = self.level3_coverage[cat].get(level3, 0) + 1
        
        # 挑战性要素统计
        challenges = sampling.get('challenge_elements', [])
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
