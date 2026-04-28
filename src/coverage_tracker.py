"""
Level3 概念覆盖追踪器
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from .concept_loader import ConceptLoader, ConceptNode


@dataclass
class Level3CoverageRecord:
    name: str
    path: List[str]
    times_covered: int = 0
    as_core: int = 0
    as_companion: int = 0
    first_covered_at: Optional[str] = None
    last_covered_at: Optional[str] = None


class CoverageTracker:
    """追踪所有 sheet 的 level3 覆盖状态，持久化到 JSON"""

    EXCLUDE_KEYWORDS = ['其他', '其它', 'Other', 'other']

    def __init__(self, loader: ConceptLoader, state_path: str = "data/coverage_state.json"):
        self.loader = loader
        self.state_path = Path(state_path)
        self.state: Dict[str, Dict[str, Level3CoverageRecord]] = {}
        self._init_from_loader()
        self._load_state()

    def _init_from_loader(self):
        """从 ConceptLoader 构建覆盖状态骨架"""
        self.state = {}
        for sheet_name, category in self.loader.categories.items():
            self.state[sheet_name] = {}
            for level3_node in category.level3_categories:
                if any(kw in level3_node.name for kw in self.EXCLUDE_KEYWORDS):
                    continue
                self.state[sheet_name][level3_node.name] = Level3CoverageRecord(
                    name=level3_node.name,
                    path=list(level3_node.path)
                )

    def _load_state(self):
        """从 JSON 恢复已有的覆盖记录"""
        if not self.state_path.exists():
            return
        with open(self.state_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        now = datetime.now().isoformat()
        for sheet_name, records in data.get('sheets', {}).items():
            if sheet_name not in self.state:
                continue
            for name, rec in records.items():
                if name not in self.state[sheet_name]:
                    continue
                existing = self.state[sheet_name][name]
                existing.times_covered = rec.get('times_covered', 0)
                existing.as_core = rec.get('as_core', 0)
                existing.as_companion = rec.get('as_companion', 0)
                existing.first_covered_at = rec.get('first_covered_at')
                existing.last_covered_at = rec.get('last_covered_at')

    def save(self):
        """持久化覆盖状态到 JSON"""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'last_updated': datetime.now().isoformat(),
            'sheets': {}
        }
        for sheet_name, records in self.state.items():
            data['sheets'][sheet_name] = {}
            for name, rec in records.items():
                data['sheets'][sheet_name][name] = {
                    'times_covered': rec.times_covered,
                    'as_core': rec.as_core,
                    'as_companion': rec.as_companion,
                    'path': rec.path,
                    'first_covered_at': rec.first_covered_at,
                    'last_covered_at': rec.last_covered_at,
                }
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 查询 ──────────────────────────────────────────

    def get_uncovered(self, sheet_name: str) -> List[Level3CoverageRecord]:
        """返回未覆盖的 level3 列表（零覆盖优先，其次最少覆盖）"""
        if sheet_name not in self.state:
            return []
        records = list(self.state[sheet_name].values())
        records.sort(key=lambda r: r.times_covered)
        return [r for r in records if r.times_covered == 0]

    def get_uncovered_count(self, sheet_name: str) -> int:
        if sheet_name not in self.state:
            return 0
        return sum(1 for r in self.state[sheet_name].values() if r.times_covered == 0)

    def get_total_count(self, sheet_name: str) -> int:
        if sheet_name not in self.state:
            return 0
        return len(self.state[sheet_name])

    def get_total_coverage_ratio(self) -> float:
        total = 0
        covered = 0
        for sheet_name, records in self.state.items():
            total += len(records)
            covered += sum(1 for r in records.values() if r.times_covered > 0)
        return covered / max(1, total)

    def get_times_covered(self, sheet_name: str, level3_name: str) -> int:
        if sheet_name not in self.state:
            return 0
        rec = self.state[sheet_name].get(level3_name)
        return rec.times_covered if rec else 0

    def get_uncovered_level3_under_level2(self, sheet_name: str, level2_path: List[str]) -> List[str]:
        """返回某二级类目下未覆盖的三级名称列表"""
        if sheet_name not in self.state:
            return []
        uncovered = []
        for rec in self.state[sheet_name].values():
            # rec.path 形如 [一级, 二级, 三级]，比较前两级
            if len(rec.path) >= 2 and len(level2_path) >= 2:
                if rec.path[0] == level2_path[0] and rec.path[1] == level2_path[1]:
                    if rec.times_covered == 0:
                        uncovered.append(rec.name)
        return uncovered

    def is_level2_fully_covered(self, sheet_name: str, level2_path: List[str]) -> bool:
        return len(self.get_uncovered_level3_under_level2(sheet_name, level2_path)) == 0

    # ── 核心概念获取 ────────────────────────────────────

    def get_next_core_concepts(self, sheet_name: str, count: int) -> List[ConceptNode]:
        """获取下一批未覆盖的 level3 作为核心概念

        优先级：只取零覆盖，同优先级内按类目树原始顺序
        """
        if sheet_name not in self.state:
            return []

        all_records = [
            rec for rec in self.state[sheet_name].values()
            if rec.times_covered == 0
        ]
        all_records.sort(key=lambda r: r.path)

        # 取前 count 个
        selected_records = all_records[:count]

        # 映射回 ConceptNode
        result = []
        level3_nodes = self.loader.get_level3_categories(sheet_name)
        node_map = {n.name: n for n in level3_nodes}

        for rec in selected_records:
            node = node_map.get(rec.name)
            if node:
                result.append(node)

        return result

    # ── 标记覆盖 ──────────────────────────────────────

    def mark_covered(self, sheet_name: str, level3_name: str, as_core: bool = False):
        if sheet_name not in self.state:
            return
        rec = self.state[sheet_name].get(level3_name)
        if not rec:
            return
        now = datetime.now().isoformat()
        if rec.first_covered_at is None:
            rec.first_covered_at = now
        rec.last_covered_at = now
        rec.times_covered += 1
        if as_core:
            rec.as_core += 1
        else:
            rec.as_companion += 1

    def mark_batch_covered(self, records: List[Dict], save: bool = True):
        """批量标记覆盖。每条 dict: {sheet_name, level3_name, as_core}"""
        for r in records:
            self.mark_covered(
                r['sheet_name'],
                r['level3_name'],
                as_core=r.get('as_core', False)
            )
        if save:
            self.save()

    def reset(self):
        """重置为从类目树初始化的零覆盖状态。"""
        self._init_from_loader()

    def rebuild_from_prompts(self, prompts: List[Dict], save: bool = False):
        """从已成功写入的 prompts 重建覆盖状态。"""
        self.reset()
        for prompt in prompts:
            records = []
            concepts = prompt.get('sampling', {}).get('concepts', {})
            categories = list(concepts.keys())
            core_category = categories[0] if categories else None
            for category_key, concept in concepts.items():
                sheet_name = self._category_key_to_sheet(category_key)
                level3_name = concept.get('level3_category')
                if sheet_name and level3_name:
                    records.append({
                        'sheet_name': sheet_name,
                        'level3_name': level3_name,
                        'as_core': category_key == core_category,
                    })
            if records:
                self.mark_batch_covered(records, save=False)
        if save:
            self.save()

    def _category_key_to_sheet(self, category_key: str) -> Optional[str]:
        mapping = {
            'subject': '主体',
            'motion': '运动',
            'scene': '场景',
            'audio': '音频类型',
        }
        return mapping.get(category_key)

    # ── 报告 ──────────────────────────────────────────

    def get_report(self) -> Dict[str, Any]:
        report = {
            'total_coverage_ratio': round(self.get_total_coverage_ratio(), 4),
            'sheets': {}
        }
        for sheet_name, records in self.state.items():
            total = len(records)
            covered = sum(1 for r in records.values() if r.times_covered > 0)
            uncovered_names = [r.name for r in records.values() if r.times_covered == 0]
            report['sheets'][sheet_name] = {
                'total': total,
                'covered': covered,
                'uncovered': total - covered,
                'coverage_ratio': round(covered / max(1, total), 4),
                'uncovered_samples': uncovered_names[:20],
            }
        return report
