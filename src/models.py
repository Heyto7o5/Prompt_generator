"""
Shared data models for prompt sampling and generation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SampledConcept:
    """A sampled concept from the taxonomy."""

    sheet_name: str
    level3_category: str
    level3_path: List[str]
    leaf: Optional[str] = None

    def to_dict(self) -> Dict:
        level1 = self.level3_path[0] if len(self.level3_path) > 0 else ''
        level2 = self.level3_path[1] if len(self.level3_path) > 1 else ''
        level3 = self.level3_path[2] if len(self.level3_path) > 2 else self.level3_category

        return {
            'level1_category': level1,
            'level2_category': level2,
            'level3_category': level3,
            'full_path': ' > '.join(self.level3_path) if self.level3_path else self.level3_category,
            'leaf': self.leaf,
        }
