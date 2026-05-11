"""
Shared data models for prompt sampling and generation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
            'path': list(self.level3_path),
            'leaf': self.leaf,
        }


@dataclass
class SampledCombination:
    """A sampled prompt combination with trace metadata."""

    concepts: Dict[str, SampledConcept]
    challenge_elements: List[Dict[str, Any]] = field(default_factory=list)
    selection_trace: Dict[str, Any] = field(default_factory=dict)
    phase: str = "phase1"

    def categories_selected(self) -> List[str]:
        return list(self.concepts.keys())

    def concepts_to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {key: concept.to_dict() for key, concept in self.concepts.items()}
