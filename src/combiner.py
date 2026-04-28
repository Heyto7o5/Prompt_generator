"""
类目组合与难度分配模块
"""
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .challenge_sampler import ChallengeSampler
from .models import SampledConcept


@dataclass
class DifficultyParams:
    """难度参数"""
    level: str
    element_categories_min: int
    element_categories_max: int
    modifier_count_min: int
    modifier_count_max: int
    text_length_min: Optional[int] = None
    text_length_max: Optional[int] = None
    challenge_count_min: int = 0
    challenge_count_max: int = 0


class DifficultyManager:
    """难度管理器"""
    
    LEVEL_MAPPING = {
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High'
    }
    
    def __init__(self, distribution: Dict[str, int], params: Dict[str, Any]):
        self.distribution = distribution
        self.params = {
            level: self._parse_params(level, params.get(level, {}))
            for level in ['low', 'medium', 'high']
        }
    
    def _parse_params(self, level: str, raw_params: Dict) -> DifficultyParams:
        """解析难度参数"""
        fixed_challenge_count = raw_params.get('challenge_count')
        return DifficultyParams(
            level=level,
            element_categories_min=raw_params.get('element_categories_min', 1),
            element_categories_max=raw_params.get('element_categories_max', 10),
            modifier_count_min=raw_params.get('modifier_count_min', 0),
            modifier_count_max=raw_params.get('modifier_count_max', 10),
            text_length_min=raw_params.get('text_length_min'),
            text_length_max=raw_params.get('text_length_max'),
            challenge_count_min=raw_params.get('challenge_count_min', fixed_challenge_count or 0),
            challenge_count_max=raw_params.get('challenge_count_max', fixed_challenge_count or 0)
        )
    
    def get_params(self, level: str) -> DifficultyParams:
        """获取指定难度的参数"""
        return self.params.get(level, self.params['medium'])


@dataclass
class PromptCombination:
    """Prompt组合结构"""
    combination_id: str
    concepts: Dict[str, SampledConcept]
    difficulty_level: str
    difficulty_params: DifficultyParams
    challenge_elements: List[Dict]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'combination_id': self.combination_id,
            'categories_selected': list(self.concepts.keys()),
            'concepts': {
                key: concept.to_dict() for key, concept in self.concepts.items()
            },
            'difficulty': {
                'level': DifficultyManager.LEVEL_MAPPING[self.difficulty_level],
                'params': {
                    'element_categories_range': [
                        self.difficulty_params.element_categories_min,
                        self.difficulty_params.element_categories_max
                    ],
                    'modifier_range': [
                        self.difficulty_params.modifier_count_min,
                        self.difficulty_params.modifier_count_max
                    ],
                    'text_length_min': self.difficulty_params.text_length_min,
                    'text_length_max': self.difficulty_params.text_length_max,
                    'challenge_count': len(self.challenge_elements)
                }
            },
            'challenge_elements': [
                {'id': elem['id'], 'name': elem['name']}
                for elem in self.challenge_elements
            ]
        }


class Combiner:
    """组合器"""
    
    def __init__(self, difficulty_manager: DifficultyManager, challenge_sampler: ChallengeSampler):
        self.difficulty_manager = difficulty_manager
        self.challenge_sampler = challenge_sampler
        self.combination_counter = 0

    def create_combination_from_core_selection(
        self, sampled_concepts: Dict[str, SampledConcept], difficulty_level: Optional[str] = None
    ) -> PromptCombination:
        """Create a prompt combination from core-concept sampler output."""
        difficulty_level = difficulty_level or 'medium'
        difficulty_params = self.difficulty_manager.get_params(difficulty_level)

        challenge_count = random.randint(
            difficulty_params.challenge_count_min,
            max(difficulty_params.challenge_count_min, difficulty_params.challenge_count_max)
        )
        challenge_elements = self.challenge_sampler.sample(challenge_count)

        self.combination_counter += 1
        combination_id = f"C-{self.combination_counter:05d}"

        return PromptCombination(
            combination_id=combination_id,
            concepts=sampled_concepts,
            difficulty_level=difficulty_level,
            difficulty_params=difficulty_params,
            challenge_elements=challenge_elements
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_combinations': self.combination_counter,
            'configured_difficulty_distribution': self.difficulty_manager.distribution,
            'challenge_distribution': self.challenge_sampler.get_distribution()
        }
