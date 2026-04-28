"""
Challenge element sampling.
"""
import random
from typing import Dict, List


class ChallengeSampler:
    """Uniform-ish sampler for challenge elements."""

    def __init__(self, challenge_elements: List[Dict]):
        self.challenge_elements = challenge_elements
        self._build_pool()

    def _build_pool(self) -> None:
        self.pool = [elem for elem in self.challenge_elements]
        self.sampled_count: Dict[str, int] = {
            elem['id']: 0 for elem in self.challenge_elements
        }

    def sample(self, count: int) -> List[Dict]:
        if count <= 0:
            return []

        sorted_elements = sorted(
            self.challenge_elements,
            key=lambda x: self.sampled_count[x['id']],
        )
        candidates = sorted_elements[:max(count * 2, len(sorted_elements))]
        selected = random.sample(candidates, min(count, len(candidates)))

        for elem in selected:
            self.sampled_count[elem['id']] += 1

        return selected

    def get_distribution(self) -> Dict[str, int]:
        return self.sampled_count.copy()
