"""
配置加载模块
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @property
    def concept_tree_path(self) -> str:
        return self.get('concept_tree.path', 'data/类目树-for生产数据.xlsx')
    
    @property
    def active_llms(self) -> list:
        return self.get('generation.active_llms', ['gemini'])
    
    @property
    def difficulty_distribution(self) -> Dict[str, int]:
        return self.get('difficulty.distribution', {'low': 30, 'medium': 50, 'high': 20})
    
    @property
    def difficulty_params(self) -> Dict[str, Any]:
        return self.get('difficulty.params', {})

    @property
    def phase1_difficulty_distribution(self) -> Dict[str, int]:
        return self.get('difficulty.phase1_distribution', {'medium': 70, 'high': 30})
    
    @property
    def llm_providers(self) -> Dict[str, Any]:
        return self.get('llm.providers', {})
    
    @property
    def challenge_elements(self) -> list:
        return self.get('challenge_elements', [])
    
    @property
    def output_path(self) -> str:
        return self.get('output.path', 'output/prompts.json')
    
    @property
    def max_prompts(self) -> int:
        """最大生成数量限制，0表示不限制"""
        return self.get('generation.max_prompts', 0)

    # ── 核心概念驱动采样配置 ──

    @property
    def dimensions(self) -> list:
        return self.get('dimensions', [
            {'key': 'subject', 'sheet': '主体', 'core_priority': 1, 'companion': True},
            {'key': 'motion', 'sheet': '运动', 'core_priority': 2, 'companion': True},
            {'key': 'scene', 'sheet': '场景', 'core_priority': 3, 'companion': True},
            {'key': 'audio', 'sheet': '音频类型', 'core_priority': 4, 'companion': True},
        ])

    @property
    def concept_selection_method(self) -> str:
        """Concept selection strategy.

        - semantic_topk: LLM selects compatible level-2 and real level-3 concepts.
        - level2: legacy flow, LLM selects level-2 and system expands level-3.
        """
        method = self.get('core_sampling.selection_method')
        if not method:
            # Backward compatibility for old config files.
            old_mode = self.get('core_sampling.sampling_mode', 'semantic_topk')
            method = 'level2' if old_mode == 'core_concept' else old_mode
        method = str(method).strip().lower()
        aliases = {
            'semantic': 'semantic_topk',
            'semantic_topk_level2_level3': 'semantic_topk',
            'core_concept': 'level2',
            'llm_level2': 'level2',
        }
        return aliases.get(method, method)

    @property
    def sampling_mode(self) -> str:
        """Deprecated compatibility alias."""
        return self.concept_selection_method

    @property
    def core_sampling_batch_size(self) -> int:
        return self.get('core_sampling.batch_size', 10)

    @property
    def selection_llm_provider(self) -> str:
        return self.get('core_sampling.selection_llm_provider', 'gemini')

    @property
    def coverage_state_path(self) -> str:
        return self.get('core_sampling.coverage_state_path', 'data/coverage_state.json')

    @property
    def core_sampling_max_retries(self) -> int:
        return self.get('core_sampling.max_retries', 2)

    @property
    def level3_mode(self) -> str:
        return self.get('core_sampling.level3_mode', 'traverse')

    @property
    def semantic_level3_per_level2(self) -> int:
        return max(1, int(self.get('core_sampling.semantic_topk.level3_per_level2', 10)))


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置的便捷函数"""
    return Config(config_path)
