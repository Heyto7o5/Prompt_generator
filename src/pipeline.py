"""
Core-concept driven prompt generation pipeline.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .challenge_sampler import ChallengeSampler
from .combiner import Combiner, DifficultyManager
from .config import Config
from .concept_loader import load_concepts
from .core_concept_sampler import CoreConceptDrivenSampler
from .coverage_tracker import CoverageTracker
from .generator import PromptGenerator
from .models import SampledConcept
from .output import OutputWriter, StatsCollector


class CoreConceptPipeline:
    """Core-concept driven sampling and generation pipeline."""

    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

        print("  [Init] Loading concept taxonomy...")
        self.loader = load_concepts(self.config.concept_tree_path)

        print("  [Init] Initializing coverage tracker...")
        self.coverage = CoverageTracker(self.loader, config.coverage_state_path)
        report = self.coverage.get_report()
        for sheet, info in report['sheets'].items():
            print(
                f"    {sheet}: {info['covered']}/{info['total']} covered "
                f"({info['coverage_ratio']:.1%})"
            )

        print("  [Init] Initializing LLM provider for concept selection...")
        self.llm_for_selection = self._get_mock_llm() if dry_run else self._get_llm_provider()

        print("  [Init] Initializing core concept sampler...")
        self.core_sampler = CoreConceptDrivenSampler(
            loader=self.loader,
            llm_provider=self.llm_for_selection,
            coverage_tracker=self.coverage,
            dimensions_config=config.dimensions,
            batch_size=config.core_sampling_batch_size,
            num_dimensions=config.num_dimensions_per_combo,
            max_retries=config.core_sampling_max_retries,
            level3_mode=config.level3_mode,
        )

        self.challenge_sampler = ChallengeSampler(self.config.challenge_elements)
        self.difficulty_manager = DifficultyManager(
            self.config.difficulty_distribution,
            self.config.difficulty_params,
        )
        self.combiner = Combiner(self.difficulty_manager, self.challenge_sampler)

        if not self.dry_run:
            print("  [Init] Initializing LLM generator for prompt text...")
            self.generator = PromptGenerator(self.config.llm_providers, self.config.active_llms)

    def _get_llm_provider(self):
        from .generator import DpskProvider, GeminiProvider, GPTProvider, QwenProvider

        provider_name = self.config.selection_llm_provider
        provider_classes = {
            'dpsk': DpskProvider,
            'gemini': GeminiProvider,
            'gpt': GPTProvider,
            'qwen': QwenProvider,
        }
        provider_class = provider_classes.get(provider_name)
        if not provider_class:
            raise ValueError(f"不支持的概念选择 LLM: {provider_name}")

        provider_config = self.config.llm_providers.get(provider_name, {})
        if not provider_config:
            raise ValueError(f"缺少 LLM 配置: llm.providers.{provider_name}")

        provider = provider_class(provider_config)
        if not provider.is_available():
            api_key_env = provider_config.get('api_key_env', '')
            raise ValueError(f"LLM {provider_name} 不可用，请检查配置和环境变量 {api_key_env}")

        print(f"    使用 {provider_name} 进行概念选择")
        return provider

    def _get_mock_llm(self):
        """Mock LLM for dry-run mode."""
        import re

        class MockLLM:
            def generate(self, prompt):
                core_names = re.findall(r'\d+\.\s+(.+?)\s+\(', prompt)
                if not core_names:
                    return json.dumps([])

                templates = [
                    {
                        'motion_level2': '体育运动-跑步行走类运动',
                        'scene_level2': '自然环境-陆地环境',
                        'audio_level2': '自然声音',
                    },
                    {
                        'motion_level2': '体育运动-球类运动',
                        'scene_level2': '人造建筑-体育场馆',
                        'audio_level2': '环境音',
                    },
                    {
                        'motion_level2': '舞蹈-现代舞',
                        'scene_level2': '人造建筑-室内空间',
                        'audio_level2': '音乐',
                    },
                    {
                        'motion_level2': '体育运动-水上类运动',
                        'scene_level2': '自然环境-水体环境',
                        'audio_level2': '自然声音',
                    },
                    {
                        'motion_level2': '家务活动-打扫清洁',
                        'scene_level2': '人造建筑-住宅',
                        'audio_level2': '环境音',
                    },
                ]

                results = []
                for i, name in enumerate(core_names):
                    template = templates[i % len(templates)]
                    results.append({
                        'core_level3': name,
                        **template,
                        'confidence': 0.8,
                        'note': 'mock',
                    })
                return json.dumps(results)

        return MockLLM()


def run_core_concept_pipeline(config: Config, dry_run: bool):
    """Run the core-concept driven pipeline."""
    print("\n[2/7] 初始化核心概念驱动流水线...")
    pipeline = CoreConceptPipeline(config, dry_run)

    output_path = Path(config.output_path)
    if dry_run:
        output_path = output_path.with_name(f"dry_run_{output_path.name}")

    existing_prompts = []
    prompt_counter = 0
    if output_path.exists() and not dry_run:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            existing_prompts = old_data.get('prompts', [])
            prompt_counter = len(existing_prompts)
            print(f"  [Resume] 加载已有 {prompt_counter} 条prompt，从中断处继续")
        except Exception:
            print("  [Resume] 无法解析已有输出，从头开始")

    if existing_prompts:
        print("  [Resume] 从已有输出重建覆盖状态")
        pipeline.coverage.rebuild_from_prompts(existing_prompts, save=not dry_run)
        pipeline.combiner.combination_counter = _get_max_numeric_id(
            existing_prompts,
            'combination_id',
            'C-',
        )

    print("\n[3/7] 初始化输出...")
    output_writer = OutputWriter(str(output_path))
    for prompt in existing_prompts:
        output_writer.add_prompt(prompt)

    stats_collector = StatsCollector()
    for prompt in existing_prompts:
        stats_collector.record(prompt)

    print("\n[4/7] 阶段一：覆盖优先...")
    max_prompts = config.max_prompts
    no_new_combo_count = 0
    last_flush_count = prompt_counter
    phase1_difficulty_pool = _build_difficulty_pool(config.phase1_difficulty_distribution)

    while True:
        if max_prompts > 0 and prompt_counter >= max_prompts:
            print(f"\n  已达到生成上限 ({max_prompts})，停止")
            break

        difficulty_level = _sample_from_pool(phase1_difficulty_pool)
        difficulty_params = pipeline.difficulty_manager.get_params(difficulty_level)
        target_num_dimensions = random.randint(
            difficulty_params.element_categories_min,
            difficulty_params.element_categories_max,
        )
        sampled = pipeline.core_sampler.sample_combination(num_dimensions=target_num_dimensions)
        if sampled is None:
            no_new_combo_count += 1
            if no_new_combo_count >= 5:
                coverage = pipeline.coverage.get_report()
                if coverage['total_coverage_ratio'] >= 0.99:
                    print("\n  三级全覆盖完成！")
                else:
                    print("\n  无更多可生成组合，阶段一结束")
                break
            continue
        no_new_combo_count = 0

        combination = pipeline.combiner.create_combination_from_core_selection(
            sampled,
            difficulty_level=difficulty_level,
        )
        prompt_counter = _generate_or_mock(
            pipeline,
            output_writer,
            stats_collector,
            combination,
            sampled,
            prompt_counter,
        )

        if prompt_counter - last_flush_count >= 10:
            _flush_output(
                output_writer,
                stats_collector,
                pipeline,
                prompt_counter,
                'phase1_in_progress',
            )
            last_flush_count = prompt_counter

        if prompt_counter % 20 == 0:
            coverage = pipeline.coverage.get_report()
            diff_dist = stats_collector.difficulty_count
            print(
                f"  [{prompt_counter}条] 覆盖率: {coverage['total_coverage_ratio']:.1%} "
                f"| 难度: L{diff_dist['low']} M{diff_dist['medium']} H{diff_dist['high']}",
                flush=True,
            )

    print("\n[5/7] 阶段二：难度补齐...")
    diff_dist = stats_collector.difficulty_count
    total_existing = diff_dist['low'] + diff_dist['medium'] + diff_dist['high']
    print(
        f"  当前难度分布: LOW={diff_dist['low']}, MEDIUM={diff_dist['medium']}, "
        f"HIGH={diff_dist['high']} (共{total_existing}条)"
    )

    target_low = int(total_existing * 3 / 10)
    target_medium = int(total_existing * 5 / 10)
    target_high = int(total_existing * 2 / 10)

    needed_medium = max(0, target_medium - diff_dist['medium'])
    if needed_medium > 0:
        scale = (diff_dist['medium'] + needed_medium) / 5 * 10
        target_low = int(scale * 3 / 10)
        target_medium = int(scale * 5 / 10)
        target_high = int(scale * 2 / 10)

    deficit = {
        'low': max(0, target_low - diff_dist['low']),
        'medium': max(0, target_medium - diff_dist['medium']),
        'high': max(0, target_high - diff_dist['high']),
    }
    total_deficit = sum(deficit.values())
    print(
        f"  目标比例 3:5:2 → 需要: LOW+{deficit['low']}, "
        f"MEDIUM+{deficit['medium']}, HIGH+{deficit['high']} (共{total_deficit}条)"
    )

    if total_deficit > 0:
        all_covered = _get_covered_pool(pipeline)
        phase2_count = 0

        for level, count in deficit.items():
            for _ in range(count):
                if max_prompts > 0 and prompt_counter >= max_prompts:
                    break

                if level == 'low':
                    num_dims = random.choice([1, 2])
                elif level == 'medium':
                    num_dims = 3
                else:
                    num_dims = 4

                sampled = _random_combination_from_pool(all_covered, num_dims, pipeline)
                if not sampled:
                    continue

                combination = pipeline.combiner.create_combination_from_core_selection(
                    sampled,
                    difficulty_level=level,
                )
                prompt_counter = _generate_or_mock(
                    pipeline,
                    output_writer,
                    stats_collector,
                    combination,
                    sampled,
                    prompt_counter,
                    dry_run_prefix='[Dry-run Phase2]',
                )

                phase2_count += 1
                if phase2_count % 20 == 0:
                    _flush_output(
                        output_writer,
                        stats_collector,
                        pipeline,
                        prompt_counter,
                        'phase2_in_progress',
                    )

        print(f"  阶段二补充了 {phase2_count} 条")

    print("\n[6/7] 写入最终输出...")
    coverage_report = _rebuild_persisted_coverage(pipeline, output_writer)
    stats_collector.record_coverage(coverage_report)
    final_stats = {
        **stats_collector.get_summary(),
        'sampler_stats': pipeline.core_sampler.get_stats(),
    }
    output_writer.set_stats(final_stats)
    output_writer.write()
    if not dry_run:
        pipeline.coverage.save()

    print("\n" + "=" * 60)
    print("生成完成!")
    print("=" * 60)
    coverage_report = pipeline.coverage.get_report()
    for sheet, info in coverage_report['sheets'].items():
        print(f"  {sheet}: {info['covered']}/{info['total']} ({info['coverage_ratio']:.1%})")
        if 0 < info['uncovered'] <= 20:
            print(f"    未覆盖: {info['uncovered_samples']}")

    print(f"\n总覆盖率: {coverage_report['total_coverage_ratio']:.1%}")
    diff_dist = stats_collector.difficulty_count
    total = diff_dist['low'] + diff_dist['medium'] + diff_dist['high']
    if total > 0:
        print(
            f"难度分布: LOW={diff_dist['low']} ({diff_dist['low'] / total:.0%}), "
            f"MEDIUM={diff_dist['medium']} ({diff_dist['medium'] / total:.0%}), "
            f"HIGH={diff_dist['high']} ({diff_dist['high'] / total:.0%})"
        )
    print(f"总生成: {prompt_counter} 条")
    print(f"LLM调用: {pipeline.core_sampler.get_stats()['llm_calls']}")


def _generate_or_mock(
    pipeline: CoreConceptPipeline,
    output_writer: OutputWriter,
    stats_collector: StatsCollector,
    combination,
    sampled: Dict[str, SampledConcept],
    prompt_counter: int,
    dry_run_prefix: str = '[Dry-run]',
) -> int:
    if pipeline.dry_run:
        mock_text = f"{dry_run_prefix} Prompt based on: {list(sampled.keys())}"
        prompt_dict = {
            'prompt_id': f"P-{prompt_counter + 1:05d}",
            'combination_id': combination.combination_id,
            'difficulty': {'level': combination.difficulty_level.upper()},
            'sampling': {
                'combination_id': combination.combination_id,
                'categories_selected': list(sampled.keys()),
                'concepts': {k: v.to_dict() for k, v in sampled.items()},
                'challenge_elements': combination.challenge_elements,
            },
            'text': mock_text,
            'text_length': len(mock_text.split()),
            'llm': {'provider': 'mock', 'model': 'dry-run'},
            'metadata': {'created_at': '2026-04-23T00:00:00Z'},
        }
        output_writer.add_prompt(prompt_dict)
        stats_collector.record(prompt_dict)
        return prompt_counter + 1

    prompts = pipeline.generator.generate(combination)
    for prompt in prompts:
        prompt_dict = prompt.to_dict()
        prompt_dict['prompt_id'] = f"P-{prompt_counter + 1:05d}"
        output_writer.add_prompt(prompt_dict)
        stats_collector.record(prompt_dict)
        prompt_counter += 1
    return prompt_counter


def _flush_output(output_writer, stats_collector, pipeline, prompt_counter, phase):
    coverage_report = _rebuild_persisted_coverage(pipeline, output_writer)
    stats_collector.record_coverage(coverage_report)
    output_writer.set_stats({
        **stats_collector.get_summary(),
        'sampler_stats': pipeline.core_sampler.get_stats(),
        'phase': phase,
    })
    output_writer.write()
    if not pipeline.dry_run:
        pipeline.coverage.save()
    print(f"  [Checkpoint] 已写入 {prompt_counter} 条", flush=True)


def _rebuild_persisted_coverage(pipeline, output_writer):
    pipeline.coverage.rebuild_from_prompts(output_writer.prompts, save=False)
    return pipeline.coverage.get_report()


def _build_difficulty_pool(distribution: Dict[str, int]) -> List[str]:
    pool = []
    for level, weight in distribution.items():
        if weight > 0:
            pool.extend([level] * weight)
    return pool or ['medium']


def _sample_from_pool(pool: List[str]) -> str:
    return random.choice(pool)


def _get_max_numeric_id(items: List[Dict], field: str, prefix: str) -> int:
    max_id = 0
    for item in items:
        raw_id = item.get(field, '')
        if isinstance(raw_id, str) and raw_id.startswith(prefix):
            try:
                max_id = max(max_id, int(raw_id[len(prefix):]))
            except ValueError:
                continue
    return max_id


def _get_covered_pool(pipeline) -> Dict[str, List]:
    pool = {}
    for sheet_name, records in pipeline.coverage.state.items():
        dim_key = pipeline.core_sampler.sheet_to_key.get(sheet_name, sheet_name)
        covered_nodes = []
        for name, rec in records.items():
            if rec.times_covered > 0:
                node = pipeline.core_sampler._find_level3_node(sheet_name, name)
                if node:
                    covered_nodes.append(node)
        if covered_nodes:
            pool[dim_key] = covered_nodes
    return pool


def _random_combination_from_pool(pool: Dict[str, List], num_dims: int, pipeline) -> Optional[Dict]:
    dims = list(pool.keys())
    if len(dims) < num_dims:
        num_dims = len(dims)
    if num_dims == 0:
        return None

    selected_dims = random.sample(dims, num_dims)
    result = {}
    for dim_key in selected_dims:
        nodes = pool[dim_key]
        node = random.choice(nodes)
        sheet = pipeline.core_sampler.key_to_sheet[dim_key]
        leaf = None
        leaves = pipeline.loader.get_leaves_under_level3(sheet, node.name)
        valid = [
            leaf_node
            for leaf_node in leaves
            if not any(kw in leaf_node.name for kw in ['其他', '其它', 'Other', 'other'])
        ]
        if valid:
            leaf = random.choice(valid).name

        result[dim_key] = SampledConcept(
            sheet_name=sheet,
            level3_category=node.name,
            level3_path=list(node.path),
            leaf=leaf,
        )
    return result
