# Prompt Generator V2

基于 Excel 类目树自动生成英文视频 prompt 的采样框架。当前只保留 `core_concept` 主流程：系统优先选择未覆盖的三级概念作为核心概念，LLM 负责选择兼容的 companion 二级类目，系统再展开三级/叶子概念并调用生成模型写出最终 prompt。

## 当前主流程

1. `ConceptLoader` 读取 `data/类目树-for生产数据.xlsx`。
2. `CoverageTracker` 从已有输出重建三级概念覆盖状态。
3. `CoreConceptDrivenSampler` 选择未覆盖核心概念，并调用 selection LLM 选择 companion 二级类目。
4. `Combiner` 按难度配置补充 challenge elements。
5. `PromptGenerator` 调用当前激活模型生成最终英文 prompt。
6. `OutputWriter` 写入独立 JSON 输出，并同步覆盖统计。

## 运行

先安装依赖：

```bash
pip install -r requirements.txt
```

配置环境变量。当前 `config.yaml` 默认使用 `dpsk`：

```bash
export DPSK_V3_API_KEY="your-api-key"
```

运行真实生成：

```bash
python3 main.py
```

运行 dry-run，不调用真实 LLM：

```bash
python3 main.py --dry-run
```

## 使用 GLM 做 Prompt Judge

当前 judge 使用 OpenAI-compatible chat completions，中转站地址配置在 `config.yaml`：

```yaml
judge:
  provider: "glm"
  base_url: "https://cloud.infini-ai.com/maas/coding/v1"
  model: "glm-5.1"
  api_key_env: "GLM_JUDGE_API_KEY"
```

先设置 judge key：

```bash
export GLM_JUDGE_API_KEY="your-transfer-station-key"
```

先 dry-run 检查抽样计划，不调用 GLM：

```bash
python3 scripts/judge_prompts.py --dry-run
```

按配置对 Gemini、GPT-4o、DPSK 各抽样 50 条进行 judge：

```bash
python3 scripts/judge_prompts.py
```

只评测某个文件并抽样 20 条：

```bash
python3 scripts/judge_prompts.py --input output/dpsk_prompts.json --sample-size 20
```

全量评测：

```bash
python3 scripts/judge_prompts.py --all
```

Judge 报告会写入 `reports/judge/`，支持断点续跑。当前 GLM-5.1 judge 默认使用动态 batch：按 `max_context_tokens=200000` 的 `40%` 估算输入 token 预算，即每批约 `80000` input tokens。实际每个 batch 装多少条 prompt 会受 prompt 长度影响，可通过 dry-run 查看：

```bash
python3 scripts/judge_prompts.py --dry-run --all
```

## 重生成 Judge 失败样本

第一轮建议使用 text-only repair：保留原始 mandatory concepts，只重新生成被 GLM judge 判为 `FAIL` 的 prompt 文本。

先 dry-run 查看计划：

```bash
python3 scripts/regenerate_failed_prompts.py \
  --source output/gemini_prompts.json \
  --judge reports/judge/gemini_glm_v1_concept_clarity_consistency.json \
  --repair-mode text \
  --regen-llm gemini \
  --round 1 \
  --dry-run
```

实际重生成：

```bash
python3 scripts/regenerate_failed_prompts.py \
  --source output/gemini_prompts.json \
  --judge reports/judge/gemini_glm_v1_concept_clarity_consistency.json \
  --repair-mode text \
  --regen-llm gemini \
  --round 1
```

脚本会生成三类文件：

```text
output/revisions/gemini_prompts_regen_text_r1.json
output/revisions/gemini_prompts_merged_after_text_r1.json
reports/judge/gemini_merged_after_text_r1_glm_v1_concept_clarity_consistency.json
```

其中 merged 文件包含原本 `PASS/PASS_WITH_MINOR_ISSUES` 的样本和新的 `P-xxxxx-R1` 样本；新的 judge report 会 carry over 已通过样本的旧 judge 结果。再次 judge 时只会评新生成的 R1 prompt：

```bash
python3 scripts/judge_prompts.py \
  --input output/revisions/gemini_prompts_merged_after_text_r1.json
```

对于 text-only repair 后仍然失败的 hard cases，再使用 concept repair：

```bash
python3 scripts/regenerate_failed_prompts.py \
  --source output/revisions/gemini_prompts_merged_after_text_r1.json \
  --judge reports/judge/gemini_merged_after_text_r1_glm_v1_concept_clarity_consistency.json \
  --repair-mode concept \
  --regen-llm gemini \
  --round 2
```

Concept repair 会保留一个 anchor concept，只允许 GLM 从 failed concept pool 中选择更兼容的 companion concepts，不允许发明 taxonomy 外概念。Concept repair 可能改变最终 coverage，因此后续需要从 merged 输出重新统计 coverage。

## 关键配置

- `generation.active_llms`: 最终 prompt 生成使用的模型列表。
- `core_sampling.selection_llm_provider`: companion 二级类目选择使用的模型。
- `core_sampling.coverage_state_path`: 当前模型对应的覆盖状态文件。
- `output.path`: 当前模型对应的输出 JSON，建议每个模型使用独立文件，避免覆盖。
- `difficulty.phase1_distribution`: 阶段一覆盖优先时使用的难度分布。
- `difficulty.distribution`: 阶段二补齐时使用的目标难度比例。
- `judge.input_files`: 需要评测的不同模型输出文件。
- `judge.sample.size_per_file`: 每个文件抽样评测数量，调通后可用 `--all` 全量评测。

## 目录说明

- `main.py`: CLI 入口，只负责加载配置和启动 core-concept pipeline。
- `src/pipeline.py`: 当前主流程编排。
- `src/core_concept_sampler.py`: 覆盖驱动采样。
- `src/selection_prompt.py`: companion 类目选择 prompt 构建和解析。
- `src/generator.py`: LLM provider 和最终 prompt 生成规则。
- `src/coverage_tracker.py`: 三级概念覆盖统计。
- `src/combiner.py`: 难度和 challenge 组合。
- `scripts/plot_prompt_stats.py`: 统计图绘制脚本。

## 注意

不要把 API key 写入 `config.yaml` 或代码文件。所有密钥应通过环境变量读取。生成的输出、覆盖状态、统计图和临时 rollout 文件默认由 `.gitignore` 忽略。
