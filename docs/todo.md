# TODO

## Prompt Format Templates

- 示例已归纳为显式结构化模板族：`shot_timeline`、`split_screen_explanation`、`field_spec`、`script_timeline`。`cinematic_paragraph` 只作为历史兼容标签，不再作为结构化 rewrite 目标。
- 当前采取“保留原 prompt + 随机 30% rewrite”为结构化模板的方式，不直接重写全部样本，以便后续和原始自然语言 prompt 做对照。
- rewrite 后的记录需要保留 `template_family` 和 `revision` 元数据，便于后续 judge、统计和回溯。
- 模板化后，judge 规则要同步区分：结构合规性、核心概念聚焦、概念保真、内部一致性和视频生成可用性；结构化 prompt 不应因换行、镜头号、字段标签本身被误判为格式错误。
- rewrite 后必须通过模板格式校验：镜头时间轴需要镜头/分镜编号，分屏解释需要成对屏幕标签，字段式设定需要多个字段标签，时间脚本需要多个时间段；校验失败则重试，最终失败不标记为结构化。

## Sampling And Judge

- 当前采样流程：难度决定目标维度数；系统先随机采样 companion 维度；LLM 只在指定维度内选择整体兼容的二级类目和 challenge，并给出 `combination_reason`；系统再在选中二级下覆盖优先采样真实三级/叶子。
- 阶段二应持续使用 anchor + LLM compatibility selection，不再随机拼接已覆盖概念。
- judge 已按五维标准组织：概念保真与聚焦、内部一致性、清晰度与具象化、语言质量、视频生成可用性。
- 后续可将 judge 中 `combination_issue.should_add_to_contradiction_pool=true` 的结果沉淀为独立矛盾模板池文件，用于采样前过滤或人工 review。
- TODO: 长度控制后续单独实现，先明确中文长度单位（汉字、词、短语或模型 tokenizer），再增加生成后校验和压缩重试。
- TODO: 持续观察 `combination_reason` 是否泄露未展示三级/叶子或外部事实；若仍出现，需要加强 selection prompt 或改为结构化输出约束。
- TODO: 统计 LLM 输出二级非法、缺失指定维度、JSON 解析失败等重试原因，判断是否需要更强解析修复或 provider-specific prompt。
- TODO: 后续如果用户提供固定 prompt 示例，需要把自然语言生成器升级为可选模板生成器，并同步更新 judge 的结构合规性检查。
