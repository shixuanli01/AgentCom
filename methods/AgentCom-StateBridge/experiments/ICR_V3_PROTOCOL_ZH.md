# ICR V3 协议草案：全集 baseline 的 prompt 与判分规格

日期：2026-09-19

状态：**草案，待批准。尚未实现代码，尚未运行任何生成。**

目的：把 `none / true_text / true_statebridge / true_latentmas` 四个 baseline
在五个数据集的**完整测试集**上跑出可冻结的数值。V3 只修正 V2 中已确认的
prompt 不对称与判分缺陷，不引入新方法、不改变 ICR 的科学问题。

---

## 1. 相对 V2 的改动清单

| # | 改动 | 依据 |
|---|---|---|
| 1 | 四个条件共用同一模板，只有 external 槽位随条件变化 | V2 中 latent 条件的 marker 在 user turn 最前，Text 的消息在中部 |
| 2 | 删除 latent 条件独有的 `Use the external message above as evidence if it is relevant.` | V2 只给 latent 这条额外指令，Text 没有 |
| 3 | 引导句统一为 `An external message from another reasoning process is available.` | V2 的 `You may now have access to…` 措辞含糊 |
| 4 | 整合指令改为条件中性措辞 | V2 在 `none` 下仍引用 "the external information"、"another message exists" |
| 5 | 答案格式抽成 per-dataset 的 `{answer_format}` 段，且置于末尾 | V2 的答案格式写死为 A-D，且 Phase-1 中它位于题目之前 |
| 6 | 判分层修复 S1-S4（见 §5） | GSM8K 千分位逗号导致评分反转；MCQ boxed 解析过严 |
| 7 | ARC-Challenge 预注册剔除 7 道非四选项题 | 全集含 3 道五选项、4 道三选项，与 A-D 输出契约不符 |

V2 的 Text 条件布局**未改变**；改动集中在 latent 侧与判分侧。

`revision_prompt_version = "icr_v3_mid_injection"`，protocol 名 `ICR-V3`。
V3 与 V2 的 artifact 根目录分开，不得混入同一 pooled 估计。

---

## 2. 段落结构

Phase 2 修订 prompt 固定为五段，顺序不变：

```text
[Task / Question]              原题全文
[Receiver's own prior]         接收者自己第一次的推理与答案
[External information]         唯一随条件变化的槽位
[How to integrate]             整合规则（条件中性）
[Output format]                per-dataset 答案格式
```

Phase 1 独立作答同样以 `[Output format]` 结尾，保证两阶段的输出契约一致。

---

## 3. 完整模板

### 3.1 Phase 1 · 独立作答

```text
You are an independent problem-solving agent.

Solve the following {task_noun} carefully and independently.
Reason from the evidence in the problem.
Do not assume another agent will review your answer.

Problem:
{question}

{answer_format}
```

`{task_noun}`：medqa = `medical multiple-choice question`；gpqa / arc_challenge =
`multiple-choice question`；gsm8k = `math word problem`；mbppplus / humanevalplus =
`programming problem`。

### 3.2 Phase 2 · 修订（选择题与数学题）

```text
You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_block}

Your task is to REVISE your belief, not to restart from scratch.

Evaluate your previous reasoning and any external message critically.

* Change your answer only if you find a concrete error in your previous reasoning, or evidence that is better supported than it.
* Do not change your answer merely because an external message is present.
* Resolve any disagreement using the evidence in the original problem.

{answer_format}
```

`{receiver_prior_answer}` 无法解析时填 `UNPARSEABLE`。

### 3.3 Phase 2 · 修订（代码任务）

```text
You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning and implementation:
{receiver_prior_reasoning}

{external_block}

Your task is to REVISE your implementation, not to restart from scratch and not to copy
an external message.

Evaluate your previous implementation and any external message critically.

* Change your implementation only if you find a concrete defect in it, or an approach that is better supported than it.
* Do not change your implementation merely because an external message is present.
* Check the required function signature, imports, examples, and edge cases against the original problem.

{answer_format}
```

### 3.4 `{external_block}` 的四种填法

```text
none:
No external message is available.

true_text:
An external message from another reasoning process is available.

External message:
{sender_reasoning}

true_statebridge / true_latentmas:
An external message from another reasoning process is available.

External message:
[EMBEDDING_CONTEXT_HERE]
```

`[EMBEDDING_CONTEXT_HERE]` 在 tokenize 前删除，其 token 位置定义连续前缀的插入点。

对齐性已逐字校验：

- `true_statebridge` 与 `true_latentmas` 的可见文本完全相同；
- `true_text` 与它们**仅**相差消息正文那一行；
- `none` 与它们**仅**相差 external 块本身。

---

## 4. Per-dataset 答案格式与题面标签

| 数据集 | 题面中的选项标签 | `{answer_format}` |
|---|---|---|
| medqa | `A. B. C. D.` | `Return your concise reasoning, then exactly one final answer as \boxed{X}, where X is one of A, B, C, or D.` |
| arc_challenge | `a: b: c: d:`（小写） | 同上，但结尾为 `where X is one of the option labels shown above: a, b, c, or d.` |
| gpqa | 内层 `a)`，外层 `A.` | `… where X is one of A, B, C, or D from the final labeled list above. Do not answer with the lowercase a)-d) items quoted inside those options.` |
| gsm8k | — | `… as \boxed{N}, where N is a single number written in plain digits, with no thousands separators, no units, and no other symbols.` |
| mbppplus / humanevalplus | — | `Return the complete final implementation in exactly one markdown Python code block. Do not put tests or explanatory prose inside that code block.` |

题面本身不修改，沿用 `data.py` 的上游渲染；只让答案指令与实际标签一致。

GPQA 的双层选项是上游 StateBridge 转换格式的既有特征：内层以 `a)`-`d)` 列出
选项正文，外层以 `A.`-`D.` 给出真正的答案标签并指向内层。gold 取自外层。
V2 的指令未区分两层，存在静默误判风险。

---

## 5. 判分层修复

以下修复**不改变任何生成**，只改变对已生成文本的判定，因此可以回灌到既有
记录上重新计分并精确报告影响面。

| 编号 | 问题 | 修复 | 已知影响面 |
|---|---|---|---|
| S1 | `extract_gold` 的 `####\s*([-+]?\d+(?:\.\d+)?)` 在 `#### 2,125` 上只截到 `2`，gold 本身被截断 | 允许千分位并去除逗号后再规范化 | GSM8K test 约 15/1319（实测前 800 题中 9 例） |
| S2 | `extract_gsm8k_answer` 对 `\boxed{2,125}` 返回 `2` | boxed 内容先剥 `\text{}`、`$`、`\!`、逗号、空格，再取数 | 同上 |
| S3 | boxed 内容形如 `C. Colorectal cancer` 时判为 `None` | 若非裸字母，取行首的独立 A-D 标签 | 全部 MCQ 的 invalid 率 |
| S4 | `\boxed{\text{C}}` 因 `[^}]*` 截断而失败 | boxed 提取改为花括号配对，并剥离 `\text{}` 等包装 | 全部 MCQ |
| S5 | `\boxed{18.0}` 与 gold `18` 字符串不等而判错 | 两侧均可解析为数时按 `Decimal` 数值比较，否则退回精确比较 | GSM8K，影响面未预先量化 |

S5 是实现阶段新增的一项（原草案只列 S1-S4）。它与 S1/S2 同类：纯格式差异不应决定
对错。V3 的 GSM8K 输出契约已要求"plain digits"，S5 是判分侧的第二道保险。

S1 的后果在 V2 数据中是**评分反转**：模型写 `\boxed{2125}`（正确）被判错，
写 `\boxed{2,125}` 反而判对。由于不同通道可能诱导不同数字格式，它是系统性
混淆而非随机噪声。

实现约束：新解析逻辑写在 `icr/` 内，**不修改 `utils.py`**，以保持冻结的
StateBridge 控制组与既有分析逐字可复现。

---

## 6. 数据集范围

| 数据集 | 全集 | V3 纳入 | 说明 |
|---|---:|---:|---|
| medqa | 300 | 300 | 上游论文的固定 300 题子集，无"全集"可用 |
| arc_challenge | 1,172 | **1,165** | 预注册剔除 7 道非四选项题 |
| gsm8k | 1,319 | 1,319 | — |
| gpqa | 198 | 198 | — |
| humanevalplus | 164 | 164 | — |
| mbppplus | 378 | **不纳入** | 本轮不跑，见 §8 |

ARC 剔除规则：**选项个数不等于 4** 的题目。该准则只依赖题面结构，不依赖
gold、模型输出或正确性，属 label-free 预注册排除。被剔除的 7 个 item id 需
原样记录在 `config.json` 的 `excluded_item_ids` 中，且 item_id 仍沿用原始
1,172 题索引，以保持可追溯性。

被剔除题目：4 道三选项（`NYSEDREGENTS_2010_4_4`、`NYSEDREGENTS_2012_8_42`、
`NYSEDREGENTS_2008_4_15`、`NYSEDREGENTS_2015_4_7`）与 3 道五选项
（`TIMSS_1995_8_J7`、`TIMSS_2007_8_pg53`、`TIMSS_1995_8_N4`）。

---

## 7. LatentMAS 的位置约束（必须在论文中声明）

StateBridge 与 Text 的载荷位置是自由参数，V3 统一放在第三段。LatentMAS 不是：
它传输的是逐层 `past_key_values`，RoPE 位置编码在发送侧即已烘焙进 K，且
Transformers 的 `generate` 没有把 KV cache 插入序列中段的接口。上游 LatentMAS
本身也始终前置。

因此 V3 中：

- `true_latentmas` 的**可见文本**与 `true_statebridge` 逐字相同；
- 其**实际载荷**仍作为因果前缀位于全部 token 之前。

论文口径：LatentMAS 的通信机制不包含明文信息槽位，因此无法与文本通道做完全
对等的位置控制；我们完全遵循其原始方法，并把这一不对称明确列为比较边界，不
将由此产生的差异解释为表示能力差异。

---

## 8. 已决定的范围

2026-09-19 由实验负责人确认：

| 项 | 决定 |
|---|---|
| MBPP+ | **不纳入本轮**。代码任务本轮只有 HumanEval+ |
| self / other 因果对照 | **本轮不跑**。后续单独选一个数据集做 ablation |
| replication | **单个 `seed_pair_00`** |
| 判分器回灌 V2 旧数据 | **不做**（理由见下） |

不回灌的理由：S1/S2 只影响数值答案，MedQA 是选择题，不适用；S3/S4 只可能改变
当前被判为解析失败的记录，而三 seed pooled 的 12,600 条记录中解析失败合计仅
**6 条**（none 3、self_text 1、other_text 1、other_statebridge 1），占 0.05%，
且 MedQA 本轮会用 V3 重跑。因此改用单元测试验证新解析器，测试用例必须覆盖
`\boxed{2,125}`、`\boxed{2125}`、`\boxed{\text{C}}`、`\boxed{C. Colorectal cancer}`
与 `#### 2,125` 这五种既有失败形态。

## 9. 待决定项

1. 后续 self/other ablation 选哪个数据集、在哪个阶段进行。
2. 位置 ablation（external 段置于最前）是否需要，以及放在哪个数据集上。

## 10. 规模估算

单 replication、条件为 `none + true_text + true_statebridge + true_latentmas`：

| 数据集 | items | prebelief | revision | 小计 |
|---|---:|---:|---:|---:|
| medqa | 300 | 600 | 2,400 | 3,000 |
| arc_challenge | 1,165 | 2,330 | 9,320 | 11,650 |
| gsm8k | 1,319 | 2,638 | 10,552 | 13,190 |
| gpqa | 198 | 396 | 1,584 | 1,980 |
| humanevalplus | 164 | 328 | 1,312 | 1,640 |
| **合计** | **3,146** | 6,292 | 25,168 | **31,460** |

4×RTX 5090 并行，按实测吞吐再行确认排期。正式排队前先用 `--limit 2` smoke 实测
单条生成延迟。
