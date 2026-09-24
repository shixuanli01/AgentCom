# Section 5.4 — V3 / V4 Receiver prompt 的精确差异

生成 2026-09-23。本文件给出两份模板全文、机械 diff，并确认除整合块外**没有其他修改**。

## 1. 版本、文件与 hash

| 项 | V3 | V4 |
|---|---|---|
| `PROMPT_VERSION` | `icr_v3_mid_injection` | `icr_v4_verify_then_decide` |
| 文件 | `icr/prompts_v3.py` | `icr/prompts_v4.py` |
| 文件 sha256 | `d31e3c61af43e15b6c459ef871a18c8ba6481351b675baa90f57470eb5749b23` | `ef94638977e2b3f38fdd5518296dcd3c19d2e655a05790f5e96af142f28427ad` |
| 文件字节数 | 6157 | 4064 |
| 非代码模板 | `REVISION_TEMPLATE` | `VERIFY_TEMPLATE` |
| 模板 sha256 | `e176b57fb1a8b62a5ddd4271261316cd7d441cc64f98dd7971834a98ff2cd4f3` | `36d0e21d7485f7af1adb0cdea1a994a4aeec82c9aa1860f4fec45c6f18d4441d` |
| 代码模板 | `CODE_REVISION_TEMPLATE` | `CODE_VERIFY_TEMPLATE` |
| 构造函数 | `revision_prompt(...)` | `revision_prompt_v4(...)` |

## 2. V3 完整 Receiver 模板（非代码任务）

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

## 3. V4 完整 Receiver 模板（非代码任务）

```text
You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_block}

Work through two steps, in this order.

Step 1 - Check the external message against the original problem.
Take each claim it makes and test it against the facts stated in the problem.
Do not compare it to your previous answer while doing this. Say which of its
claims hold and which do not. If no external message is available, say so and
go to Step 2.

Step 2 - Check your previous reasoning the same way, against the problem.
Say which of its claims hold and which do not.

Then give the answer supported by the claims that survived both steps. If the
surviving claims point to the answer you already gave, keep it. If they point
elsewhere, change it.

{answer_format}
```

## 4. 逐行 diff（非代码任务）

```diff
--- icr/prompts_v3.py :: REVISION_TEMPLATE
+++ icr/prompts_v4.py :: VERIFY_TEMPLATE
@@ -11,12 +11,19 @@
 
 {external_block}
 
-Your task is to REVISE your belief, not to restart from scratch.
+Work through two steps, in this order.
 
-Evaluate your previous reasoning and any external message critically.
+Step 1 - Check the external message against the original problem.
+Take each claim it makes and test it against the facts stated in the problem.
+Do not compare it to your previous answer while doing this. Say which of its
+claims hold and which do not. If no external message is available, say so and
+go to Step 2.
 
-* Change your answer only if you find a concrete error in your previous reasoning, or evidence that is better supported than it.
-* Do not change your answer merely because an external message is present.
-* Resolve any disagreement using the evidence in the original problem.
+Step 2 - Check your previous reasoning the same way, against the problem.
+Say which of its claims hold and which do not.
+
+Then give the answer supported by the claims that survived both steps. If the
+surviving claims point to the answer you already gave, keep it. If they point
+elsewhere, change it.
 
 {answer_format}
```

## 5. 逐行 diff（代码任务模板，供完整性）

```diff
--- icr/prompts_v3.py :: CODE_REVISION_TEMPLATE
+++ icr/prompts_v4.py :: CODE_VERIFY_TEMPLATE
@@ -3,18 +3,22 @@
 Original problem:
 {question}
 
-Your previous reasoning and implementation:
+Your previous implementation:
 {receiver_prior_reasoning}
 
 {external_block}
 
-Your task is to REVISE your implementation, not to restart from scratch and not to copy
-an external message.
+Work through two steps, in this order.
 
-Evaluate your previous implementation and any external message critically.
+Step 1 - Check the external message against the original problem.
+Test what it does against the stated requirements and against the examples in
+the problem. Do not compare it to your previous implementation while doing
+this. Say where it is correct and where it fails. If no external message is
+available, say so and go to Step 2.
 
-* Change your implementation only if you find a concrete defect in it, or an approach that is better supported than it.
-* Do not change your implementation merely because an external message is present.
-* Check the required function signature, imports, examples, and edge cases against the original problem.
+Step 2 - Check your previous implementation the same way, against the problem.
+Say where it is correct and where it fails.
+
+Then give the implementation supported by what survived both steps.
 
 {answer_format}
```

## 6. 修改的究竟是哪一段

差异**完全局限在第 4 块（“如何整合”）**。以下是被替换掉的与替换上去的原文。

### 被移除（V3）

```text
Your task is to REVISE your belief, not to restart from scratch.

Evaluate your previous reasoning and any external message critically.

* Change your answer only if you find a concrete error in your previous reasoning, or evidence that is better supported than it.
* Do not change your answer merely because an external message is present.
* Resolve any disagreement using the evidence in the original problem.
```

### 被加入（V4）

```text
Work through two steps, in this order.

Step 1 - Check the external message against the original problem.
Take each claim it makes and test it against the facts stated in the problem.
Do not compare it to your previous answer while doing this. Say which of its
claims hold and which do not. If no external message is available, say so and
go to Step 2.

Step 2 - Check your previous reasoning the same way, against the problem.
Say which of its claims hold and which do not.

Then give the answer supported by the claims that survived both steps. If the
surviving claims point to the answer you already gave, keep it. If they point
elsewhere, change it.
```

字符数：V3 整合块 411，V4 整合块 653。

## 7. 除该段之外是否还有其他修改 —— 逐项核对

| 检查项 | 结果 |
|---|---|
| 第 1–3 块（`Original problem` / `Your previous reasoning` / `Your previous answer`）逐字节 | **完全相同** |
| `{external_block}` 渲染函数 | V4 **不定义自己的** `external_block`，直接沿用 `icr.prompts_v3.external_block()` → 各通道的消息块**逐字节相同** |
| `ANSWER_FORMAT` 字典 | V4 从 V3 `import`，是**同一个对象**（`is` 判定为 True）|
| `answer_format('medqa')` | 相同：True |
| `answer_format('gpqa')` | 相同：True |
| `CODE_TASKS` | V4 从 V3 `import`，同一对象 |
| 消息注入位置 | 两者都是第 4 个占位符之前的 `{external_block}`，块序 `[题目] → [自身先验] → [外部信息] → [如何整合] → [输出格式]` 未变 |
| 缺失先验答案的占位 | 两者都用 `UNPARSEABLE` |

**结论：只改了整合块，但不是“只改了一句”。** V3 的 1 句任务陈述 + 1 句评估要求 + 3 条 bullet （共 415 字符）被整体替换为 V4 的 2 个编号步骤 + 1 条决策规则（共 657 字符）。

### 需要在论文中声明的一个副作用

V4 额外**要求 receiver 输出中间内容**（“Say which of its claims hold and which do not”），V3 没有这个要求。因此 V4 改变的不只是判定准则，还改变了 receiver 生成内容的结构与长度。把 V3→V4 的全部差异都归因于“判定准则”会过度归因。

## 8. V4 自述的设计意图（取自 `icr/prompts_v4.py` docstring）

V4 的文件头声明它针对的是：V3 的指令把举证责任放在“在**自己**的推理中找出具体错误”上，因此 receiver 对自己越没把握就越让步；V4 改为**先把外部消息单独对照原题检验**（明确禁止此时与自己的答案比较），再以同样方式检验自己的推理，最后由两步都存活的断言决定答案。

该 docstring 同时声明：**“This is a protocol version, not a prompt tweak: V3 results are not comparable across it, and every channel must be rerun under it.”** 这句话是作者当时的设计声明，本轮的实际核验（见 `receiver_policy_provenance.json`）表明 MedQA 上四个通道确实都在 V4 下重跑过，且 `revision_seed` 逐条相同，因此 MedQA 的 V3/V4 比较是受控的。
