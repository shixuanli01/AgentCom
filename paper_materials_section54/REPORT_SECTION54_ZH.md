# Section 5.4 — Sensitivity to Receiver Revision Policies

生成 2026-09-23。analysis seed 20260923，10,000 次题目聚类配对 bootstrap。
GPQA-D 四通道 V4 补跑完成；MedQA StateBridge 两侧统一为 `seed_pair_01`，比较已受控。

评价范围：混合正确性方向（CR 组 + PR 组）。所有分母从逐条记录核验。

---

## A. V3 / V4 究竟改变了什么

模板全文与机械 diff 见 [`receiver_policy_prompts.md`](receiver_policy_prompts.md)。

| | V3 | V4 |
|---|---|---|
| `PROMPT_VERSION` | `icr_v3_mid_injection` | `icr_v4_verify_then_decide` |
| 文件 | `icr/prompts_v3.py` | `icr/prompts_v4.py` |
| sha256 | `d31e3c61af43e15b…` | `ef94638977e2b3f3…` |

**差异完全局限在第 4 块「如何整合」。** 逐项核对确认其余未动：

- 第 1–3 块（`Original problem` / `Your previous reasoning` / `Your previous answer`）**逐字节相同**；
- V4 **不定义自己的** `external_block`，直接沿用 `icr.prompts_v3.external_block()`，
  各通道的消息块**逐字节相同**，注入位置与五块顺序不变；
- `ANSWER_FORMAT` 由 V4 从 V3 `import`，是**同一个对象**；
- 缺失先验答案的占位都是 `UNPARSEABLE`。

被替换掉的（V3，415 字符）：

```text
Your task is to REVISE your belief, not to restart from scratch.
Evaluate your previous reasoning and any external message critically.
* Change your answer only if you find a concrete error in your previous reasoning,
  or evidence that is better supported than it.
* Do not change your answer merely because an external message is present.
* Resolve any disagreement using the evidence in the original problem.
```

替换上去的（V4，657 字符）：

```text
Work through two steps, in this order.
Step 1 - Check the external message against the original problem. ...
  Do not compare it to your previous answer while doing this. Say which of its
  claims hold and which do not. ...
Step 2 - Check your previous reasoning the same way, against the problem.
  Say which of its claims hold and which do not.
Then give the answer supported by the claims that survived both steps. ...
```

**「只改了一句」是不准确的描述。** 改动限于一个块，但该块被整体重写：
1 句任务陈述 + 1 句评估要求 + 3 条 bullet → 2 个编号步骤 + 1 条决策规则。

**一个必须声明的副作用**：V4 额外**要求 receiver 输出中间检验内容**
（"Say which of its claims hold and which do not"），V3 没有这个要求。
因此 V4 改变的不只是判定准则，还改变了 receiver 生成内容的结构与长度。
把 V3→V4 的全部差异归因于「判定准则」会过度归因。

---

## B. 覆盖与可比性

表：[`receiver_policy_completeness.csv`](receiver_policy_completeness.csv)、
[`receiver_policy_provenance.json`](receiver_policy_provenance.json)

### B.1 覆盖情况

| 数据集 | 条件 | V3 | V4 |
|---|---|---|---|
| MedQA | No Message / Full Text / StateBridge / LatentMAS | 116/116 ✓ | 116/116 ✓ |
| MedQA | Answer Only | 116/116 ✓ | MISSING |
| GPQA-D | No Message / Full Text / StateBridge / LatentMAS | 114/114 ✓ | **114/114 ✓** |
| GPQA-D | Answer Only | 114/114 ✓ | MISSING |

GPQA-D 的四个通道 × V4 于 2026-09-23 补跑完成，**912 条修订**。
LatentMAS 首次尝试因每卡 2 worker 触发 CUDA OOM，只写了 135/228；
改为**每卡 1 worker**（潜空间条件的规定并发）续跑后达到 228/228，0 失败。

### B.2 可比性 —— 逐条核验

| 检查项 | MedQA（每条件 232） | GPQA-D（每条件 228） |
|---|---|---|
| `revision_seed` 逐条相同 | 232/232（StateBridge 除外，见 B.3） | **228/228，四条件全部** |
| `receiver_prior_sha256` 相同 | 232/232 | 228/228 |
| 载荷（剔除墙钟）相同 | 232/232 | 228/228 |
| `prompt_sha256` 相同 | 0/232（被操纵变量） | 0/228 |

### B.3 MedQA StateBridge 的跨种子问题 —— **已解决**

全文 MedQA StateBridge 采用 `seed_pair_01`。V3 来自 `runs/medqa/replication_01`，
**V4 来自 `runs/medqa/v4_verify_seed01`**（2026-09-24 补跑，232 条，0 失败）。
`revision_seed` 匹配 **232/232**，载荷 232/232 相同，`prompt_sha256` 0/232 相同。
**所有格子现在都是 seed 受控的。**

**这次补跑改变了结论。** 此前跨种子版本给出
ΔSI = −6.47 [−11.64, −1.29]（区间不含零）；受控后是
**−5.60 [−12.07, +0.86]（区间含零）**。
**跨种子版本是假阳性**，那 0.87 pp 的差与区间的收窄都来自修订采样而非接收策略。
这是本研究中修订种子混淆导致虚假显著的直接实例，值得写入论文。

`runs/medqa/v4_verify` 的 `true_statebridge`（seed_pair_00）已不再用于 MedQA。

### B.4 StateBridge 载荷未被重算

`messages/` 均为符号链接；`true_statebridge` 的载荷在 V3/V4 间逐字节相同；
`true_latentmas` 仅 `communication_seconds` 不同。「只改 Receiver 指令」对消息内容成立。

---

## C. 各通道的纠错、保护与改变率

### C.1 分层结果

| 数据集 | 条件 | V3 CR | V3 PR | V3 SI | V4 CR | V4 PR | V4 SI |
|---|---|---:|---:|---:|---:|---:|---:|
| MedQA | No Message | 6.90% | 98.28% | 52.59 | 3.45% | 99.14% | 51.29 |
| MedQA | Full Text | 80.17% | 42.24% | 61.21 | 56.90% | 67.24% | 62.07 |
| MedQA | StateBridge | 54.31% | 71.55% | **62.93** | 27.59% | 87.07% | 57.33 |
| MedQA | LatentMAS | 69.83% | 32.76% | 51.29 | 43.97% | 78.45% | 61.21 |
| GPQA-D | No Message | 14.04% | 96.49% | 55.26 | 7.02% | 97.37% | 52.19 |
| GPQA-D | Full Text | 74.56% | 47.37% | 60.96 | 55.26% | 61.40% | 58.33 |
| GPQA-D | StateBridge | 64.04% | 51.75% | 57.89 | 50.88% | 70.18% | 60.53 |
| GPQA-D | LatentMAS | 61.40% | 50.88% | 56.14 | 43.86% | 74.56% | 59.21 |

MedQA StateBridge 两侧均为 `seed_pair_01`：V3 63/116、83/116；V4 32/116、101/116。

### C.2 CR↓ / PR↑ —— 唯一跨数据集稳健的结果

| 数据集 | 条件 | 分层 | 都对 / V3错V4对 / V3对V4错 / 都错 | Δ (pp) | 95% CI |
|---|---|---|---|---:|---|
| MedQA | Full Text | CR | 62 / 4 / 31 / 19 | −23.28 | [−31.90, −14.66] |
| MedQA | Full Text | PR | 42 / 36 / 7 / 31 | +25.00 | [+14.66, +35.34] |
| MedQA | StateBridge | CR | 28 / 4 / 35 / 49 | −26.72 | [−36.21, −17.24] |
| MedQA | StateBridge | PR | 82 / 19 / 1 / 14 | +15.52 | [+7.76, +24.14] |
| MedQA | LatentMAS | CR | 35 / 16 / 46 / 19 | −25.86 | [−39.66, −12.07] |
| MedQA | LatentMAS | PR | 33 / 58 / 5 / 20 | +45.69 | [+34.48, +56.90] |
| GPQA-D | Full Text | CR | 57 / 6 / 28 / 23 | −19.30 | [−29.82, −8.77] |
| GPQA-D | Full Text | PR | 44 / 26 / 10 / 34 | +14.04 | [+3.51, +24.56] |
| GPQA-D | StateBridge | CR | 49 / 9 / 24 / 32 | −13.16 | [−22.81, −2.63] |
| GPQA-D | StateBridge | PR | 52 / 28 / 7 / 27 | +18.42 | [+8.77, +28.07] |
| GPQA-D | LatentMAS | CR | 43 / 7 / 27 / 37 | −17.54 | [−28.07, −7.02] |
| GPQA-D | LatentMAS | PR | 48 / 37 / 10 / 19 | +23.68 | [+11.40, +35.96] |

**12 个区间全部不跨零。** `No Message` 上两个数据集都跨零（MedQA −3.45 / +0.86，GPQA-D −7.02 / +0.88）。

### C.3 ΔSI

| 数据集 | 条件 | ΔSI | 95% CI | 跨零 | seed 受控 |
|---|---|---:|---|---|---|
| MedQA | No Message | −1.29 | [−3.88, +1.29] | 是 | 是 |
| MedQA | Full Text | +0.86 | [−5.17, +6.90] | 是 | 是 |
| MedQA | StateBridge | −5.60 | [−12.07, +0.86] | 是 | 是 |
| MedQA | **LatentMAS** | **+9.91** | **[+1.29, +18.53]** | **否** | 是 |
| GPQA-D | No Message | −3.07 | [−7.02, +0.88] | 是 | 是 |
| GPQA-D | Full Text | −2.63 | [−9.21, +4.39] | 是 | 是 |
| GPQA-D | StateBridge | +2.63 | [−4.39, +9.21] | 是 | 是 |
| GPQA-D | LatentMAS | +3.07 | [−4.39, +10.53] | 是 | 是 |

### C.4 改变率 —— 8/8 一律下降

| 数据集 | No Message | Full Text | StateBridge | LatentMAS |
|---|---|---|---|---|
| MedQA | 4.31% → 2.16% | 68.97% → 44.83% | 41.38% → 21.55% | 68.53% → 32.76% |
| GPQA-D | 11.89% → 8.07% | 63.44% → 47.37% | 56.39% → 40.79% | 54.46% → 33.63% |

---

## D. 相对 No Message 的通信增量

| 数据集 | 通道 | G(V3) | G(V4) | ΔG | 95% CI | 跨零 |
|---|---|---:|---:|---:|---|---|
| MedQA | Full Text | +8.62 | +10.78 | +2.16 | [−3.88, +8.19] | 是 |
| MedQA | StateBridge | +10.34 | +6.03 | −4.31 | [−11.64, +2.59] | 是 |
| MedQA | **LatentMAS** | −1.29 | +9.91 | **+11.21** | **[+2.16, +20.26]** | **否** |
| GPQA-D | Full Text | +5.70 | +6.14 | +0.44 | [−7.89, +8.77] | 是 |
| GPQA-D | StateBridge | +2.63 | +8.33 | +5.70 | [−2.19, +13.60] | 是 |
| GPQA-D | LatentMAS | +0.88 | +7.02 | +6.14 | [−2.63, +15.35] | 是 |

**LatentMAS 的 ΔG 在两个数据集上同号**（+11.21 / +6.14），但只有 MedQA 有区间支持。
**StateBridge 的 ΔG 在两个数据集上异号**（−5.17 / +5.70），两者都跨零。

---

## E. 通道相对优势是否对策略敏感 —— 两个数据集都不支持

| 数据集 | V3 SB−FT | V4 SB−FT | I | 95% CI | 跨零 |
|---|---:|---:|---:|---|---|
| MedQA | +1.72 | −4.74 | −6.47 | [−14.22, +1.29] | **是** |
| GPQA-D | −3.07 | +2.19 | +5.26 | [−4.39, +14.91] | **是** |

两个数据集的点估计**符号相反**，两个区间**都跨零**。

**结论：现有证据不支持通道 × 策略交互存在。** 之前基于 seed_pair_00 报告的
MedQA 交互 −10.78 [−18.97, −2.59] 已随 StateBridge 统一到 seed_pair_01 而失效，
「排名反转」这一表述在任何数据集上都没有区间支撑，应从摘要、引言、4.4 与结论中撤下。

---

## F. 区间支持 vs 点估计

### F.1 两个数据集都有区间支持

1. **V4 在每个有消息的通道上显著压低 CR、抬高 PR** —— 12 个区间全部不跨零（C.2）。
   这是 Section 5.4 唯一稳健的发现。

### F.2 仅 MedQA 有区间支持

2. LatentMAS ΔSI +9.91 [+1.29, +18.53] —— GPQA-D 同号 (+3.07) 但跨零，**未复现**
3. LatentMAS ΔG +11.21 [+2.16, +20.26] —— GPQA-D 同号 (+6.14) 但跨零，**未复现**
（原第 4 条 StateBridge ΔSI 已移入 F.3：受控后区间跨零。）

### F.3 任何数据集都无区间支持

5. 通道 × 策略交互（MedQA −6.47、GPQA-D +5.26，均跨零）
6. StateBridge ΔG（MedQA −4.31、GPQA-D +5.70，均跨零）
6b. **StateBridge ΔSI**（MedQA −5.60 [−12.07, +0.86]、GPQA-D +2.63，均跨零）。
   受控前的 −6.47 [−11.64, −1.29] 是跨种子假阳性，见 B.3
7. `No Message` 的 ΔSI（MedQA −1.29、GPQA-D −3.07，均跨零）。
   **「V4 降低协议噪声」只是点估计**，且 GPQA-D 上一半以上效应来自 5 条截断记录
   （见 `VERIFICATION_2026-09-23_transitions.md`）
8. Full Text 的 ΔSI 与 ΔG（四个区间全跨零）

### F.4 未验证的解释

- 「V4 更保守」—— 改变率 8/8 下降是描述，不是机制。
- 「V4 的效应来自判定准则」—— V4 同时改变了输出结构（A 节副作用），无法分离。
- 「隐状态通道需要更宽松的接收方」—— 被 GPQA-D 的相反方向直接削弱。

---

## G. 缺失项

| 缺失项 | 需要 GPU | 记录数 |
|---|---|---:|
| MedQA / GPQA-D × Answer Only × V4 | 是 | 460 |
| GPQA-D V4 的 SR / SCR / Acc_ret | 是 | 约 480（补四方向） |
| 第三个数据集的 V3/V4 | 是 | ARC-C 784 或 GSM8K 736（四通道混合方向） |

### G.1 优先级

跨种子问题已在 2026-09-24 解决，并使 StateBridge 的 ΔSI 失去区间支持。
现在 §5.4 只剩两类有区间支持的结果：
**所有通道的 CR↓/PR↑（两个数据集，12 个区间）**，以及
**MedQA LatentMAS 的 ΔSI 与 ΔG（未在 GPQA-D 复现）**。

**下一步最有价值的是第三个数据集**（ARC-C 784 条或 GSM8K 736 条，四通道混合方向）：
唯一还开放的问题是 LatentMAS 那两条只在 MedQA 成立的结果究竟是真实效应还是噪声。
补更多通道价值最低——StateBridge 与 LatentMAS 都已显示效应不跨数据集复现。

---

本轮未修改论文 LaTeX，未覆盖任何已有图。
