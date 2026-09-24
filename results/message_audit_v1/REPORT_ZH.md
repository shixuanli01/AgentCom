# Section 5.3 消息内容审计 — 报告

生成时间 2026-09-23。analysis seed 20260923，bootstrap 10,000 次题目聚类配对重采样。
冻结配置见 [`manifest.json`](manifest.json)。

本轮回答三个问题，不寻找表现最好的新方法：

1. 完整推理相比只传答案，提供了什么增量？
2. 选择消息内容是否优于等预算随机选择？
3. 这些方法是否超越随机减少消息发送所能达到的表现？

---

## A. 运行完整性和比较范围

### A.1 实际完成量

| 项目 | 数量 |
|---|---:|
| 新增 Receiver revision 调用 | **0** |
| 复用的 Answer Only 记录 | **460**（MedQA 232 + GPQA-D 228） |
| 复用的基线记录 | 1,840（4 通道 × 460） |
| 消息构造调用 | 0（`answer_only` 是字符串格式化，不调模型） |
| 未完成任务 | `random_spans`、`selected_spans` —— **BLOCKED** |

`answer_only` 在本轮开始前已跑完，且经核验与本轮规格完全一致，因此**没有重复提交任务**。

### A.2 数量核验（从 phase-1 记录重算，非沿用旧 summary）

| 数据集 | 三信念完整题数 | 封闭式 CR = PR = Σk(3−k) | 实际 answer_only 记录 |
|---|---:|---:|---:|
| MedQA | 300 | 116 | 116 + 116 = **232** |
| GPQA-D | 198 | 114 | 114 + 114 = **228** |
| 合计 | | | **460** |

与预期完全一致，没有删除或补造任何记录。

### A.3 兼容性核验

| 检查项 | 结果 |
|---|---|
| answer_only 与 4 个基线的 (item, sender, receiver) 键集合 | **完全相同**，缺 0 多 0 |
| `revision_seed` 逐条比对 | **460/460 相同** |
| phase-1 轨迹 | 符号链接指向同一 `artifacts/icr_v3/<ds>_full_seed42/prebeliefs`，未改动 |
| revision prompt | 同为 `icr_v3_mid_injection`（V3），**未使用 V4** |
| parser / scorer | 同为 `icr.parsing_v3@ICR-V3` |
| `replication_id` | 同为 `seed_pair_00` |

**发现并已解决的一处不兼容嫌疑**：MedQA 与 GPQA-D 的基线记录各含两个
`config_fingerprint`。追查 `superseded_fingerprints` 确认两者**仅差 `max_new_tokens: 8192 → 16384`**。
进一步核验：受旧配置影响的混合方向记录中，**没有任何一条接近 8192 上限**
（MedQA 最长 2994，GPQA-D 最长 6098），未终止记录为 0。
因此该上限在本轮分析范围内**从未生效**，两个 fingerprint 行为等价。

### A.4 StateBridge 来源声明

MedQA 存在两个 StateBridge run。本轮**只使用** `artifacts/icr_v3/medqa_full_seed42`，
`replication_id = seed_pair_00`，`global_seed = 42`。
`runs/medqa/replication_01`（`seed_pair_01`）**未使用**，没有把两个 run 的 CR/PR/Acc 或区间拼接。
StateBridge 本轮不重跑，仅作参照。

### A.5 作用域限制

本轮**只报告混合方向**（CR 组 + PR 组）。
新条件的 `Acc_ret`、全集 Acc、SR、SCR **一律不报告**，因为没有覆盖对应评价范围。
也没有把旧的"全部 retained 改变率"与本轮混合组改变率直接比较。

恒等式 `Acc_mixed = SI` 在 **10/10 个格子上验证通过**（依赖 CR、PR 分母相等，本轮均相等）。

---

## B. 完整推理相比只传答案的增量

表：[`mixed_results.csv`](mixed_results.csv)、[`paired_differences.csv`](paired_differences.csv)

### B.1 主结果

| 数据集 | 条件 | CR | PR | SI | Acc_mixed |
|---|---|---:|---:|---:|---:|
| MedQA | No Message | 6.90% | 98.28% | 52.59% | 52.59% |
| MedQA | Answer Only | 38.79% | 80.17% | 59.48% | 59.48% |
| MedQA | Full Text | 80.17% | 42.24% | 61.21% | 61.21% |
| GPQA-D | No Message | 14.04% | 96.49% | 55.26% | 55.26% |
| GPQA-D | Answer Only | 56.14% | 63.16% | 59.65% | 59.65% |
| GPQA-D | Full Text | 74.56% | 47.37% | 60.96% | 60.96% |

### B.2 配对差异（题目聚类 bootstrap，10,000 次）

| 数据集 | 比较 | ΔCR | ΔPR | **ΔSI** | 95% CI | 判定 |
|---|---|---:|---:|---:|---|---|
| MedQA | Answer Only − No Message | +31.90 | −18.10 | **+6.90** | [+0.86, +12.93] | Answer Only 更高 |
| MedQA | **Full Text − Answer Only** | +41.38 | −37.93 | **+1.72** | [−4.31, +7.76] | **证据不确定** |
| GPQA-D | Answer Only − No Message | +42.11 | −33.33 | **+4.39** | [−3.51, +11.84] | **证据不确定** |
| GPQA-D | **Full Text − Answer Only** | +18.42 | −15.79 | **+1.32** | [−6.14, +8.77] | **证据不确定** |

### B.3 结论

**完整推理相比只传答案，在 SI 上没有可检出的增量。** 两个数据集的 ΔSI 都是小正值
（+1.72、+1.32），两个区间都跨零。按本轮口径，这写作"证据不确定"，
**不能写成"两者等价"**。

但 ΔCR 和 ΔPR 都很大且方向相反：MedQA 上 Full Text 把 CR 抬高 41.38 个百分点，
同时把 PR 压低 37.93 个百分点，两者几乎抵消。消息长度差距是 **1 token vs 中位 487 token**
（见 [`message_audit.csv`](message_audit.csv)）。

也就是说：**多传约 490 倍的 token，换来的是 receiver 改得更多，而不是改得更对。**

**不能据此断言 receiver 没有阅读推理。** 答案迁移表
（[`answer_transitions.csv`](answer_transitions.csv)）显示 receiver 对两种消息的反应差别巨大，
说明它显然处理了文本。本轮设计不采集 receiver 的处理过程，无法支持关于阅读行为的断言。
另外，"匹配 Sender 答案"只是输出一致，不等于因果意义上的采纳。

---

## C. 选择片段相比随机片段的增量 — **BLOCKED**

**无法回答。**

仓库中不存在 E2 Random / E3 Selected 的实现：没有片段选择器、没有候选池构造、
没有 token 预算配置。全仓库搜索 `E0`–`E4`、`random_span`、`selected_span`、
`span_budget`、`candidate_pool` 在 `*.py`/`*.md`/`*.yaml` 中无命中。

这不是本轮新发现——它已经是 [`../AUDIT_ISSUES.md`](../AUDIT_ISSUES.md) 的 **C1（MISSING，blocking）**，
2026-09-21 的审计就记录过"本仓库不存在 75 题 MedQA pilot 和 E0–E4 变体"。

按执行要求，**没有临时发明替代算法**，也没有用现有的 EGR 句级删除冒充 span 选择——
EGR 删的是"含答案断言的整句"，不是"在预算下选片段"，两者的候选池、预算和包装格式都不同，
把它当作 E3 会是错误的复现。

要关闭这一项，必须从仓库外导入原始 E2/E3 的选择器、候选池构造和预算配置。

---

## D. 方法是否超越随机发送参照

表：[`random_delivery_reference.csv`](random_delivery_reference.csv)、
[`matched_reference_comparisons.csv`](matched_reference_comparisons.csv)

### D.1 参照定义

一个与题目、消息内容、正确性都无关的策略：以概率 α 发送 Full Text，以概率 1−α 发送 No Message。
期望值直接计算，不做额外随机抽样：

```
CR(α) = α·CR_full + (1−α)·CR_none
PR(α) = α·PR_full + (1−α)·PR_none
```

α 网格 0 到 1，步长 0.05。

### D.2 相同 PR 下的 CR 比较

`α = (PR_target − PR_none) / (PR_full − PR_none)`，仅在分母非零且 α ∈ [0,1] 时计算。

| 数据集 | 条件 | target PR | 匹配 α | 参照 CR | 实测 CR | **差值** | 95% CI | 可匹配比例 | 区间稳定 |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| MedQA | Answer Only | 80.17% | 0.323 | 30.57% | 38.79% | **+8.22** | [−2.99, +19.86] | 100.0% | YES |
| MedQA | **StateBridge** | 75.00% | 0.415 | 37.33% | 57.76% | **+20.42** | [+6.36, +35.22] | 100.0% | YES |
| MedQA | LatentMAS | 32.76% | 1.169 | — | — | `OUT_OF_RANGE` | — | — | — |
| GPQA-D | Answer Only | 63.16% | 0.679 | 55.11% | 56.14% | **+1.03** | [−14.52, +15.57] | 100.0% | YES |
| GPQA-D | StateBridge | 51.75% | 0.911 | 69.16% | 64.04% | −5.12 | [−16.42, +9.70] | **82.5%** | **NO** |
| GPQA-D | LatentMAS | 50.88% | 0.929 | 70.24% | 61.40% | −8.83 | [−19.64, +8.55] | **75.1%** | **NO** |

Full Text 的 α 恒为 1.000、差值恒为 0，这是参照端点本身，不是结果。

### D.3 结论

**只有 MedQA 上的 StateBridge 确定性地超越了随机发送参照**：相同 PR 下 CR 高出 20.42 个百分点，
区间 [+6.36, +35.22] 不跨零，10,000 次重采样 100% 可匹配。

其余全部不确定：

- MedQA Answer Only +8.22，区间跨零 → 证据不确定。
- GPQA-D Answer Only +1.03，区间几乎以零为中心 → 证据不确定。
- GPQA-D 的 StateBridge 与 LatentMAS 差值为负，但**只有 82.5% / 75.1% 的重采样可匹配**，
  低于 95% 阈值，按规则标记**区间不稳定**，其有效子样本分位数**不作为常规 95% CI 使用**。
  不能据此说它们低于随机参照。
- MedQA LatentMAS 的 target PR = 32.76%，低于 PR_full = 42.24%，落在可匹配区间外，
  标记 `OUT_OF_RANGE`。**未外推，未把 α 裁剪到 [0,1]。**

### D.4 解释边界

这条参照是**随机减少消息暴露**的参照线，它**不是**通信能力上限，**不是** ROC 曲线，
**也不是**在独立测试集上验证过的最优门控策略。
StateBridge 与 LatentMAS 的记录纳入了参照比较，它们与 Answer Only 共享同一批初始轨迹和
同一 Receiver prompt（见 A.3），没有为纳入它们而混合不同配置。

---

## E. 不确定的发现与限制结论的配置

### E.1 不确定的

1. **Full Text 相对 Answer Only 的增量** —— 两个数据集区间都跨零。
2. **Answer Only 相对 No Message 的增量** —— MedQA 确定（[+0.86, +12.93]），GPQA-D 不确定。
3. **GPQA-D 上任何方法与随机参照的比较** —— 可匹配比例不足，区间不稳定。

### E.2 限制结论的配置

| 限制 | 影响 |
|---|---|
| **只有 2 个数据集、单次采样** | `replication_id = seed_pair_00` 单一。MedQA StateBridge 的第二个种子显示 SI 漂移 3.45 pp，说明单次采样的不确定性不可忽略 |
| **只有混合方向** | 新条件的 SR / SCR / Acc_ret / 全集 Acc 全部未测，**不可用推算值填充** |
| **样本量小** | CR/PR 每组仅 116 / 114 条，这是所有区间偏宽的直接原因 |
| **E2/E3 缺失** | 问题 2 完全无法回答 |
| **GPQA-D 的 Full Text 消息含截断** | 消息 token 最大 **16383**，即某些 sender 信念本身撞到了生成上限。GPQA-D 的 CR/PR 子集截断污染率为 **10.53%**（见 `../05_INTEGRITY.md`），这是 GPQA-D 各项区间更宽、可匹配比例更低的可能原因之一 |
| **构造成本未记录** | `cost.csv` 中 message_construction 的 seconds 与 cache_reuse 标记 `MISSING`，不是 0 |
| **代码未提交** | commit `5ef9925`，另有 52 处未提交改动，已存 `uncommitted.patch` |

### E.3 本轮没有做的事

- 没有重跑 phase-1。
- 没有使用 V4 prompt。
- 没有改动任何已有 baseline 的原始记录。
- 没有因结果不好更换 seed、消息预算或选择器。
- 没有绘图，没有改动论文 LaTeX，没有覆盖 `figures/cr_pr_profiles.pdf`。
