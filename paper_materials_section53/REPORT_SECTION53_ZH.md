# Section 5.3 — What Do Communication Messages Contribute?

生成 2026-09-23。analysis seed 20260923，10,000 次题目聚类配对 bootstrap。
**本轮新增 GPU 调用 0 次**，全部为已有记录的审计、统计与导出。
冻结来源见 [`provenance.json`](provenance.json)。

比较范围：四个数据集的**混合正确性有向配对**（Sender 对/Receiver 错，Sender 错/Receiver 对）。
全部分母从 phase-1 逐条记录重算，未读取任何汇总表。

| 数据集 | 三信念完整题数 | 封闭式 Σk(3−k) | CR 分母 | PR 分母 | 与预期一致 |
|---|---:|---:|---:|---:|---|
| MedQA | 300 | 116 | 116 | 116 | YES |
| ARC-C | 1165 | 98 | 98 | 98 | YES |
| GSM8K | 1319 | 92 | 92 | 92 | YES |
| GPQA-D | 198 | 114 | 114 | 114 | YES |

重复键 0，相对 No Message 的缺失 0、多余 0。三个条件共享同一 phase-1 轨迹
（符号链接）、同一 V3 receiver prompt、同一 parser/scorer、同一 `replication_id = seed_pair_00`。
`Acc_mixed = SI` 在 **12/12 个格子**上成立。

---

## A. 完整推理相比答案直传的增量

表：[`answer_only_comparison.csv`](answer_only_comparison.csv)、[`paired_differences.csv`](paired_differences.csv)

### A.1 点估计

| 数据集 | 条件 | CR | PR | SI |
|---|---|---:|---:|---:|
| MedQA | No Message | 8/116 = 6.90% | 114/116 = 98.28% | 52.59% |
| MedQA | Answer Only | 45/116 = 38.79% | 93/116 = 80.17% | 59.48% |
| MedQA | Full Text | 93/116 = 80.17% | 49/116 = 42.24% | 61.21% |
| ARC-C | No Message | 8/98 = 8.16% | 91/98 = 92.86% | 50.51% |
| ARC-C | Answer Only | 38/98 = 38.78% | 62/98 = 63.27% | 51.02% |
| ARC-C | Full Text | 79/98 = 80.61% | 34/98 = 34.69% | 57.65% |
| GSM8K | No Message | 12/92 = 13.04% | 85/92 = 92.39% | 52.72% |
| GSM8K | Answer Only | 36/92 = 39.13% | 65/92 = 70.65% | 54.89% |
| GSM8K | Full Text | 47/92 = 51.09% | 48/92 = 52.17% | 51.63% |
| GPQA-D | No Message | 16/114 = 14.04% | 110/114 = 96.49% | 55.26% |
| GPQA-D | Answer Only | 64/114 = 56.14% | 72/114 = 63.16% | 59.65% |
| GPQA-D | Full Text | 85/114 = 74.56% | 54/114 = 47.37% | 60.96% |

### A.2 配对差异与区间

| 数据集 | 比较 | ΔCR (pp) | ΔPR (pp) | **ΔSI (pp)** | ΔSI 95% CI | 跨零 |
|---|---|---:|---:|---:|---|---|
| MedQA | Answer Only − No Message | +31.90 | −18.10 | **+6.90** | [+0.86, +12.93] | **否** |
| ARC-C | Answer Only − No Message | +30.61 | −29.59 | **+0.51** | [−6.63, +7.65] | 是 |
| GSM8K | Answer Only − No Message | +26.09 | −21.74 | **+2.17** | [−6.52, +10.33] | 是 |
| GPQA-D | Answer Only − No Message | +42.11 | −33.33 | **+4.39** | [−3.51, +11.84] | 是 |
| MedQA | Full Text − Answer Only | +41.38 | −37.93 | **+1.72** | [−4.31, +7.76] | 是 |
| ARC-C | Full Text − Answer Only | +41.84 | −28.57 | **+6.63** | [−0.51, +13.27] | 是 |
| GSM8K | Full Text − Answer Only | +11.96 | −18.48 | **−3.26** | [−12.50, +6.52] | 是 |
| GPQA-D | Full Text − Answer Only | +18.42 | −15.79 | **+1.32** | [−6.14, +8.77] | 是 |

### A.3 配对四格表：Full Text − Answer Only

基准 = Answer Only，比较条件 = Full Text。

| 数据集 | 分层 | 两者都对 | **gained** | **lost** | 两者都错 | 净 |
|---|---|---:|---:|---:|---:|---:|
| MedQA | CR | 42 | **51** | 3 | 20 | +48 |
| MedQA | PR | 46 | 3 | **47** | 20 | −44 |
| ARC-C | CR | 36 | **43** | 2 | 17 | +41 |
| ARC-C | PR | 30 | 4 | **32** | 32 | −28 |
| GSM8K | CR | 28 | **19** | 8 | 37 | +11 |
| GSM8K | PR | 41 | 7 | **24** | 20 | −17 |
| GPQA-D | CR | 55 | **30** | 9 | 20 | +21 |
| GPQA-D | PR | 46 | 8 | **26** | 34 | −18 |

四个数据集上，CR 组的净增益都被 PR 组的净损失几乎完全抵消：
混合组的**净正确数变化**分别是 **+4、+13、−6、+3**（总分母 232 / 196 / 184 / 228）。

### A.4 结论

**ΔCR 和 ΔPR 都很大且符号相反；ΔSI 很小，且 8 个比较中有 7 个区间跨零。**
唯一区间不跨零的是 MedQA 的 Answer Only − No Message（+6.90，[+0.86, +12.93]）。

ΔSI 的符号在数据集之间不一致：ARC-C 上 Full Text 领先 +6.63，GSM8K 上落后 −3.26。

消息长度对比（token 中位数）：

| 数据集 | Answer Only | Full Text | 倍数 |
|---|---:|---:|---:|
| MedQA | 1 | 487 | ~487× |
| ARC-C | 1 | 401 | ~401× |
| GSM8K | 2 | 309 | ~155× |
| GPQA-D | 1 | 731 | ~731× |

**已获支持的观察**：在混合正确性方向上，把消息从 1 个 token 扩展到数百个 token，
带来的是 CR 与 PR 的大幅反向移动，而不是 SI 的稳定提升。

**不能下的结论**：不能写"两者等价"或"完整推理没有效果"。本轮**没有做等价检验**，
区间跨零只说明证据不足以判定方向。

---

## B. 答案迁移如何解释上述差异

表：[`transitions_by_stratum.csv`](transitions_by_stratum.csv)

五个类别互斥、总数等于分母：`invalid_output`、`kept_receiver_prior`、
`matched_sender_answer`、`third_answer_sender_valid`、`changed_sender_answer_missing`。
Sender 初始答案缺失的记录单独归入最后一类，**未计入"匹配 Sender"**
（ARC-C 2 条、GPQA-D 12 条）。

### B.1 从 Answer Only 到 Full Text 的迁移变化

| 数据集 | 分层 | 保持 Receiver 原答案 | 匹配 Sender 答案 | 第三答案 |
|---|---|---|---|---|
| MedQA | CR | 71 → 23 (**−48**) | 45 → 93 (**+48**) | 0 → 0 (0) |
| MedQA | PR | 93 → 49 (**−44**) | 23 → 67 (**+44**) | 0 → 0 (0) |
| ARC-C | CR | 60 → 19 (**−41**) | 38 → 79 (**+41**) | 0 → 0 (0) |
| ARC-C | PR | 62 → 34 (**−28**) | 36 → 64 (**+28**) | 0 → 0 (0) |
| GSM8K | CR | 55 → 43 (−12) | 36 → 47 (+11) | 1 → 2 (+1) |
| GSM8K | PR | 65 → 48 (−17) | 23 → 43 (+20) | 4 → 1 (−3) |
| GPQA-D | CR | 48 → 29 (−19) | 64 → 85 (+21) | 2 → 0 (−2) |
| GPQA-D | PR | 72 → 54 (−18) | 38 → 58 (+20) | 3 → 1 (−2) |

**差异几乎完全表现为"保持原答案"向"匹配 Sender 答案"的转移。**
第三答案在两个多选数据集上恒为 0，在 GSM8K 和 GPQA-D 上变动不超过 3 条。

关键在于：这个转移在 **CR 组和 PR 组同时发生，方向相同、量级相近**。
MedQA 上 CR 组多匹配 48 次、PR 组多匹配 44 次；ARC-C 是 41 与 28。
也就是说 Full Text 并没有让 receiver 更会区分"该匹配"与"不该匹配"，
它只是让 receiver 在两种情形下都更倾向于匹配。

### B.2 有效答案改变率

无效输出（parser 失败）**从改变率的分子和分母中同时剔除**，不当作正常答案变化；
无效输出率单独报告。

| 数据集 | No Message | Answer Only | Full Text | 无效输出率 |
|---|---:|---:|---:|---:|
| MedQA | 4.31% | 29.31% | 68.97% | 0.00% |
| ARC-C | 9.23% | 37.76% | 72.96% | 0.51% / 0 / 0 |
| GSM8K | 11.96% | 34.78% | 50.54% | 0.00% |
| GPQA-D | 11.89% | 47.14% | 63.44% | 0.44% |

改变率单调上升：No Message < Answer Only < Full Text，四个数据集一致。

### B.3 措辞限制

"匹配 Sender 答案"**只表示输出一致**，不等于已经识别出的因果采纳。
本轮设计不采集 receiver 的处理过程，任何关于"receiver 是否阅读了推理"的断言
都超出本数据的支持范围。

以上改变率**仅在混合组计算**，未与全部 retained records 的改变率作比较。

---

## C. Answer Only 是否超越随机发送参照

表：[`random_delivery_reference.csv`](random_delivery_reference.csv)、
[`answer_only_vs_reference.csv`](answer_only_vs_reference.csv)

### C.1 参照定义

与题目、消息内容、正确性标签都无关的策略：以概率 α 发送 Full Text，以概率 1−α 发送 No Message。
从整数分子分母计算，无需新增模型调用：

```
CR(α) = α·CR_full + (1−α)·CR_none
PR(α) = α·PR_full + (1−α)·PR_none
```

α 网格 0 到 1，步长 0.05。

### C.2 在 Answer Only 的 PR 处匹配

`α = (PR_answer_only − PR_none) / (PR_full − PR_none)`

| 数据集 | Answer Only PR | 匹配 α | 参照 CR | Answer Only CR | **差值 (pp)** | 95% CI | 可匹配比例 | 标记 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| MedQA | 80.17% | 0.3231 | 30.57% | 38.79% | **+8.22** | [−2.99, +19.86] | 100.0% | OK |
| ARC-C | 63.27% | 0.5088 | 45.02% | 38.78% | **−6.25** | [−19.78, +7.86] | 100.0% | OK |
| GSM8K | 70.65% | 0.5405 | 33.61% | 39.13% | **+5.52** | [−9.85, +19.01] | 99.8% | OK |
| GPQA-D | 63.16% | 0.6786 | 55.11% | 56.14% | **+1.03** | [−14.52, +15.57] | 100.0% | OK |

每个重采样内都重新计算了 No Message / Full Text 端点、Answer Only 的 PR、匹配 α
以及相同 PR 下的 CR 差异。四个数据集均未出现分母为零的重采样；
越界重采样仅 GSM8K 有 0.2%，全部保留在计数中（见 CSV 的
`resamples_out_of_range` 列），未删除后取分位数。所有可匹配比例 ≥ 95%，
因此没有数据集被标记 `CI_UNSTABLE`。

### C.3 结论

**四个数据集的区间全部跨零。** 点估计方向不一致：MedQA +8.22 和 GSM8K +5.52 为正，
ARC-C −6.25 为负，GPQA-D +1.03 接近零。

**Answer Only 是否超越随机发送参照，本轮证据不足以判定。**
这不是"Answer Only 等同于随机发送"——没有做等价检验，不得如此表述。

### C.4 解释边界

这条参照是**随机减少消息暴露**的参照线。它不是通信能力上限，不是 ROC 曲线，
也不是在独立测试集上验证过的最优门控策略。

---

## D. 内容选择是否优于随机选择 — **BLOCKED**

表：[`span_selection_audit.csv`](span_selection_audit.csv)、
[`span_selection_comparison.csv`](span_selection_comparison.csv)

**无法回答。E2 Random / E3 Selected 在本仓库中不存在。**

检索证据：

| 检索 | 结果 |
|---|---|
| 磁盘 `*.py` `*.md` `*.json` `*.yaml` `*.sh` 搜 `E0`–`E4`、`random_span`、`selected_span`、`span_budget`、`candidate_pool` | 无命中 |
| 磁盘搜 `medqa75`、`n=75`、`75-item`、`pilot` | 无命中 |
| 全部 48 个 git commit 的**文件名** | 无命中 |
| 全部 git commit 的**内容** (`git grep` 跨全历史) | 无命中 |
| 已删除文件中唯一形似的命中 | `tests/test_state_selection.py` —— 经查为 StateBridge 的**隐状态**选择（`select_hidden_states` / `turning_point_scores`），选的是发哪 64 个 hidden state，**不是文本片段选择** |

具体缺失项：片段选择器实现、候选内容池构造、token 预算配置、消息包装格式、
历史运行记录、MedQA75 pilot 题目清单。

这与 2026-09-21 审计记录的问题 **C1（MISSING，blocking）** 一致，不是本轮新发现。

**没有临时发明替代算法**，也没有用现有 EGR 的句级删除冒充 E3——
EGR 删的是"含答案断言的整句"，没有候选池、没有 token 预算、包装格式也不同，
把它当作 E3 的复现会是错误的。

要关闭这一项，必须从仓库外导入原始 E2/E3 的选择器、候选池构造与预算配置。

---

## E. 哪些结论已获支持，哪些仍不确定

### E.1 配对区间支持的结论（仅 1 条）

1. **MedQA 上 Answer Only 的 SI 高于 No Message**：+6.90 pp，[+0.86, +12.93]，不跨零。

### E.2 点估计观察（区间不支持，只能作为描述）

2. Full Text 相对 Answer Only 的 ΔSI 很小（+1.72 / +6.63 / −3.26 / +1.32），符号不一致，四个区间全部跨零。
3. Answer Only 相对随机发送参照的 CR 差异符号不一致（+8.22 / −6.25 / +5.52 / +1.03），四个区间全部跨零。
4. ARC-C、GSM8K、GPQA-D 上 Answer Only 相对 No Message 的 ΔSI 区间跨零。

### E.3 已获支持的结构性观察（非区间推断，而是计数事实）

5. 混合组的净正确数变化极小：Full Text 相对 Answer Only 分别为 +4 / +13 / −6 / +3。
6. Full Text 与 Answer Only 的差异几乎全部是"保持原答案 → 匹配 Sender 答案"的转移，
   第三答案变动 ≤ 3 条；且该转移在 CR 组与 PR 组同时发生、方向相同。
7. 有效答案改变率严格单调：No Message < Answer Only < Full Text，四个数据集一致。

### E.4 尚未验证的解释

- "receiver 没有阅读推理" —— **未验证**，本设计不采集 receiver 处理过程。
- "匹配 Sender 答案 = 因果采纳" —— **未验证**，只是输出一致。
- "Full Text 与 Answer Only 等价" —— **不成立的表述**，没有做等价检验。

### E.5 限制结论的配置

| 限制 | 影响 |
|---|---|
| 单次采样 `seed_pair_00` | MedQA StateBridge 的第二个种子显示 SI 漂移 3.45 pp，单次采样不确定性不可忽略 |
| 样本量 92–116 / 层 | 所有区间偏宽的直接原因 |
| 仅混合方向 | Answer Only 的 SR / SCR / Acc_ret / 全集 Acc **未测**，本文件夹**全部未报告**，未填充推算值 |
| E2/E3 缺失 | D 节完全无法回答 |
| Full Text 消息含截断 | ARC-C / GSM8K / GPQA-D 的 Full Text 消息 token 最大值达 16382–16383，即部分 sender 信念本身撞到生成上限后被当作消息发出 |
| StateBridge MedQA 用 seed_pair_01 | 与三个核心条件的 `seed_pair_00` **不同**，不构成配对设计，仅作参考行导出，未计算任何对它的配对差异或区间 |
| 代码未提交 | commit `5ef9925`，另有 52 处未提交改动 |

---

## F. 剩余缺失材料及是否真的需要新增 GPU 实验

| 缺失项 | 是否需要 GPU | 说明 |
|---|---|---|
| **E2/E3 选择器与候选池** | **需要**，但先要**导入代码** | 没有选择器就无法运行。优先级是找回或重建选择器定义，而不是先申请算力 |
| **区间过宽** | **需要** | 唯一的解决办法是增加题目或增加复现种子。当前 92–116 的分层 n 决定了 ±7–15 pp 的区间宽度，任何离线分析都无法收窄 |
| **Answer Only 的 SR / SCR / Acc_ret** | **需要** | 需补跑 both_wrong 与 both_correct 方向。但注意主审计显示这两层在各通道间无区分度（SR 0.30–2.55%，SCR 91.67–100%），补跑的信息收益可能很低 |
| **等价性判定** | **不需要** | 若要写"无实质差异"，需要预先指定等价边界并做等价检验（如 TOST）。这是离线统计工作，用现有记录即可完成 |
| **receiver 处理过程证据** | **需要** | 现有记录不含 logprob 或置信度，只有 `generated_token_ids`。要支持任何关于"是否阅读"的断言必须重新前向 |
| **截断消息的敏感性分析** | **不需要** | 剔除含截断 sender 信念的配对后重算，用现有记录即可 |

**结论**：本轮的三个可回答问题中，A 与 B 的材料已经完整，C 的材料完整但结论不确定，
D 被代码缺失阻塞。**最有价值的下一步是离线的**（等价检验、截断敏感性分析），
其次才是补种子以收窄区间。

---

本轮未更新论文图片或 LaTeX，未覆盖 `figures/cr_pr_profiles.pdf`。
