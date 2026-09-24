# AgentCom — 正式审计结果

本文件夹是**审计结果的唯一权威来源**。它回答的问题是：

> 当一个 agent 收到另一个 agent 的信息后，它能否有选择地纠正错误、保留正确答案？

这里只放**审计**：ICR 协议本身、已发表通信通道的行为测量、对照参考、以及使这些测量可信所需的完整性检查。

我们自己提出的通信方法（EGR、Hybrid、CR-DNC、Adjudication）**不在这里**，全部移到
[`../parked_methods/`](../parked_methods/)，暂时不用于论文。

整理时间 2026-09-23。

---

## 怎么读这个文件夹

| 文件 | 内容 |
|---|---|
| [`01_SCOPE.md`](01_SCOPE.md) | 口径：ICR 协议、六方向、四个分层、CR/PR/SI/SR/SCR 定义、五种准确率口径、计数恒等式 |
| [`02_MAIN_RESULTS.md`](02_MAIN_RESULTS.md) | 主结果：6 个通道 × 5 个数据集 |
| [`03_CONTROLS.md`](03_CONTROLS.md) | 对照实验：Keep Initial、接收策略 V4 消融、种子复现 |
| [`04_DIAGNOSTICS.md`](04_DIAGNOSTICS.md) | 解释性诊断：SI 究竟测的是什么、有没有可用的正确性信号 |
| [`05_INTEGRITY.md`](05_INTEGRITY.md) | 完整性：截断污染、排除与复用、恒等式校验、敏感性 |
| [`AUDIT_ISSUES.md`](AUDIT_ISSUES.md) | 已知问题清单，按 A/B/C 严重度分级 |
| [`tables/`](tables/) | 全部 CSV |
| [`message_audit_v1/`](message_audit_v1/) | Section 5.3 **第一轮**（MedQA + GPQA-D）。**已被取代**，见其 `SUPERSEDED.md`；仅 `cost.csv` 与 `message_audit.csv` 仍是唯一来源 |
| [`../paper_materials_section53/`](../paper_materials_section53/) | **Section 5.3 权威版**：消息内容审计，四个数据集。见 `REPORT_SECTION53_ZH.md` |
| [`../paper_materials_section54/`](../paper_materials_section54/) | **Section 5.4**：接收策略敏感性，MedQA + GPQA-D 的 V3/V4。见 `REPORT_SECTION54_ZH.md` |

`_legacy_MAIN_TEXT_EVIDENCE.md` 和 `_legacy_APPENDIX_DETAILS.md` 是 2026-09-21 生成的旧材料包，
数值仍然有效（核心四通道部分）。它们的 **§5.3 / §5.4 关于 EGR 的章节已作废**，见 `../parked_methods/`。
`manifest_legacy.json` 同理只覆盖核心四通道。

---

## 一句话结论

1. **SI 不测通信质量，它测接收方的判别力。** `SI − 50 ≡ (π_CR − π_PR)/2`，20 个非代码条件上 r = 0.9879。
2. **没有任何通道在四个数据集上稳定取胜。** 每个通道都有它赢的数据集和输的数据集。
3. **推理文本稳定地把采纳率翻倍，但对判别力没有一致方向的作用。** Answer Only 在 GSM8K 上判别力是 Full Text 的 3.3 倍。
4. **不存在可用的、无需 ground truth 的正确性信号**（见 `04_DIAGNOSTICS.md`）。这解释了为什么 (2) 成立。

---

## MedQA StateBridge 的数据来源（2026-09-23 变更）

全文 MedQA StateBridge **一律采用 `replication_id = seed_pair_01`**
（`runs/medqa/replication_01`，CR 63/116、PR 83/116、SI 62.93%、Acc_ret 200/720 = 27.78%）。

`seed_pair_00` 的 MedQA StateBridge 结果（66.38% 等）**已从所有报告结果中撤下**，
原始记录仍在磁盘上仅作溯源，不得再引用。

**由此产生的一处已知代价**：V4 接收策略消融的 V4 侧仍是 `seed_pair_00`，
MedQA StateBridge 的 V3/V4 比较因此是**跨种子**的，不再控制修订采样种子。
这一取舍是明确接受的，凡引用该比较处均已标注。

其余数据集的 StateBridge 与所有其它通道不受影响。

---

## 三条硬性口径规则

写论文时必须遵守，表格里已经按这三条标注：

1. **`NOT_MEASURED` 不可填充。** Answer Only 只跑了混合方向，它的 SR / SCR / Acc_ret **底层记录不存在**，
   不是"未知但可估"。不要用推算值补。
2. **全集准确率是 `ESTIMATED`。** `tables/full_set_accuracy_estimated.csv` 里的数值是外推的，
   引用时必须保留 estimated 标记。
3. **HumanEval+ 与其余四个数据集不可比。** 它的分层 n = 28，截断污染占 CR/PR 子集的 32.14%，
   且"采纳 = 答案相同"对自由形式代码无定义（`tables/si_discrimination_identity.csv` 标为 `UNDEFINED`）。

---

## 原始数据在哪

原始记录 4.6 GB，**没有移动**，仍在原处：

| 内容 | 路径（相对 `methods/AgentCom-StateBridge/`） |
|---|---|
| 核心四通道 × 5 数据集 | `artifacts/icr_v3/<dataset>_full_seed42/revisions/merged.jsonl` |
| Answer Only | `runs/medqa/cr_dnc_v1/icr_root`、`runs/{gpqa,arc_challenge,gsm8k}/hybrid_root`，condition `true_answer` |
| 接收策略 V4 | `runs/medqa/v4_verify` |
| 种子复现 | `runs/medqa/replication_01` |
| Answer Only 独立报告 | `runs/answer_only_baseline.md` |

`artifacts/icr_v3_superseded/` 是被取代的旧版本，**不要引用**。
