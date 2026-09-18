# EGR V1 跨数据集阶段报告

日期：2026-09-18

状态：MedQA300、ARC-Challenge random-300、ARC disagreement-70 和 GSM8K
random-300 已完成。ARC/GSM8K 结果已生成冻结指纹。MBPP+ random-300 正在运行，
HumanEval+ 已排队，本文不提前填写代码任务结果。

## 1. 固定设置

- 模型：`Qwen/Qwen3-4B`
- replication：`seed_pair_00`（MedQA 另有三 seed pooled 诊断）
- global/selection seed：42
- decoding：temperature 0.6，top-p 0.95
- 每题两个独立 prebelief、两个 directional records
- baseline：No-message、Full Text、StateBridge、LatentMAS
- receiver policy：EGR Contrast、EGR Permutation
- 报告单位：directional records；EGR 的两个方向共享 item-level verdict

方法定义和比较边界见
[EGR V1 方法冻结说明](EGR_V1_METHOD_FREEZE_ZH.md)。

## 2. MedQA300 三 seed

三个 seed 共 1,800 个方向记录。LatentMAS 没有相同的三 seed 完整矩阵，因此不在
本表中混合报告。

| Condition | Accuracy | CR | PR | Rescue/Destroy |
|---|---:|---:|---:|---:|
| No-message | 71.50% | 2.97% | 97.03% | 9/5 |
| Full Text | 72.72% | 87.13% | 37.62% | 89/63 |
| StateBridge | 71.06% | 85.15% | 11.88% | 87/91 |
| EGR Contrast | **72.78%** | 63.37% | 63.37% | 64/37 |
| EGR Permutation | 72.72% | 52.48% | 73.27% | 53/27 |

EGR Contrast 相对 Full Text 仅 `+0.06 pp`，95% CI `[-0.83,+0.94]`；Permutation
相对 Contrast `-0.06 pp`，95% CI `[-0.50,+0.39]`。EGR 明显改变了 CR/PR 平衡，
但没有稳定 Accuracy 或 SI 优势。

## 3. ARC-Challenge random-300

Prebelief 为 `530/600 = 88.33%`，oracle-2 为 `272/300`，A/B 答案一致
`284/300`。EGR 只有 16 题触发真实裁决。

| Condition | Correct | Accuracy | CR | PR | Rescue/Destroy |
|---|---:|---:|---:|---:|---:|
| No-message | 561/600 | 93.50% | 57.14% | 100.00% | 31/0 |
| Full Text | **562/600** | **93.67%** | 92.86% | 71.43% | 36/4 |
| StateBridge | 558/600 | 93.00% | 100.00% | 78.57% | 31/3 |
| LatentMAS | **562/600** | **93.67%** | 85.71% | 100.00% | 32/0 |
| EGR Contrast | 543/600 | 90.50% | 92.86% | 100.00% | 13/0 |
| EGR Permutation | 542/600 | 90.33% | 85.71% | 100.00% | 12/0 |

EGR Contrast 相对 no-message 为 `-3.00 pp`，gained/lost `5/23`，exact
`p=0.000912`。这个差距不能解释为 evidence adjudication 本身失败：EGR 在 284 个
agreement item 上零调用，而 no-message 对两个方向都重新生成，能够修复 shared
wrong。它首先暴露的是调用拓扑不匹配。

冻结 fingerprint：`2ddf3817dbb21522b9dc38c8011816dbe6d58ad0f74c35889292173bcc8c3b00`。

## 4. ARC-Challenge disagreement-70

该集合由完整 cache 中 A/B 初始答案不同的全部 70 题组成，只用于机制诊断，不是
随机 benchmark accuracy。Prebelief 为 `54/140 = 38.57%`，oracle-2 为 `54/70`。

| Condition | Correct | Accuracy | CR | PR | Rescue/Destroy |
|---|---:|---:|---:|---:|---:|
| No-message | 92/140 | 65.71% | 62.96% | 98.15% | 39/1 |
| Full Text | 94/140 | 67.14% | 98.15% | 72.22% | 55/15 |
| StateBridge | 93/140 | 66.43% | 96.30% | 72.22% | 54/15 |
| LatentMAS | 92/140 | 65.71% | 77.78% | 90.74% | 43/5 |
| EGR Contrast | **98/140** | **70.00%** | 85.19% | 96.30% | 46/2 |
| EGR Permutation | 96/140 | 68.57% | 79.63% | 98.15% | 43/1 |

EGR Contrast 相对 no-message 为 `+4.29 pp`，但 gained/lost `13/7`，exact
`p=0.263`；相对 Full Text 为 `+2.86 pp`，`p=0.541`。方向符合预期但样本不足，
不能声称显著优势。

冻结 fingerprint：`052706e9ad17b71dd3520f708e2c0b9ecf8f711d1f0a41ed8de3eafd7a06ce23`。

## 5. GSM8K random-300

Prebelief 为 `535/600 = 89.17%`，oracle-2 为 `275/300`，A/B 答案一致
`270/300`。

| Condition | Correct | Accuracy | CR | PR | Rescue/Destroy |
|---|---:|---:|---:|---:|---:|
| No-message | 556/600 | 92.67% | 73.33% | 100.00% | 21/0 |
| Full Text | 559/600 | 93.17% | 100.00% | 86.67% | 26/2 |
| StateBridge | 553/600 | 92.17% | 93.33% | 73.33% | 22/4 |
| LatentMAS | 556/600 | 92.67% | 80.00% | 86.67% | 23/2 |
| EGR Contrast | **560/600** | **93.33%** | 93.33% | 93.33% | 26/1 |
| EGR Permutation | 558/600 | 93.00% | 86.67% | 100.00% | 23/0 |

EGR Contrast 相对 no-message 为 `+0.67 pp`，gained/lost `6/2`，exact
`p=0.289`；相对 Full Text 只多 `1/600`，gained/lost `5/4`，`p=1.0`。所有方法
差距都很小。CR/PR 分母各只有 15，一个样本会改变 6.67 pp。

冻结 fingerprint：`ac1c900afc773113f395a626370a05e83db5dae6213ab23e3f2260bf1809fc50`。

## 6. 当前跨数据集判断

1. No-message revision 在 ARC/GSM8K 已能带来明显增益，额外推理是强控制条件。
2. Full Text 通常有较高 CR，但可能牺牲 PR；StateBridge 在当前 ICR Receiver 中也
   显示相似的高影响倾向。
3. LatentMAS 尚未显示稳定优于 no-message 或 Full Text。
4. EGR Contrast 在分歧子集上更平衡，但 random ARC 因零调用 agreement 设计而明显
   落后；它与 directional baseline 不是纯 transport 的公平比较。
5. Permutation 稳定地减少 destruction，也会拒绝一部分正确裁决，尚未提高最终
   accuracy。
6. 当前没有方法取得跨数据集统计显著、稳定的优势。最可靠的发现是 communication
   utility 必须拆成 correction、preservation、来源匹配和额外计算，而不能只看最终分数。

## 7. 未完成项

- MBPP+ random-300：运行中；
- HumanEval+ full-164：排队中；
- 代码任务不运行字符串相等的 permutation gate；
- 第一波代码任务完成前，不形成 EGR 跨领域成功结论。
