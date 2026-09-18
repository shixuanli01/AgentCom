# EGR 在 MedQA300 上的阶段性总报告

日期：2026-09-16

状态：完整 MedQA300 诊断已完成。后续评估固定使用这 300 题，不扩展到完整
dev/test。本文所有结果都是 development diagnostic，不是 held-out 结论。

## 1. 实验问题

冻结 ICR 的核心矛盾是：Full Text 纠错能力强，但也容易把原本正确的 Receiver
带错；StateBridge 的覆盖更强，保护率更差。EGR 测试以下原则：

> 外部证据必须先证明自己有用，外部结论不能自动获得影响力。

所有实验复用 Qwen3-4B、缓存的独立 A/B prebelief、原题顺序、解析器和随机种子。
冻结的 None、Full Text、StateBridge、LatentMAS 实现均未修改。

指标含义：CR 是 Receiver 原错、Sender 对时的纠错率；PR 是 Receiver 对、Sender
错时的保护率；SI=(CR+PR)/2；FCS/FWS 分别是 Sender 对/错时跟随 Sender 的比率；
FS=FCS-FWS。统计采用 10,000 次 item-cluster bootstrap，同时保留同一题的两个
方向和所有 seed。

## 2. M1：信息拆解与零阈值门控

M1 使用 seed_pair_00，共 300 题、600 个方向样本。

| 条件 | Acc (%) | CR (%) | PR (%) | SI (%) | FCS (%) | FWS (%) | FS (pp) | Rescue/Destroy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 72.50 | 5.71 | 97.14 | 51.43 | 5.71 | 2.86 | 2.86 | 3/1 |
| Full Text | 74.33 | 91.43 | 42.86 | 67.14 | 91.43 | 57.14 | 34.29 | 33/20 |
| Claim Only | 73.67 | 34.29 | 88.57 | 61.43 | 34.29 | 11.43 | 22.86 | 13/4 |
| Evidence Only | 73.50 | 85.71 | 34.29 | 60.00 | 85.71 | 65.71 | 20.00 | 31/23 |
| EGR-zero | 71.00 | 25.71 | 54.29 | 40.00 | 25.71 | 45.71 | -20.00 | 9/16 |
| StateBridge | 71.67 | 82.86 | 5.71 | 44.29 | 82.86 | 94.29 | -11.43 | 30/33 |
| LatentMAS | 71.83 | 31.43 | 62.86 | 47.14 | 31.43 | 37.14 | -5.71 | 11/13 |

M1 的直接结论：

1. Claim Only 不像 StateBridge。它影响较弱、PR 较高，因此不能把 StateBridge 的
   覆盖行为简单归因于“看到了 Sender 结论”。
2. Evidence Only 没有提高 PR，反而相对 Full Text 降低 8.57 pp；删除显式结论并不
   会自动消除错误证据的说服力。
3. EGR-zero 显著失败。相对 Full Text，Acc -3.33 pp，95% CI
   [-5.33, -1.50]；SI -27.14 pp，95% CI [-41.46, -12.16]。
4. 五折 cross-fit threshold 也失败：Acc 71.83、CR 5.71、PR 88.57、SI 47.14。
   它主要学会了拒绝通信，没有学会识别正确证据。

证据过滤后显式答案 cue 为 0/600，但答案标签仍出现 203/600，完整答案文本出现
475/600，token retention 为 96.52%。因此该表示只能称 claim-suppressed，不能称
answer-blind。

根因诊断：likelihood shift 测量的是一段证据对模型有多“自洽/有说服力”，不是它
是否为真。错误 Sender 的证据同样会稳定提高错误候选的似然。

## 3. 对称证据裁决

`EGR-Contrast v1` 只在 A/B 初始答案不一致时运行一次匿名裁决；一致时零调用。
裁决器看到原题和两份 claim-suppressed evidence，要求核对事实和因果关系，不按
多数投票。三个 seed 均完整运行，每个条件共 1,800 个方向记录。

| 条件 | Acc (%) | CR (%) | PR (%) | SI (%) | FCS (%) | FWS (%) | FS (pp) | Rescue/Destroy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 71.50 | 2.97 | 97.03 | 50.00 | 3.00 | 3.00 | 0.00 | 9/5 |
| Full Text | 72.72 | 87.13 | 37.62 | 62.38 | 87.00 | 63.00 | 24.00 | 89/63 |
| StateBridge | 71.06 | 85.15 | 11.88 | 48.51 | 85.00 | 89.00 | -4.00 | 87/91 |
| EGR-Contrast | 72.78 | 63.37 | 63.37 | 63.37 | 63.00 | 37.00 | 26.00 | 64/37 |

相对 Full Text 的 paired bootstrap：

- Accuracy +0.06 pp，95% CI [-0.83, +0.94]；
- CR -23.76 pp，95% CI [-32.71, -15.46]；
- PR +25.74 pp，95% CI [+15.29, +36.78]；
- SI +0.99 pp，95% CI [-6.96, +8.67]；
- FS +2.00 pp，95% CI [-13.98, +17.53]。

因此 EGR-Contrast 明确改变了行为：它显著减少破坏，但以几乎对称的纠错损失为
代价。Accuracy 只多 1/1,800 个方向记录；Accuracy 与 SI 均未显示显著优势，
不能宣称稳定优于 Full Text。

分 seed 的 Accuracy/SI：

| Seed | Full Text Acc/SI (%) | EGR-Contrast Acc/SI (%) |
|---|---:|---:|
| seed_pair_00 | 74.33 / 67.14 | 75.00 / 74.29 |
| seed_pair_01 | 71.33 / 59.38 | 71.67 / 62.50 |
| seed_pair_02 | 72.50 / 60.29 | 71.67 / 52.94 |

机制审计：三个 seed 共裁决 133 个分歧，其中 100 个是“一边正确、一边错误”。
裁决器选对 63/100；正确来源位于 A 时为 32/52，位于 B 时为 31/48，没有明显的
位置偏好。它在 33 个“两边都错”的分歧上从未生成第三个正确答案。

运行成本：47/43/43 次生成，共 133 次；三次 wall clock 合计约 59.0 分钟。由于
Full Text 对每个方向都生成，二者不是完全相同的调用拓扑，结果不能解释成纯传输
通道的公平胜负。

## 4. 后续最小修补

以下版本只在 seed_pair_00 完整跑完；它们均为失败诊断，因此没有扩到另外两个
seed。

| 版本 | 设计 | Acc (%) | CR (%) | PR (%) | SI (%) | 调用 | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| Falsify v1 | 把初步裁决交给同模型反证 | 75.00 | 74.29 | 74.29 | 74.29 | 47 | 仅改 1 题，仍错；完全锚定 |
| Ledger v1 | 先做无结论事实账本，再独立决策 | 74.33 | 68.57 | 68.57 | 68.57 | 94 | 0 个错转对，2 个对转错 |
| Hypothesis v1 | 证据显式绑定待证伪候选 | 73.67 | 62.86 | 62.86 | 62.86 | 47 | 1 个错转对，5 个对转错 |

双向 Text 收敛后再回退 EGR-Contrast 的离线组合也只有 pooled Acc 73.00、SI
65.35；高于单阶段 Acc 0.22 pp，但仍不稳定，且不是干净的单一通信条件。

最后测试了零阈值 `EGR-Permutation Gate`：交换两份匿名证据的展示顺序，并复用
相同随机种子重新裁决；仅当交换前后结论一致时才采纳，否则各方向保留 Receiver
原答案。三个 seed 的 pooled 结果如下：

| 条件 | Acc (%) | CR (%) | PR (%) | SI (%) | FCS (%) | FWS (%) | FS (pp) | Rescue/Destroy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Text | 72.72 | 87.13 | 37.62 | 62.38 | 87.00 | 63.00 | 24.00 | 89/63 |
| EGR-Contrast | 72.78 | 63.37 | 63.37 | 63.37 | 63.00 | 37.00 | 26.00 | 64/37 |
| EGR-Permutation | 72.72 | 52.48 | 73.27 | 62.87 | 52.00 | 27.00 | 25.00 | 53/27 |

相对 EGR-Contrast，Permutation 的 Acc -0.06 pp，95% CI [-0.50, +0.39]；
SI -0.50 pp，95% CI [-4.59, +3.54]。133 个分歧中有 35 个顺序不稳定；其中
21 个是一边正确、一边错误。门控拦下 10 个错误裁决，也拒绝 11 个正确裁决。
因此 order instability 是有效的脆弱性信号，但不是可靠的错误信号。

## 5. 当前能够与不能够证明的结论

能够证明：

1. Full Text 的高纠错和低保护确实是同一“高影响”机制的两面。
2. 只删除结论、只做似然门控、或让同一模型再次自检，都不足以识别证据真假。
3. 对称裁决可把 CR/PR 从 87.13/37.62 平衡到 63.37/63.37，并显著提高 PR。
4. EGR 当前真正的瓶颈是 evidence reliability estimation，而不是消息容量。
5. 展示顺序不变性也不能单独充当 correctness estimator；它提高 PR 时仍会对称
   损失 CR。

不能证明：

1. 不能证明 EGR 在最终准确率或 SI 上稳定优于 Full Text；pooled CI 跨过 0。
2. 不能把三 seed 当作 900 道独立题；它们仍是同一 MedQA300 的重复采样。
3. 不能证明收益来自更好的“传输编码”；EGR-Contrast 同时改变了交互拓扑和调用数。
4. 不能称 evidence packet 为 answer-blind，也不能称本结果为 held-out generalization。

## 6. 当前决策

保留 `EGR-Contrast v1` 作为目前最好的选择性通信候选，但状态应写为：

```yaml
MedQA300 diagnostic:
  evidence_decomposition: FAIL
  likelihood_gate: FAIL
  preservation_gain: PASS
  stable_accuracy_gain_vs_text: FAIL
  stable_SI_gain_vs_text: FAIL
  overall_method_gate: FAIL
```

不继续在 MedQA300 上微调阈值或 prompt，否则只会增加同一 300 题上的选择偏差。
下一次方法变化若仍限定 MedQA300，应作为新的 exploratory checkpoint 明确记录，
不能升级为泛化证据。

## 7. 产物

- M1：`artifacts/egr/medqa300_diagnostic/seed_pair_00/`
- Cross-fit：`artifacts/egr/medqa300_diagnostic/seed_pair_00/gate_crossfit/`
- 对称 likelihood duel：`artifacts/egr/medqa300_dual_v1/seed_pair_00/`
- EGR-Contrast 三 seed：`artifacts/egr/medqa300_contrast_v1/seed_pair_00..02/`
- 三 seed pooled：`artifacts/egr/medqa300_contrast_v1/pooled/`
- Permutation Gate 三 seed与 pooled：`artifacts/egr/medqa300_permutation_v1/`
- Falsify/Ledger/Hypothesis：`artifacts/egr/medqa300_{falsify,ledger,hypothesis}_v1/`

关键机器可读文件是 pooled `summary.json`、`metrics.csv`、
`verdict_audit.csv` 和 `paired_cases.parquet`。
