# EGR V1 方法冻结说明

日期：2026-09-18

状态：方法定义冻结，用于 MedQA300 和跨数据集第一波诊断。后续若改变
evidence 过滤规则、裁决 prompt、调用拓扑或 fallback，必须使用新的方法版本，
不得覆盖本版本结果。

## 1. 方法定位

Evidence-Grounded Revision（EGR）研究的是接收者如何使用另一个推理过程提供的
证据。它包含文本 evidence message 和 receiver-side adjudication，不是新的 latent
transport。因此论文和报告必须区分两个问题：

1. transport comparison：固定 directional revision，比较 Text、StateBridge、
   LatentMAS 等消息载体；
2. receiver-policy comparison：固定 evidence，比较 ordinary revision、Contrast、
   Permutation 等信息整合规则。

EGR 与 latent baseline 的最终准确率可以作为系统级诊断并列展示，但不能解释成
纯通信编码的公平胜负。

## 2. 冻结输入

每题先由同一个 Qwen3-4B 使用两个确定性派生 seed 独立生成：

```text
Q + seed_A -> reasoning_A, answer_A
Q + seed_B -> reasoning_B, answer_B
```

两条轨迹共享模型、题目、solver prompt、temperature、top-p、parser 和 token 上限。
A/B 只表示独立样本，不表示固定角色。评估保留 `A_to_B` 和 `B_to_A` 两个方向。

## 3. Evidence packet

Evidence packet 由确定性规则直接过滤 Sender 的原始 reasoning，不调用第二个 LLM，
也不生成摘要。过滤器删除显式答案声明，包括 `final answer`、`answer is`、
`choose option X`、独立选项标签和 `\boxed{...}`。输出同时保存原文、删除 span、
删除 cue、token 数和残留泄漏诊断。

它只保证 provenance 和可复现性，不保证 factual correctness。选项文本可作为正常
推理内容保留，因此正确名称是 `claim-suppressed reasoning`，不得称为
`answer-blind evidence` 或 `verified facts`。

MedQA300 seed 00 的 600 个 packet 中：显式 answer cue 为 0，Sender 答案标签残留
203 个，完整答案文本残留 475 个，token retention 为 96.52%。这说明当前步骤主要
删除答案格式，没有完成事实级抽取。

## 4. EGR Contrast

当 A/B 答案可解析且不一致时，运行一次匿名、对称裁决：

```text
Original question Q
Evidence set 1 = filtered reasoning_A
Evidence set 2 = filtered reasoning_B
```

裁决器被要求核对决定性事实、计算或因果关系，不投票、不信任来源身份，并输出
一个最终答案。裁决可选择 A、B 的答案，也可重新求解得到第三个答案。

冻结决策：

| 输入状态 | 决策 |
|---|---|
| A/B 答案相同 | 零调用，保留共同答案 |
| 一方不可解析 | 零调用，采用可解析答案 |
| 双方均不可解析 | 保留不可解析状态 |
| 双方可解析且不同 | 一次 Contrast generation |

同一个 item-level verdict 被写入两个 directional record。因此两个方向不是独立
生成，统计不应把它们当成独立样本。

## 5. Permutation gate

对每个可解析分歧，保持模型、seed、decoding、问题和 evidence 内容不变，仅交换
Evidence Set 1/2 的顺序，再运行一次裁决：

```text
V_AB == V_BA -> 接受 V_AB
V_AB != V_BA -> 各方向回退到 Receiver prebelief
```

Permutation 检查 order robustness，不检查事实正确性。两个顺序可以稳定地产生
同一个错误答案。它不会选择 swapped verdict，也不是第二票。

## 6. 与 ICR baseline 的边界

`none`、`true_text`、`true_statebridge` 和 `true_latentmas` 使用相同的 directional
revision prompt：Receiver 明确看到自己的 prior reasoning/answer，再接收一个
Sender message，并独立生成一次 revised answer。

EGR Contrast 改为匿名双证据裁决，只在分歧题调用一次，并把 verdict 复制到两个
方向。因此以下变量并未同时固定：Receiver 身份、prompt、调用数、agreement
处理和方向独立性。EGR 与 ICR baseline 的比较必须标记为 `system-level diagnostic`。

## 7. 固定报告要求

每次运行至少报告：

- exact correct/total、Accuracy；
- Correction Rate（CR）和 Prior Retention（PR）及其分母；
- rescue、destruction、invalid output；
- 与 no-message、Full Text 的 paired gained/lost 和 exact McNemar p；
- prebelief accuracy、agreement 数和 oracle-2；
- 数据选择 seed、代码快照和结果 fingerprint；
- smoke、partial、resumed、full/frozen 状态。

不得用最终准确率单独声称通信质量提升，也不得把 EGR 的 receiver-policy 增益归因
于 latent transport。

## 8. 当前方法结论

EGR V1 已证明对称 adjudication 能改变 correction/preservation trade-off，并且
order instability 是可测量的脆弱性信号。它尚未证明 evidence factuality、稳定优于
Full Text，或提供了更好的 latent communication。当前主要瓶颈是 evidence
reliability estimation，而不是消息容量。
