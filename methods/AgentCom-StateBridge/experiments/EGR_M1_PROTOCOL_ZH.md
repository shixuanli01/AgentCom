# EGR M1：MedQA300 诊断协议

状态：M1 已完成；后续探索也固定在 MedQA300，不扩展到完整 dev/test。

范围说明：按当前实验约束，所有 EGR 结果均标记为
`MedQA300 development diagnostic`。MedQA300 已被反复检查，因此不把它写成
held-out，不在其上选择可迁移阈值，也不把多个 seed 当作新的独立题目。

本实验只开发新的通信/修订方法，不修改冻结的 None、Text、StateBridge、
LatentMAS 或 ICR 实现。所有条件复用 `seed_pair_00` 的 600 份独立
prebelief；不得重新采样 Agent A/B。

## 条件

- `claim_only`：只发送 Sender 解析后的选项标签，再执行一次冻结 ICR revision。
- `evidence_only`：发送确定性的 claim-suppressed evidence packet，不显式发送
  Sender 标签，再执行一次冻结 ICR revision。
- `egr_zero`：只在 Sender/Receiver 答案不同时，对完整选项文本计算长度归一化
  conditional log likelihood。仅当 `ME > 0` 且 `G = ME - M0 > 0` 时采用
  Sender 答案，否则保留 Receiver 答案。不执行 revision generation。

候选评分关闭 Qwen thinking，只做 teacher forcing；它不生成解释，也不使用 gold、
Sender correctness、Sender confidence 或额外 verifier。

## Evidence 限制

Evidence V1 只承诺 suppression，不承诺 answer blindness。每份 packet 必须记录：

- 过滤后 Sender 标签是否仍出现；
- Sender 完整选项文本是否仍出现；
- 显式答案声明 cue 是否仍出现；
- 原始与过滤后 token 数；
- 删除片段和删除 cue。

显式答案 cue 若仍存在，正式推理必须停止。选项文本或选项标签的残留必须如实报告，
并在解释结果时视为潜在 lexical-overlap confound。

## M1 边界

M1 是 MedQA300 diagnostic，不允许用其 gold 调可迁移阈值。按当前范围约束，
不再启动 MedQA-dev、held-out test 或跨数据集 M2-M4；新增方法只能作为同一
MedQA300 上明确标记的 exploratory checkpoint。

运行：

```bash
bash scripts/run_egr_m1_medqa300.sh \
  artifacts/icr_medqa300/seed_pair_00 \
  artifacts/egr/medqa300_diagnostic/seed_pair_00 0
```
