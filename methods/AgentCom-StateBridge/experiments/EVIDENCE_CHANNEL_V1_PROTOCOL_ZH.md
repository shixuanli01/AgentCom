# Evidence Channel V1 实验协议

日期：2026-09-19

状态：实现阶段；方法与评估变量冻结后只追加到同一个 ICR-V3 baseline artifact。

## 1. 问题

在不改变 Receiver prompt、pipeline、parser、seed 或调用次数的情况下，将 Sender 的
完整文本 reasoning 换成 claim-suppressed evidence，是否能提高通信的纠错/保护
平衡？

该实验测试的是消息内容选择，不是新的 latent transport，也不是 EGR Contrast。

## 2. Message 构造

输入是 Sender 的 V3 prebelief：`reasoning_text`、`parsed_answer` 和题目选项表。
确定性过滤器逐行删除显式答案承诺：

- `final answer`、`answer is`、`correct answer`；
- `choose option X` 等选择声明；
- 独立的 A-D 答案行；
- `\boxed{...}` 和明确带答案的 conclusion 行。

其他原文逐字保留，不调用额外 LLM、不总结、不改写。若过滤后仍检测到显式答案
cue，运行 hard stop。每条 message 记录 payload SHA256、过滤前后 token 数、删除
span 数，以及答案标签/答案文本残留诊断。

Evidence V1 是 `claim-suppressed reasoning`，不是 `verified facts` 或
`answer-blind evidence`。它不判断医学事实真假，选项语义也可能保留。

## 3. Receiver 接口

条件名为 `true_evidence`。A→B 时只发送 A 的 evidence packet，B 仍看到自己的
prior reasoning/answer，并使用冻结的 `icr_v3_mid_injection` revision prompt。

`true_evidence` 与 `true_text` 的可见 prompt wrapper、message slot、revision seed
和单次 Receiver generation 完全相同；唯一变化是 external message body。

Phase 2 对 `both-correct` item 使用冻结的 `item_id % 10 == 0` 抽样规则。其余
`both-correct` pair 不进入任何 communication condition；这节约主要推理开销，同时
保留 10% 样本用于估计 SCR，而不是把“路由器跳过后保持正确”误报为通信方法的保护
能力。所有对照方法必须使用相同 pair 集合。

## 4. 数据与对照

- benchmark：MedQA300；
- model：Qwen3-4B；
- replication：`seed_pair_00`；
- directional records：600；
- baseline：同一个 artifact 中的 `none`、`true_text`、`true_statebridge`、
  `true_latentmas`；
- method：只追加 `true_evidence`，不重新生成 prebelief 或 baseline。

Seed 00 是依据早期 EGR 结果选出的表现较好 seed，因此本轮是 exploratory method
check，不是无偏泛化验证。任何正结果都必须在未用于选择方法的 seed/benchmark
复验。

## 5. 报告

主表报告 Accuracy、CR、PR、SI、SRA、FCS、FWS、FollowSelectivity、rescue 和
destruction。Evidence 分别与 no-message、Full Text、StateBridge、LatentMAS 做
同方向 paired comparison，并报告 item-cluster bootstrap 95% CI、gained/lost。

首要机制判断：

1. Evidence 相对 Full Text 是否减少 destruction、提高 PR；
2. 上述保护是否以过大的 CR 损失为代价；
3. Evidence 相对 no-message 是否具有正 communication effect；
4. message 中答案标签/答案文本残留是否解释 observed gain。

最终准确率不能单独证明 evidence communication 更好。

## 6. 增量运行

Baseline 完成后，在同一个 artifact root 上运行：

```bash
cd methods/AgentCom-StateBridge
CUDA_DEVICES="0 1 2 3" WORKERS_PER_GPU=1 \
  bash scripts/run_v3_evidence_channel.sh \
  artifacts/icr_v3/medqa_full_seed42
```

Launcher 会检查 V3 protocol、prompt version、MedQA300、`seed_pair_00` 和四个已完成
baseline 条件；任何一项不匹配都会拒绝启动。

## 7. 冻结 prebelief 的快速方法检查

在 baseline artifact 尚未同步到本机时，可以复用既有的 600 条 A/B prebelief，只对
Receiver 使用最新 `icr_v3_mid_injection` prompt：

```bash
PREBELIEF_SOURCE_ROOT=artifacts/icr_medqa300/seed_pair_00 \
  WORKERS_PER_GPU=2 bash scripts/run_v3_evidence_only.sh \
  artifacts/icr_v3/medqa300_legacy_prebelief_v3prompt_evidence_v1
```

该设置必须标记为 `ICR-V3-RECEIVER-FROZEN-PREBELIEF-V1`，而不能称为纯 ICR-V3。
它不生成 Phase 1，只复制并哈希冻结 prebelief，然后执行 600 条 `true_evidence`
revision。它可以衡量 Evidence 对这批固定轨迹的修订效果；要进行严格 channel
比较，所有 baseline 必须复用完全相同的 prebelief，并使用同一个 V3 Receiver
prompt、parser、revision seed 与 decoding 设置。
