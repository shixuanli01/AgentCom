# 跨数据集第一波评估协议

## 目的

停止继续扩充 MedQA 随机种子，使用单一冻结 replication
`seed_pair_00` 检查通信现象和 EGR 是否能迁移到不同知识领域。

## 数据集

1. ARC-Challenge：固定 seed 42 无放回随机抽取 300/1172 题。
2. GSM8K：固定 seed 42 无放回随机抽取 300/1319 题。
3. MBPP+：固定 seed 42 无放回随机抽取 300/378 题。
4. HumanEval+：完整 164 题。

ARC-C 另报告一个 label-free disagreement audit：从完整 prebelief cache 中选择两名
agent 初始答案不同的全部 70 题。该集合只用于 correction/destruction 机制诊断，
不得与随机 300 题合并计算 benchmark accuracy。

固定答案任务先运行：ARC-C 使用选择题精确匹配，GSM8K 使用最终数值精确匹配。
随后运行开放答案代码任务：MBPP+ 和 HumanEval+ 使用代码提取与测试执行评分。
所有数据集只运行 `seed_pair_00`，不根据测试准确率搜索 seed。

## 第一波条件

- `none`
- `true_text`
- `true_statebridge`
- `true_latentmas`
- `egr_contrast`
- `egr_permutation_gate`

`egr_permutation_gate` 只用于 ARC-C/GSM8K。对代码任务，源码字符串不相同不代表
程序行为不同；在没有一个不读取隐藏测试的行为等价判据前，MBPP+/HumanEval+
只报告 `egr_contrast`，不报告伪造的 permutation 稳定性。

这一波优先比较主方法，不增加新的随机种子，也暂不运行 self/other 控制。若某个
新数据集出现有意义的主方法差异，再为该数据集补齐 self/other 因果控制。

## 冻结项

- 模型：`Qwen/Qwen3-4B`
- replication：`seed_pair_00`
- 全局 seed：42
- temperature：0.6
- top-p：0.95
- ARC-C max-new-tokens：2048
- GSM8K max-new-tokens：2048
- MBPP+/HumanEval+ max-new-tokens：4096
- 两个独立 prebelief、两个方向、相同 revision seed
- StateBridge：post-think last-64 + Procrustes prefix
- LatentMAS：10 个 latent steps
- EGR：claim-suppressed evidence、Contrast 和 order-permutation gate

不得根据正式准确率调整 prompt、token 上限、通信预算或选择规则。

## 启动

```bash
cd methods/AgentCom-StateBridge

bash scripts/run_cross_benchmark_first_wave.sh arc_challenge
BENCHMARK_SAMPLE_SIZE=300 bash scripts/run_cross_benchmark_first_wave.sh gsm8k
BENCHMARK_SAMPLE_SIZE=300 bash scripts/run_cross_benchmark_first_wave.sh mbppplus
bash scripts/run_cross_benchmark_first_wave.sh humanevalplus
```

脚本可恢复已完成的逐题 JSON。正式报告位于：

```text
artifacts/cross_benchmark/<run>/cross_benchmark_analysis/report.md
artifacts/cross_benchmark/<run>/cross_benchmark_analysis/report.json
```

报告给出精确正确数、CR、PR、rescue、destruction，以及相对 no-message 和 text
的配对增减与 exact McNemar p 值。最终准确率不能单独证明通信质量。
