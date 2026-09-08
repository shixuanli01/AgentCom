# Multi-Path StateBridge Refiner 因果冒烟验证

## 结论

```yaml
Refiner causal smoke check: PASS
Scope: 5 complete Oracle@5-miss samples
Full 32-sample statistical evaluation: NOT RUN (stopped by design)
```

该验证支持如下机制结论：在完全固定、贪心解码的同一个 Judger 下，仅替换
Refiner 传入的 hidden-state prefix，就会系统性地改变最终答案。原实验中的正确
分支也能被确定性重放。因此，已观察到的多路径 Oracle 增益不能仅用最终 Judger
的随机采样解释，Refiner 输出确实携带了会影响答案的路径特异信息。

## 协议

- 数据：MedQA 多路径运行中 majority vote 错误但 Oracle@5 正确的样本。
- 模型：`Qwen/Qwen3-4B`。
- Judger：同一 prompt、同一生成配置、贪心解码。
- 实验条件：5 个已保存 Refiner prefix、无 prefix、64-token 全零 prefix。
- 唯一主要干预变量：输入 Judger 的 Refiner hidden-state prefix。

## 结果

| 指标 | 结果 |
|---|---:|
| 完整样本数 | 5 |
| Refiner prefix 导致分支答案不同 | 5/5 |
| 至少一个分支区别于 no-prefix | 5/5 |
| 至少一个分支区别于 zero-prefix | 5/5 |
| 原正确路径确定性保留 | 7/7 (100%) |
| 原错误路径被固定 Judger 修复 | 0/18 (0%) |
| 固定 Oracle@5 | 5/5 (100%) |
| 平均两两答案分歧率 | 56% |
| 解析失败 | 0 |

五个样本的固定分支答案均复现原始运行中的逐分支答案。该现象说明正确答案在
这些案例中已经存在于特定 Refiner message 中，而不是由最终 Judger 的随机 seed
偶然生成。

## 边界

本实验是机制冒烟验证，不是完整的总体效果估计。由于样本来自 Oracle@5-miss
条件子集，`Oracle@5 = 100%` 是选择条件的一部分，不能被报告为方法在 MedQA
全集上的准确率。实验按计划暂停，完整 32 样本的显著性分析未执行。

## 产物

- 结果目录：`artifacts/mp_statebridge_v1/refiner_causal_oracle_misses_qwen3_4b_greedy_seed7300_v1`
- 状态：`paused`
- 完整结果：`summary.json`
- 单条件输出：`generations/`
- 单样本汇总：`items/`
