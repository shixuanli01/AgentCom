# Trajectory Memory Relay V1

## 目的

TMR 是独立于 StateBridge control 的训练自由通信方法。两种预注册变体为：

- `tmr_last64`：保持 StateBridge 的 post-`</think>` last-64 位置语义，只替换
  Procrustes pseudo-token transport；
- `tmr_coverage64`：保持相同 TMR transport，改用 full-trajectory facility-location
  coverage，其中 48 个代表位置加最后 16 个 tail anchors。

第一组比较回答 transport 问题，第二组比较回答 selection 问题，不混合解释。

## 冻结控制变量

- model：Qwen3-4B；
- task：MedQA 300；
- pipeline：Planner -> Critic -> Refiner -> Judger；
- prompts：原 StateBridge `build_agent_message_embedding_mas`，不改写；
- decoding：temperature 0.6，top-p 0.95，sampling enabled，max-new-tokens 8192；
- per-role seeds：`stable_seed(42, medqa, item_id, 0, role)`；
- parser：原 `extract_gsm8k_answer` + `normalize_answer`；
- communication budget：每跳、每层 K=64。

## TMR 数据流

Sender 在 Qwen decoder layers `11,23,35` 的 self-attention pre-hook 中捕获
`input_layernorm` 后、`q_proj/k_proj/v_proj` 前的最后位置状态。每次 generation forward
捕获一个 decision state，并要求三层 trajectory 长度与 generated token 数完全一致。

选中位置后，三层分别形成 `[1,K,2560]` native memory。Receiver 的同层使用自己的冻结
`q_proj/k_proj/v_proj/o_proj`、Q/K RMSNorm、GQA KV repetition 和 attention scale：

```text
receiver normalized states -> native Q
sender native memory       -> native K/V
position-free QK attention -> entropy gate -> native O
                            -> 0.25 self-attention norm cap
                            -> second residual attention branch
```

外部 attention 不使用 RoPE、causal mask、position IDs 或 positional bias；不修改 normal
self-attention KV cache。Memory 不作为 token，不通过 `inputs_embeds` 插入 prompt。

## 零参数保证

TMR 只安装 scoped PyTorch hooks，并保存临时 memory tensors。启动及每题结束均比较 model
parameter identity，断言新增 trainable parameters 为 0。没有 projector、learned query、
LoRA 或训练过程。

## 运行

从 `methods/AgentCom-StateBridge` 执行：

```bash
PYTHONPATH=. python \
  -u -m agentcom.multipath \
  --run-dir artifacts/mp_statebridge_v1/medqa_qwen3_4b_m5_seed42

PYTHONPATH=. python \
  -u -m agentcom.tmr_eval \
  --communication-method tmr \
  --tmr-selection last64 \
  --run-dir artifacts/tmr_v1/medqa300_last64_seed42

PYTHONPATH=. python \
  -u -m agentcom.tmr_eval \
  --communication-method tmr \
  --tmr-selection coverage64 \
  --coverage-scope full \
  --run-dir artifacts/tmr_v1/medqa300_coverage64_seed42
```

先用 `--limit 1` smoke。运行可在每题边界安全停止和恢复。

## 输出

每个 run directory 包含：

- `manifest.json`：方法、环境、实现 hash 和 frozen config；
- `records/item_NNNN.json`：四阶段 visible trace 与三跳诊断；
- `diagnostics.jsonl`：选择位置、三层 shape、gate、entropy、norm；
- `paired_vs_statebridge.jsonl`：逐题 paired outcome；
- `summary.json`：准确率、rescue、destruction、McNemar、invalid、latency 和 gate 汇总；
- `run.log`：可恢复进度。

不保存 hidden-state tensor。

## Smoke 结果

2026-09-14，RTX 5090，item 0：

| condition | result | latency | new parameters |
|---|---:|---:|---:|
| StateBridge control | correct C | 45.10s | 0 |
| TMR-last64 | correct C | 54.64s | 0 |
| TMR-coverage64 | correct C | 55.50s | 0 |

TMR-last64 三层 memory 均为 `[1,64,2560]`。平均 entropy `0.9268`，平均 gate
`0.0732`，norm cap 触发比例 `0.0484`。

TMR-coverage64 每跳均为 48 coverage + 16 tail。item 0 Planner hop facility objective
从 `935.2740` 上升至 `1054.8059`，selected memory 中 `28.125%` 来自原 final 64。

以上只验证实现和数值路径，不作为准确率证据。
