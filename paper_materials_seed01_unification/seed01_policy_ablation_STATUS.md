# 策略消融统一到 seed_pair_01 —— **BLOCKED**

2026-09-23。未新增模型生成。

## 缺口

统一到 seed_pair_01 需要一组 **seed_pair_01 的 V4 记录**。**它不存在。**

扫描 `artifacts/` 与 `runs/` 下全部产物根目录，携带 `replication_id = seed_pair_01` 的
**只有** `runs/medqa/replication_01`，其配置为：

```
revision_prompt_version = icr_v3_mid_injection
receiver_policy         = revise
```

即 **V3**。该 run 只含两个条件：`true_statebridge`（720，完整）与 `none`（183，不完整）。

现存的唯一 V4 运行是 `runs/medqa/v4_verify`，`replication_id = seed_pair_00`。

## 逐项状态

| 要求 | 状态 | 说明 |
|---|---|---|
| 确认主表运行的策略版本 | **已确认为 V3** | `receiver_policy = revise`，故 V3 侧采用 54.31 / 71.55 / 62.93 |
| V4 使用可与该 V3 匹配的 seed1 记录 | **BLOCKED** | 无 seed_pair_01 V4 记录 |
| V3/V4 的 CR、PR、SI | **待定** | V4 侧缺失 |
| ΔCR、ΔPR、ΔSI | **待定** | 同上 |
| G 与 ΔG（相对各策略匹配的 No Message） | **双重阻塞** | V4 缺失；且 seed_pair_01 的 No Message 仅 183/720 |
| StateBridge × 策略 交互（相对 Full Text） | **三重阻塞** | V4 缺失；seed_pair_01 无 Full Text（0 条）；No Message 不完整 |
| 上述差异的配对 bootstrap CI | **待定** | 无可计算的点估计 |

## 跨通道交互的额外阻塞

即便补上 seed_pair_01 的 V4 StateBridge，交互仍不可算：
`runs/medqa/replication_01` **没有 Full Text 记录**（0 条），也没有完整的 No Message。
交互定义需要四个格子都在同一 `replication_id` 下，目前只有一个格子存在。

**不得**把 seed_pair_01 的 StateBridge 拼进 seed_pair_00 的 Full Text / No Message 比较。
已核验：seed_pair_01 与 seed_pair_00 的 `revision_seed` 在 720 条上 **0/720 相同**，
两者不构成配对设计。

## 补齐所需的最小生成量（本轮不执行）

| 条件 | replication_id | 策略 | 记录数 |
|---|---|---|---:|
| `none` 补齐 | seed_pair_01 | V3 | 537（现有 183，补到 720） |
| `true_text` | seed_pair_01 | V3 | 720 |
| `none` | seed_pair_01 | V4 | 720 |
| `true_text` | seed_pair_01 | V4 | 720 |
| `true_statebridge` | seed_pair_01 | V4 | 720 |
| | | **合计** | **3,417** |

若把作用域限制为混合方向（与 Section 5.4 的评价范围一致），
每条件 232 条，合计 **1,160 条**（`none` 补齐 210 + 其余四格 × 232 ≈ 1,138）。

在补齐之前，Section 5.4 中所有 MedQA StateBridge 的策略消融结论标记为 **待定**，
并按下述要求撤下旧值，不以 seed_pair_00 结果填补。
