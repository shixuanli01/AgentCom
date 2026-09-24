# 01 — 口径与定义

所有数值的统计口径都在这一份里。表格里任何数字对不上，先回来核对这里的定义。

## 1. ICR 协议

每道题 3 个 agent（A、B、C），三个阶段：

1. **Independent** — 每个 agent 独立作答，得到 3 个初始信念。
2. **Communicate** — 对每个有序对 (sender → receiver) 构造一条消息。3 个 agent 有 **6 个有序方向**：
   `A→B, A→C, B→A, B→C, C→A, C→B`。
3. **Revise** — receiver 看到消息后**修订一次**，产生一条记录。

一道题 × 一个通道 = **6 条有向记录**。每条记录独立修订，互不影响。

模型 `Qwen/Qwen3-4B`，temperature 0.6，top_p 0.95，max_new_tokens 16384，`enable_thinking=True`，bfloat16。
`global_seed=42`，`replication_id=seed_pair_00`。修订种子由
`replicated_stable_seed(global_seed, replication_id, item_id, ...)` 决定，**不包含 condition**——
这是刻意的匹配设计：同一道题同一方向，不同通道用同一个修订采样种子。

## 2. 四个分层

按 (sender 初始对错, receiver 初始对错) 把 6 个方向分成四类：

| 分层 | sender | receiver | 问的问题 |
|---|---|---|---|
| `correction_opportunity` | 对 | **错** | receiver 能被纠正吗？ |
| `destruction_risk` | 错 | **对** | receiver 能顶住错误信息吗？ |
| `both_wrong` | 错 | 错 | 两个都错还能救回来吗？ |
| `both_correct` | 对 | 对 | 两个都对还会被搞坏吗？ |

**混合方向** = 前两类。这是本研究的核心测量对象，因为只有这两类里"该不该改"有明确答案。

## 3. 指标

| 指标 | 定义 | 分层 |
|---|---|---|
| **CR** correction rate | P(修订后正确) | `correction_opportunity` |
| **PR** preservation rate | P(修订后正确) | `destruction_risk` |
| **SI** selective integration | (CR + PR) / 2 | 两者 |
| **SR** rescue rate | P(修订后正确) | `both_wrong` |
| **SCR** self-consistency retention | P(修订后正确) | `both_correct` |
| **Acc_ret** | P(修订后正确) | 全部 6 方向，保留题集上 |

补充的两个分解量（见 `04_DIAGNOSTICS.md`）：

| 量 | 定义 |
|---|---|
| **π_CR** | P(修订后答案 = sender 答案 \| `correction_opportunity`) |
| **π_PR** | P(修订后答案 = sender 答案 \| `destruction_risk`) |
| **采纳率 π̄** | (π_CR + π_PR) / 2 —— 消息把 receiver 推动了多少 |
| **判别力 Δπ** | π_CR − π_PR —— 这些推动里有多少是选择性的 |

**SI = 50 的含义**：receiver 完全不区分对错。`Keep Initial`（永不改变，CR=0 / PR=100）
和"以固定概率盲目采纳"（CR=π / PR=1−π）**SI 都恰好等于 50**。
所以论文里应该报 **SI − 50**，而不是 SI 本身。

## 4. 计数恒等式

一道题若 3 个信念中有 k 个正确，则该题贡献的方向数是封闭形式的：

```
CR 方向数 = PR 方向数 = k(3−k)
SR 方向数 = (3−k)(2−k)
SCR 方向数 = k(k−1)
```

`tables/count_identity_check.csv`：**全部 20 个格子 YES**。这是最强的一条完整性证据——
它证明分层标注和信念记录完全一致，没有错配。

## 5. 样本

`tables/sample_statistics.csv`（全部 VERIFIED）：

| 数据集 | 原始题数 | 全对跳过 | 保留题数 | CR/PR 分层 n | 每通道记录数 |
|---|---:|---:|---:|---:|---:|
| MedQA | 300 | 180 | 120 | 116 | 720 |
| ARC-C | 1165 | 1067 | 98 | 98 | 588 |
| GSM8K | 1319 | 1225 | 94 | 92 | 564 |
| GPQA-D | 198 | 80 | 118 | 114 | 708 |
| HumanEval+ | 164 | 136 | 28 | 28 | 168 |

**"全对跳过"是主要的作用域限制。** 三个 agent 全部答对的题不进入修订阶段——
这类题上 CR 分层为空（k=3 ⟹ k(3−k)=0），跑它们不产生任何混合方向信息，只消耗算力。
代价是 **Acc_ret 只在保留题集上定义**，不是全集准确率。

MedQA 每个通道另有 34 条 `all_correct_sample`，是 `keep_one_in=10` 抽样留下的全对题记录，
已从 `merged.jsonl` 分离到 `revisions/all_correct_sample.jsonl`。它们不进入任何主表。

## 6. 五种准确率口径 —— 不要混用

`tables/accuracy_scopes.csv` 列了全部五种。最常被搞混的是后三种：

| 口径 | 分母 | 状态 |
|---|---|---|
| 初始答案准确率 | 全集，通信前 | 已测 |
| 混合方向准确率 | CR+PR 分层 | 已测，**恒等于 SI** |
| `Acc_ret` | 保留题集 × 6 方向 | 已测 |
| 全集实测准确率 | 全集 | **不可得**（全对题未跑修订） |
| 全集外推准确率 | 全集 | **ESTIMATED**，见 `tables/full_set_accuracy_estimated.csv` |

"混合方向准确率恒等于 SI"这一条在 20 个格子上逐一验证过。

## 7. 作用域标记

表格里出现的标记，含义固定：

| 标记 | 含义 |
|---|---|
| `all_six_directions` | 6 个方向全跑，SR/SCR/Acc_ret 有效 |
| `mixed_directions_only` | 只跑了 CR+PR 分层 |
| `NOT_MEASURED` | 底层记录不存在。**不可用推算值填充** |
| `NOT_APPLICABLE` | 该量对此条件无定义（如 Keep Initial 的 SR） |
| `NOT_RUN` | 计划内但没跑 |
| `ESTIMATED` | 外推值，引用时必须保留标记 |
| `INCOMPLETE m/n` | 跑了但没跑完 |
| `UNDEFINED` | 该量在此任务类型上无定义（如代码任务的采纳率） |
