# MedQA StateBridge 统一到 seed_pair_01

2026-09-23。**未新增任何模型生成。** 全部数值从逐条记录核验，未修改任何记录，
未通过挑选记录去匹配目标数字。

仓库内**不存在论文 LaTeX 源**（`find . -name '*.tex'` 无命中），因此 D 项交付修改清单而非 diff。

---

## A. seed1 数据来源和可用性清单

### A.1 权威来源

| 项 | 值 |
|---|---|
| artifact root | `methods/AgentCom-StateBridge/runs/medqa/replication_01` |
| 记录 | `revisions/*/records/*.json`，`condition == true_statebridge` |
| 记录数 | **720**（分层 116 / 116 / 436 / 52，与封闭式计数一致） |
| `replication_id` | `seed_pair_01` |
| `global_seed` | 42 |
| `config_fingerprint` | `355c5073cd75bb6fba78bad536a8081cca7855c7263452fe59411417cc2b8f52` |
| `revision_prompt_version` | `icr_v3_mid_injection` |
| `receiver_policy` | `revise` → **策略版本为 V3** |
| parser / scorer | `icr.parsing_v3@ICR-V3` |
| 模型 / 采样 | `Qwen/Qwen3-4B`；temperature 0.6，top_p 0.95，max_new_tokens 16384 |
| dataset sha256 | `ddee8dc64d3b2a1e5061c409c68d626109237640009168838a038db29efe802c`，300 行 |
| phase-1 | `prebeliefs` 符号链接指向 `artifacts/icr_v3/medqa_full_seed42/prebeliefs`，未改动 |
| 消息 | `messages` 同为符号链接，未改动 |
| seed 派生 | `replicated_stable_seed(global_seed, replication_id, item_id, direction)`，condition 不参与 |

### A.2 同一 run 内其它条件的可用性 —— 这是本轮的主要限制

| 条件 | 记录数 | 状态 |
|---|---:|---|
| `true_statebridge` | 720 | **完整** |
| `none` | **183** | **不完整**（应为 720；分层 22/22/133/6） |
| `true_text` | **0** | **缺失** |
| `true_latentmas` | 0 | 缺失 |
| `all_correct_sample.jsonl` | — | **不存在** |

### A.3 seed_pair_01 的 V4 —— **不存在**

扫描 `artifacts/` 与 `runs/` 全部产物根目录，携带 `replication_id = seed_pair_01` 的
**只有** `runs/medqa/replication_01`，而它是 V3。
唯一的 V4 运行 `runs/medqa/v4_verify` 是 `seed_pair_00`。详见
[`seed01_policy_ablation_STATUS.md`](seed01_policy_ablation_STATUS.md)。

---

## B. 核验后的主表、全集估计及 No Message 配对统计

### B.1 主表 —— 六项声明值**全部逐条核验一致**

表：[`seed01_main_table.csv`](seed01_main_table.csv)

| 指标 | 计数 | 值 | 声明值 | 核验 | 95% CI |
|---|---|---:|---|---|---|
| CR | 63/116 | 54.31% | 63/116 = 54.31% | ✓ | [42.62, 65.45] |
| PR | 83/116 | 71.55% | 83/116 = 71.55% | ✓ | [61.70, 81.03] |
| SR | 2/436 | 0.46% | 2/436 | ✓ | [0.00, 1.28] |
| SCR | 52/52 | 100.00% | 52/52 | ✓ | [100.00, 100.00] |
| SI | — | 62.93% | 62.93% | ✓ | [55.74, 70.09] |
| Acc_ret | 200/720 | 27.78% | 200/720 = 27.78% | ✓ | [21.67, 34.17] |

**区间说明**：上表用 **2000 次重采样 / analysis seed 20260921**，与主表其余行的设置一致。
`runs/medqa/replication_01/export/` 里已有一份 seed01 区间，但用的是
**10000 次 / seed 20260922**，与主表不同口径，**不得混入同一张表**。
两套并列供核对：

| 指标 | 主表口径 2000/20260921 | 旧导出 10000/20260922 |
|---|---|---|
| CR | [42.62, 65.45] | [42.59, 65.83] |
| PR | [61.70, 81.03] | [60.83, 81.45] |
| SI | [55.74, 70.09] | [55.45, 70.24] |
| Acc_ret | [21.67, 34.17] | [21.94, 33.89] |

### B.2 附录 C — Acc_ret

**28.89% → 27.78%（200/720）**，区间 [21.67, 34.17]。
旧值 28.89%（208/720）来自 seed_pair_00，**必须整行替换，不能只换点估计保留旧区间**。

### B.3 Table 14 / 全集估计 —— 点估计不变，证据基础变弱

表：[`seed01_fullset_estimate.csv`](seed01_fullset_estimate.csv)

**核查结果：seed_pair_01 没有 skipped sample。** `runs/medqa/replication_01` 下不存在
`all_correct_sample.jsonl`。那份 34/34 属于 `artifacts/icr_v3/medqa_full_seed42`（seed_pair_00），
按要求**未自动复用**。

因此你给出的条件「若 seed1 对应的 skipped sample 确为 34/34」**不成立**。
实际可用的输入只有：

```
分母           = 300 题 × 6 方向 = 1800
retained 观测  = 720，判对 200
未观测方向     = 1800 − 720 = 1080   （其中 34 条在 seed_pair_00 被实测，seed_pair_01 未测）
假设           = 未观测方向全部记为正确
估计           = (200 + 1080) / 1800 = 1280/1800 = 71.11%
```

**点估计恰好等于你给的 71.11%**，因为 `34 + 1046 = 1080`，而两部分都按 1 计入。
但证据基础不同：seed_pair_00 有 34 条实测支撑（34/34 全对），**seed_pair_01 一条都没有**。

现有 `results/tables/full_set_accuracy_estimated.csv` 的 MedQA StateBridge 行是
**71.5556%**（seed0 的 208 correct），须改为 **71.11%**；其区间 [68.61, 74.51] 与
`item_unit_accuracy_pct = 72.0` 一并失效，须重算或撤下。

**全集准确率始终标 `ESTIMATED`。**

### B.4 与 No Message 的配对 —— 声明值来自跨种子配对

表：[`seed01_vs_no_message.csv`](seed01_vs_no_message.csv)

你给的候选四格 **144 / 56 / 41 / 479**（合计 720）**能够复现**，但它来自
**StateBridge seed_pair_01 × No Message seed_pair_00**：

| 配对 | n | 都对 | gain | loss | 都错 | net | `revision_seed` 相同 | ΔAcc | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| **B: None seed_pair_00** | 720 | **144** | **56** | **41** | **479** | +15 | **0/720** | **+2.08 pp** | [−0.42, +4.58] |
| **A: None seed_pair_01** | 183 | 24 | 7 | 11 | 141 | −4 | **183/183** | **−2.19 pp** | [−6.52, +1.65] |

**这是本轮最需要你决定的一点。** 两个选项给出**相反符号**，且都跨零：

- **选项 B** 复现了你给的计数，覆盖全部 720 方向，但 `revision_seed` **0/720 匹配**——
  StateBridge 用 seed_pair_01 采样、No Message 用 seed_pair_00 采样，
  差值同时包含通道效应与修订采样效应。已实测该采样漂移在 StateBridge 上是 SI 3.45 pp，
  与这里的 ΔAcc 量级（2.08 pp）同阶，因此混淆不可忽略。
- **选项 A** 是唯一 seed 一致的配对（183/183 匹配），但只覆盖 183/720 方向，
  且其 No Message 的 Acc_ret 仅 19.13%，与完整 seed_pair_00 的 25.69% 差距很大，
  说明这 183 条不是随机子集，**不能视为无偏**。

我按两者都算并都落盘。**若要严格执行「统一到 seed_pair_01」，选项 B 不满足要求**
（它的基线仍是 seed_pair_00）；而选项 A 满足要求但样本不完整且可能有选择偏差。
补齐 `none` 的 537 条即可消除这一取舍。

---

## C. seed1 策略消融及派生比较 —— **BLOCKED，全部待定**

详见 [`seed01_policy_ablation_STATUS.md`](seed01_policy_ablation_STATUS.md)。

- **主表运行的策略版本已确认为 V3**（`receiver_policy = revise`），
  故消融的 V3 侧采用 **54.31 / 71.55 / 62.93**。
- **V4 侧无 seed_pair_01 记录**，`ΔCR / ΔPR / ΔSI / G / ΔG / 交互 / 全部 CI` **无法计算**。
- 交互另有两重阻塞：`replication_01` 无 Full Text（0 条），No Message 不完整。
- **未使用任何 seed_pair_00 值填补。**

最小补齐量：混合方向口径约 **1,160 条**，全方向口径 **3,417 条**（本轮不执行）。

---

## D. 论文修改清单

仓库内无 LaTeX，以下按内容定位。**逐条手改，不要做全局查找替换**——
`75.0`、`57.76` 等数值在其它方法与任务上也出现，盲替会误改无关结果。

### D.1 必须替换（有可靠的 seed1 替代值）

| 位置 | 旧值（seed_pair_00） | 新值（seed_pair_01） |
|---|---|---|
| 主表 / Table 9 MedQA StateBridge CR | 57.76% [47.06, 68.03] | **54.31% [42.62, 65.45]** |
| 主表 / Table 9 MedQA StateBridge PR | 75.00% [65.09, 84.45] | **71.55% [61.70, 81.03]** |
| 主表 / Table 9 MedQA StateBridge SI | 66.38% | **62.93% [55.74, 70.09]** |
| 主表 SI−50 | +16.38 | **+12.93** |
| 附录 C / Table 9 Acc_ret | 28.89% [22.50, 35.56] | **27.78% [21.67, 34.17]** |
| Table 9 SR | 0.46%（2/436，seed0） | **0.46%（2/436，seed1）** — 数值同，来源须改 |
| Table 9 SCR | 100.00%（52/52） | **100.00%（52/52）** — 数值同，来源须改 |
| Table 14 全集估计 | 71.56%，CI [68.61, 74.51] | **71.11%（`ESTIMATED`），CI 须重算或撤下** |
| Table 14 `measured_skipped_sample` | 34 | **MISSING** — seed1 无 skipped sample |
| Table 14 `retained_correct` | 208 | **200** |
| Table 14 `item_unit_accuracy_pct` | 72.0 | **须重算或撤下** |

### D.2 须撤下（无 seed1 替代值，标记待定）

| 位置 | 旧值 | 处理 |
|---|---|---|
| 4.4 策略消融 V3 侧 | 57.76 / 75.00 / 66.38 | 换为 54.31 / 71.55 / 62.93 |
| 4.4 StateBridge ΔSI | −9.91 [−16.38, −3.45] | **撤下，标待定** |
| 4.4 StateBridge ΔG | −8.62 [−16.38, −1.29] | **撤下，标待定** |
| 4.4 StateBridge−Full Text 交互 | −10.78 [−18.97, −2.59] | **撤下，标待定** |
| 摘要 / 引言 / 结论中的「排名反转」表述 | — | **撤下**，见 E.3 |
| Table 4 / Table 10 中引用上述差值的行 | — | 标待定 |

### D.3 须核查来源但**不一定要改**

| 位置 | 说明 |
|---|---|
| Table 2 | 若只含 phase-1 初始准确率，与修订 seed 无关，**不受影响**；须确认 |
| Table 15 / 成本表 | StateBridge 载荷为 64×2560=327,680 bytes，seed0 与 seed1 **逐字节相同**（已核验 232/232），**不受影响** |
| `results/tables/change_rate_parameterisation.csv` | 含 MedQA StateBridge 的 57.76/75.00，须用 seed1 重算，R² 会变 |
| `results/tables/cr_pr_plot_data.csv` | 同上，绘图数据须重算 |
| `results/tables/seed_replication.csv` | 该表的用途正是对比两个种子，**保留两行**，但须注明主表现采用 seed_pair_01 |
| `paper_materials_section53/` 全部 | 不含 StateBridge，**不受影响** |

### D.4 明确不受影响

ARC-C / GSM8K / GPQA-D 的 StateBridge、以及所有数据集的 No Message / Full Text /
LatentMAS / Answer Only，**一律不动**。本轮只改 MedQA StateBridge 及直接依赖它的比较。

---

## E. 旧结论中哪些保留、哪些改变、哪些暂时无法支持

### E.1 保留

1. **MedQA StateBridge 在 V3 下 SI 高于 No Message**：62.93 vs 52.59，方向不变（幅度从 +13.79 降到 +10.34）。
2. **SR 与 SCR 无区分度**：seed1 的 2/436 与 52/52 与 seed0 完全相同。
3. **StateBridge 载荷与成本**：两个种子逐字节相同，成本表不变。
4. **种子复现本身的结论**：SI 漂移 3.45 pp 落在 seed_pair_00 的 bootstrap 区间内，
   说明率在重采样下稳定。这一条现在反而更重要——它是量化上面 B.4 混淆的依据。

### E.2 改变

5. MedQA StateBridge 的全部主表数值下移约 3.4 pp（CR −3.45、PR −3.45、SI −3.45、Acc_ret −1.11）。
6. 全集估计 71.56% → **71.11%**，且 seed1 **无任何 skipped sample 实测支撑**。
7. **MedQA 上 StateBridge 是否仍为最优通道，需要重新判断。**
   seed0 下 StateBridge SI 66.38 高于 Full Text 61.21；seed1 下 62.93 vs 61.21，
   差距从 5.17 pp 缩到 1.72 pp。但注意 Full Text 仍是 seed_pair_00，
   **这个比较本身现在是跨种子的**，在补齐 seed1 的 Full Text 之前无法干净地下结论。

### E.3 暂时无法支持

8. **「V4 使 StateBridge 的 SI 显著下降」（ΔSI −9.91）** —— 无 seed1 V4，**待定**。
9. **「V4 显著压缩 StateBridge 的通信增量」（ΔG −8.62）** —— **待定**。
10. **「通道相对优势对接收策略敏感」（交互 −10.78）** —— **待定**，且该结论在 GPQA-D 上
    本就未复现（+5.26 [−4.39, +14.91]，跨零）。撤下 MedQA 这条之后，
    **Section 5.4 目前没有任何数据集支持该交互**。
11. **「StateBridge 优于 Full Text」在 MedQA 上** —— 跨种子，**待定**（见 E.2 第 7 条）。

### E.4 本轮新增的一条限制

12. **MedQA StateBridge 与 No Message 的配对方向取决于基线来源**：
    跨种子配对给 +2.08 pp，seed 一致配对给 −2.19 pp，两者都跨零。
    在补齐 seed_pair_01 的 No Message 之前，**不应报告该配对的方向性结论**。
