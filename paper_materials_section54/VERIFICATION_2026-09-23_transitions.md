# 离线核验：GPQA-D receiver_policy_transitions.csv

2026-09-23。触发：报告接收方指出三处计数疑点。**未新增任何 GPU 调用。**
全部计数从逐条 revision 记录独立重算，未读取任何已生成的 CSV 或百分比。

## 结论先行

**三处疑点都源自对话中汇总表格的列标签错误，`receiver_policy_transitions.csv` 本身正确。
主结果、正确性评分与配对分析均不受影响。**

逐格比对：12 个（条件 × 策略 × 分层）格子 × 5 个类别 = 60 个计数，
CSV 与逐条重算 **0 处不一致**。

## 逐条重算结果

五类互斥，优先级：`invalid_output` → `kept_receiver_prior` → `changed_sender_answer_missing`
→ `matched_sender_answer` → `third_answer`。

| 条件 | 策略 | 层 | 分母 | 无效 | 保持 | 匹配 | 第三 | 缺sender | 合计 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| No Message | V3 | CR | 114 | 1 | 90 | 16 | 7 | 0 | **114** |
| No Message | V3 | PR | 114 | 0 | 110 | 1 | 3 | 0 | **114** |
| Full Text | V3 | CR | 114 | 0 | 29 | 85 | 0 | 0 | **114** |
| Full Text | V3 | PR | 114 | 1 | 54 | 58 | 1 | 0 | **114** |
| StateBridge | V3 | CR | 114 | 1 | 40 | 73 | 0 | 0 | **114** |
| StateBridge | V3 | PR | 114 | 0 | 59 | 53 | 1 | **1** | **114** |
| No Message | V4 | CR | 114 | 3 | 94 | 8 | 9 | 0 | **114** |
| No Message | V4 | PR | 114 | 2 | 111 | 1 | 0 | 0 | **114** |
| Full Text | V4 | CR | 114 | 0 | 50 | 63 | 1 | 0 | **114** |
| Full Text | V4 | PR | 114 | 0 | 70 | 43 | 0 | **1** | **114** |
| StateBridge | V4 | CR | 114 | 0 | 55 | 58 | 1 | 0 | **114** |
| StateBridge | V4 | PR | 114 | 0 | 80 | 32 | 1 | **1** | **114** |

## 逐项答复

### 疑点 1 — Full Text V4 的 PR 行合计 113，预期 114

**不成立。** 该行为 `0 + 70 + 43 + 0 + 1 = 114`。
第 5 类 `changed_sender_answer_missing = 1` 在对话的汇总表中被漏列，
导致读者看到的是 `70 + 43 + 0 + 0 = 113`。CSV 的 `sums_to_den` 列本就是 `YES`。

### 疑点 2 — StateBridge V3 的 PR 行合计 113，预期 114

**不成立。** 同一原因：`0 + 59 + 53 + 1 + 1 = 114`，漏列的同样是 `changed_sender_answer_missing = 1`。

### 疑点 3 — StateBridge V4 含 1 条无效输出，有效改变率应为 92/227

**不成立。StateBridge V4 的 `invalid_output` 为 0。**

对话汇总表把 PR 行写作 `80 / 32 / 1 / 1`，第四个 `1` 被置于「无效」表头下，
但它实际是 `changed_sender_answer_missing`。正确计数：

```
有效改变率分母 = 228 − 0(无效) = 228
有效改变率分子 = 228 − 135(保持: CR 55 + PR 80) = 93
              = 匹配 90 + 第三 2 + 缺sender 1
有效改变率 = 93 / 228 = 40.79%
无效输出率 = 0 / 228 = 0.00%
```

报告中的 40.79% 正确。92/227 = 40.53% 是基于被漏列的类别反推出来的。

## 涉及的具体记录

### A. `changed_sender_answer_missing`（sender 初始答案不可解析，全部为 `destruction_risk`）

| item_id | 方向 | 条件 | 策略 | receiver_pre | receiver_post | sender_pre |
|---|---|---|---|---|---|---|
| 21 | C→B | StateBridge | V3 | a | d | `None` |
| 162 | C→B | Full Text | V4 | d | b | `None` |
| 21 | C→A | StateBridge | V4 | a | d | `None` |

这三条 receiver 都改了答案，但 sender 没有可解析的初始答案，
因此「匹配 sender」对它们无定义，归入独立类别而非 `matched_sender_answer` 或 `third_answer`。

### B. 无效输出逐条归因（GPQA-D 全部 8 条）

按三类归因：**截断**（`hit_eos=False`）、**缺少答案格式**（已终止但无 `\boxed{}`）、
**解析失败**（有 `\boxed{}` 但 parser 取不出合法选项）。

| item_id | 方向 | 条件 | 策略 | 分层 | hit_eos | gen_len | 有 `\boxed` | 归因 |
|---|---|---|---|---|---|---:|---|---|
| 63 | A→B | StateBridge | V3 | CR | False | 16384 | False | **截断** |
| 63 | B→C | Full Text | V3 | PR | False | 16384 | False | **截断** |
| 63 | C→B | No Message | V3 | CR | False | 16384 | False | **截断** |
| 71 | C→B | No Message | V4 | PR | False | 16384 | False | **截断** |
| 63 | A→B | No Message | V4 | CR | False | 16384 | False | **截断** |
| 63 | B→A | No Message | V4 | PR | False | 16384 | False | **截断** |
| 21 | B→C | No Message | V4 | CR | False | 16384 | False | **截断** |
| 63 | C→B | No Message | V4 | CR | False | 16384 | False | **截断** |

**8 条全部是截断。** 缺少答案格式 0 条，解析失败 0 条。
所问的 No Message V4 那 5 条即表中后 5 行。

`item 63` 贡献 8 条中的 6 条，是一道退化题目。

## 核验中发现的一个真实问题（非导出错误）

截断记录按现行规则**保留并判错**，V3 与 V4 规则相同，无差别排除——
全部 8 条的 `receiver_post_correct` 均为 False。这一点符合 `AUDIT_ISSUES` 的 A1。

但**截断数在两个策略间不对称**：

| 条件 | V3 截断 | V4 截断 |
|---|---:|---:|
| No Message | 1 | **5** |
| Full Text | 1 | **0** |
| StateBridge | 1 | **0** |

V4 的两步格式在**没有外部消息**时更容易让生成跑满 16384 token；有消息时反而降到 0。
由于截断一律判错，这会压低 No Message V4 的 CR。

敏感性（剔除任一策略下截断的配对后重算 ΔSI）：

| 条件 | 全部 n=228 | 剔除截断配对后 | 差异 |
|---|---:|---:|---:|
| **No Message** | **−3.07** | **−1.36** (n=223) | **+1.71** |
| Full Text | −2.63 | −3.01 (n=227) | −0.38 |
| StateBridge | +2.63 | +2.13 (n=227) | −0.50 |

**No Message 的 ΔSI 有一半以上来自 5 条截断记录。**
该 ΔSI 的区间本就跨零（[−7.02, +0.88]），这条敏感性进一步削弱它。
「V4 降低协议噪声」在 GPQA-D 上**不能**作为结论写。

Full Text 与 StateBridge 的 ΔSI 对截断不敏感（变动 < 0.5 pp）。

## 影响范围

| 项目 | 是否受影响 |
|---|---|
| `receiver_policy_transitions.csv` | **不受影响**，60 个计数全部正确 |
| 正确性评分（`receiver_post_correct`） | **不受影响** |
| `receiver_policy_results.csv` 的 CR/PR/SI | **不受影响** |
| `receiver_policy_paired_differences.csv` | **不受影响** |
| `receiver_policy_vs_no_message.csv` | **不受影响** |
| `receiver_policy_interactions.csv` | **不受影响** |
| `REPORT_SECTION54_ZH.md` | 不含该列标签错误；已补入上节截断敏感性 |
| 对话中的汇总表格 | **有错**：漏列 `changed_sender_answer_missing` 列 |
