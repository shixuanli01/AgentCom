# 05 — 完整性

主表数字可信到什么程度。所有检查在 2026-09-21 跑过，产物是原始记录，**记录是权威**，文档不是。

## 1. 一致性检查 —— 全部 PASS

表：[`tables/integrity_checks.csv`](tables/integrity_checks.csv)

五个数据集全部 `VERIFIED`，逐项：

| 检查 | 结果 |
|---|---|
| 每题恰好 3 个信念 | 5/5 数据集全通过，`items_partial = 0` |
| 重复记录键 | 0 |
| receiver 先验与 phase-1 信念一致 | 全部 `pristine`，`stale = 0`，`no_belief = 0` |
| 同一配对在不同条件下先验是否分歧 | 0 —— 这是匹配设计成立的直接证据 |
| `status != complete` 的修订记录 | 0 |

## 2. 计数恒等式 —— 20/20 格子 YES

表：[`tables/count_identity_check.csv`](tables/count_identity_check.csv)

一题若 k 个信念正确，则 CR/PR 方向数 = k(3−k)，SR = (3−k)(2−k)，SCR = k(k−1)。
封闭形式与实际观测在全部 20 个（数据集 × 指标）格子上完全相等。

这条比任何其它检查都强：它同时验证了信念记录、分层标注、方向枚举三者一致。

## 3. 截断污染 —— 集中在被分析的子集里

表：[`tables/truncation_contamination.csv`](tables/truncation_contamination.csv)

未终止（达到 16384 token 上限）的信念本身占比很低，但**它们不是均匀分布的**：

| 数据集 | 信念级截断率 | CR/PR 子集中被污染的配对 |
|---|---:|---:|
| MedQA | 1/900 = 0.11% | **0.00%** |
| ARC-C | 2/3495 = 0.06% | 2.04% |
| GSM8K | 2/3957 = 0.05% | 4.35% |
| GPQA-D | 13/594 = 2.19% | **10.53%** |
| HumanEval+ | 12/492 = 2.44% | **32.14%** |

**截断的信念几乎总是错的，所以必然落进混合方向分层。**
信念级 0.06% 可以放大成 CR/PR 子集里的 4%，GPQA-D 是 10.5%，HumanEval+ 是 32.1%。

论文里报截断率时，必须报**被分析子集里的比例**，不能只报信念级比例。

敏感性分析见 [`tables/truncation_sensitivity.csv`](tables/truncation_sensitivity.csv)：
剔除被污染配对后，ARC-C 变动 −1.00 ~ +0.01 pp、GSM8K −2.20 ~ −1.98 pp、GPQA-D −4.46 ~ −2.76 pp，
**通道排名在四个数据集上都不变**。HumanEval+ 未做该分析（n 太小）。

## 4. 排除与复用

表：[`tables/exclusions_and_reuse.csv`](tables/exclusions_and_reuse.csv)

- **全对题跳过**是唯一的题目级排除，规模见 `01_SCOPE.md` §5。
- MedQA 每条件 34 条 `all_correct_sample` 已从 `merged.jsonl` 分离，不进任何主表。
- 信念与 StateBridge 前缀在所有条件间**复用同一份冻结产物**（符号链接），
  这是匹配设计的基础，也是 §1 中"先验分歧 = 0"的原因。

## 5. 已知问题

完整清单见 [`AUDIT_ISSUES.md`](AUDIT_ISSUES.md)。影响主结果的四条：

| 编号 | 问题 | 状态 |
|---|---|---|
| A1 | 文档称未终止记录已排除，实际产物保留了它们 | **产物为准**，敏感性已做（§3） |
| A2 | GPQA-Diamond 相关问题 | RESOLVED |
| A3 | 没有按题目的 dev/test 划分 | MISSING |
| A4 | 单次复现（只有 MedQA StateBridge 有第二个种子） | 作用域限制，见 `03_CONTROLS.md` §4 |

影响解释的还有 B3（消息构造成本不可跨通道比较）和 B5（StateBridge 并非处处 PR 最高）。

## 6. 代码与模型未固定的部分

`AUDIT_ISSUES.md` C3：模型 revision 与 attention 实现**未 pin**。
`APPENDIX_DETAILS.md` 记录了 `Qwen/Qwen3-4B` 但没有 commit hash。
复现时这是最可能出偏差的一项。
