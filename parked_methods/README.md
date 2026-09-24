# 我们自己提出的通信方法 —— 暂存，不用于论文

整理时间 2026-09-23。这里是 AgentCom 项目中**我们自己设计并跑过**的通信方法的全部结果。

**状态：全部暂停使用。** 论文已改为"通信审计框架 + 解释性对照"的定位，
不需要我们的方法赢。审计结果在 [`../results/`](../results/)。

保留这些材料的理由有两条：(1) 它们是真实跑出来的，不该丢；
(2) 其中几条**否定性结果**是审计结论的直接支撑——见本文末尾 §5。

原始记录 **没有移动**，仍在 `methods/AgentCom-StateBridge/runs/` 下，每张表的
`source_root` 列给出确切路径。

---

## 1. EGR — Evidence-Guided Revision

表：[`tables/egr.csv`](tables/egr.csv)

思路：不传完整推理，只传"证据"——删掉宣告答案的部分，保留推导过程。
所以 sender→receiver 与 receiver→sender 传的是同一份证据，方向对称。

| 变体 | 实现 | 做法 |
|---|---|---|
| V1 | `egr/evidence.py` | 原版 EGR 证据抽取 |
| V2 | `egr/evidence_v2.py` | 切句后整句删除含答案断言的句子；`policy="all_options"` 对称；`_ARITHMETIC` 正则保留推导步骤 |
| V3 | `egr/evidence_v3.py` | 统一规则"宣告但不推导即删"。**离线验证失败，泄漏率 89%，从未上机** |

结果（SI−50）：

| 数据集 | EGR V1 | EGR V2 | 参照：Full Text | 参照：StateBridge |
|---|---:|---:|---:|---:|
| MedQA | +11.64 | +10.34 | +11.21 | **+16.38** |
| ARC-C | — | +6.12 | **+7.65** | +7.14 |
| GSM8K | — | −0.54 | +1.63 | **+6.52** |
| GPQA-D | — | +11.07 `INCOMPLETE 633/708` | **+10.96** | +7.89 |

**结论：没有稳定优势。** EGR V2 在 GSM8K 上是 −0.54，比完全不通信还差。
V1 只在 MedQA 上跑过，+11.64 与 Full Text 的 +11.21 在噪声内。

### 交叉对照 `other_evidence`

用**另一道题**的证据当消息，MedQA 720 条：CR **0.00%**、PR 93.10%、采纳率 1.7%。
receiver 几乎完全不理会不相关的消息。这条对照说明 receiver 至少能识别内容是否匹配当前题目——
但注意它的混合方向有效样本极小（CR n=3、PR n=2），不足以单独作为"随机投递"对照使用。

---

## 2. Hybrid 系列 — StateBridge + 明文

表：[`tables/hybrid.csv`](tables/hybrid.csv)

思路：在 StateBridge 的隐状态载荷之上再附加一段明文。

| 变体 | 附加内容 |
|---|---|
| Hybrid | 结构化明文摘要 + 隐状态 |
| RawHybrid | 原始推理文本 + 隐状态（决定性消融） |
| Hybrid V3 | V3 版明文规则 + 隐状态 |
| Hybrid keep-chain | 保留推导链（仅 GSM8K） |

最好成绩是 MedQA 的 Hybrid **SI−50 = +17.24**，略高于 StateBridge 的 +16.38。
但其余三个数据集上都没有超过最好的已有通道：

| 数据集 | Hybrid | RawHybrid | Hybrid V3 | 该数据集最优已有通道 |
|---|---:|---:|---:|---:|
| MedQA | **+17.24** | +11.64 | +8.62 | StateBridge +16.38 |
| ARC-C | +7.65 | +4.59 | +4.59 | Full Text +7.65 |
| GSM8K | +3.26 | +1.09 | +1.63 | StateBridge +6.52 |
| GPQA-D | +7.46 | +5.38 `INCOMPLETE` | +6.45 `INCOMPLETE` | Full Text +10.96 |

**新颖性问题**：RawHybrid 消融证明这就是"StateBridge 上加明文"。
Hybrid(+17.24) 与 RawHybrid(+11.64) 的差别全部来自明文那一段怎么写，
而不是来自任何新的通信机制。这条路线不构成方法贡献。

---

## 3. CR-DNC — Decision Nullspace Communication

表：[`tables/cr_dnc.csv`](tables/cr_dnc.csv)

思路：把 sender 隐状态投影到"决策零空间"，按 alpha 控制注入强度，
试图只传递推理内容而不传递答案倾向。实现见 `communication/decision_nullspace.py`、
`communication/cr_dnc_runtime.py`、`communication/build_cr_dnc_prefixes.py`。

**全部 `INCOMPLETE 138/720`**，只在冻结的 dev 切片（`runs/medqa/cr_dnc_v1/split.json`）上跑过。

| 变体 | SI−50 |
|---|---:|
| alpha=0.25 | +9.09 |
| alpha=0.50 | +11.36 |
| alpha=0.75 | +11.36 |
| alpha=1.00 | +11.36 |
| **随机方向对照** | **+20.45** |
| **随机范数对照** | **+25.00** |

**这条路线被自己的对照否定了。** 两个随机对照都**大幅超过**所有真实的 alpha 设置。
如果随机方向的隐状态注入比精心构造的零空间投影效果更好，
那么"零空间"没有承载任何有用信息，效果来自注入本身的扰动。

n=138 是 dev 切片，不可与主表的 720 条并列；但对照与实验在同一 n 上，
这个比较本身是有效的。

---

## 4. Evidence Adjudication

表：[`tables/adjudication.csv`](tables/adjudication.csv)。实现 `egr/run_adjudication.py`。

思路：对无序配对做对称证据仲裁——同时给出两份证据，让模型裁决。

| 版本 | n | 准确率 |
|---|---:|---:|
| v1 | 360/360 | 26.94% |
| v2 | 315/360 `INCOMPLETE` | 27.94% |

MedQA 上各通道的 `Acc_ret` 落在 24.72%–28.89%。**仲裁没有跑出这个区间。**

---

## 5. 这些失败为什么值得留着

四条路线各自独立地撞上同一堵墙，这是 [`../results/04_DIAGNOSTICS.md`](../results/04_DIAGNOSTICS.md) 的实验依据：

| 路线 | 失败方式 | 对应的诊断结论 |
|---|---|---|
| EGR V2 | 删掉答案后 CR 降、PR 升，SI 没动 | 变化率参数化：通道只改变化率 |
| Hybrid | 增益全部来自明文措辞，RawHybrid 消融证实 | 同上 |
| CR-DNC | 随机对照胜过真实方法 | 隐状态里没有可分离的正确性方向 |
| Adjudication | 准确率落在基线区间内 | 没有无 ground-truth 的正确性信号 |

换句话说：我们不是"没调好"，而是**在当前协议和模型下，
可供选择性传输的正确性信号不存在**。这是一条可以写进论文的否定性结论，
但它属于审计部分，不属于方法部分。

---

## 6. 代码位置

| 路线 | 代码 |
|---|---|
| EGR | `egr/evidence.py`、`evidence_v2.py`、`evidence_v3.py`、`run_adjudication.py` |
| Hybrid | `icr/channels.py` 中的 `HybridEvidenceBridgeChannel`、`RawHybridBridgeChannel`、`HybridV3BridgeChannel` |
| CR-DNC | `communication/` 全目录 |

代码**没有删除**，随时可以重启。
