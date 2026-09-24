# 03 — 对照实验

这些实验不是为了让某个通道赢，而是为了确定**主表里的数字有多少来自通信、多少来自协议本身**。

## 1. Keep Initial —— 解析式参考

receiver 永不修改答案。则按定义：

```
CR = 0%      PR = 100%      SI = 50%      Δπ = 0
SR、SCR、Acc_ret 无定义（NOT_APPLICABLE）
```

零 GPU 成本。它的作用是**锚定 SI 的零点**：任何 SI ≤ 50 的通道，其表现不如完全不通信。

按这个标准，主表里 **LatentMAS 在 MedQA 上（51.29）几乎踩线**，
**Full Text 在 GSM8K 上（51.63）同样几乎踩线**。

## 2. No Message —— 修订过程自身的噪声

receiver 不收任何消息，但仍被要求走一遍修订。它与 Keep Initial 的差值就是**协议噪声**：

| 数据集 | No Message SI−50 | 解读 |
|---|---:|---|
| ARC-C | +0.51 | 噪声可忽略 |
| MedQA | +2.59 | 噪声小 |
| GSM8K | +2.72 | 噪声小 |
| GPQA-D | +5.26 | 噪声不可忽略 |
| HumanEval+ | +25.00 | **噪声主导，该数据集的通信效应无法分离** |

GPQA-D 的 +5.26 意味着该数据集上 StateBridge 的 +7.89 里，
有相当部分不是通信带来的。这一点在写 GPQA 结论时必须说。

## 3. 接收策略消融 —— V4 verify-then-decide

表：[`tables/receiver_policy_v4.csv`](tables/receiver_policy_v4.csv)。数据集 MedQA，四通道 × 720 条，完整。

V3（主表用的）修订提示让 receiver 直接对比"外部消息 vs 我之前的答案"。
V4 改成两步：先把外部消息单独对照原题检查（明确禁止此时与自己的答案比较），
再用同样方式检查自己的推理，最后选两步都存活的一方。

| 通道 | V3 SI−50 | V4 SI−50 | 采纳率 V3 → V4 |
|---|---:|---:|---|
| No Message | +2.59 | **+1.29** | 3.9% → 2.2% |
| Full Text | +11.21 | +12.07 | 69.0% → 44.8% |
| StateBridge | +12.93 | **+7.33** | 41.4% → 21.1% |
| LatentMAS | +1.29 | **+11.21** | 68.5% → 32.8% |

三条结论：

1. **V4 把协议噪声减半**（No Message +2.59 → +1.29），这是它做对的地方——
   `No Message` 的理想值就是 CR=0 / PR=100 / SI=50。
2. **V4 大幅压低采纳率**（每个通道都降一半左右），说明"先独立检查消息"确实让 receiver 更难被推动。
3. **排名变动**。V3 下 StateBridge 第一（+12.93）、LatentMAS 垫底（+1.29）；
   V4 下 LatentMAS 升到 +11.21、StateBridge 掉到 +7.33。
   **2026-09-24 起 StateBridge 两侧都是 seed_pair_01，该比较已受控（`revision_seed` 232/232 相同）。**
   受控后 StateBridge 的 ΔSI 区间变为跨零：跨种子版 −6.47 [−11.64, −1.29] 是假阳性。

第 3 条是本研究最重要的稳健性警告：**通道排名对接收策略的措辞高度敏感**。
任何"某通道更好"的断言都必须绑定接收策略。

作用域：**仅 MedQA**。`runs/gpqa/v4_verify` 有配置但 **0 条记录**（未运行）。

## 4. 种子复现

表：[`tables/seed_replication.csv`](tables/seed_replication.csv)。

信念与 StateBridge 前缀是冻结产物（符号链接，未改动），只改 `replication_id`，
从而改变 `revision_seed`，也就是 receiver 的采样。

| 通道 | seed_pair_00 | seed_pair_01 | 差 |
|---|---:|---:|---:|
| StateBridge（n=720 / 720） | 66.38%（**已撤下**） | **62.93%（全文采用）** | −3.45 |
| No Message（n=720 / **183**） | 52.59% | 54.55% | +1.96 |

StateBridge 的 3.45 个百分点漂移落在 seed_pair_00 的 bootstrap 区间
[55.45%, 70.24%] 内，说明 MedQA 的率在重采样下稳定。

**No Message 的 seed_01 只有 183/720 条，标记 `INCOMPLETE`，不要引用。**

详细复现包见 `methods/AgentCom-StateBridge/runs/medqa/replication_01/export/`。

只有 MedQA 有第二个种子；其余四个数据集和其余三个通道都只有 seed_pair_00。
这是 `AUDIT_ISSUES.md` 的 A4（单次复现）。

## 5. 随机消息投递 —— **NOT RUN**

计划中的对照：给 receiver 一条**内容不匹配**的消息（来自随机选取的另一道题），
用来分离"消息内容有用"与"仅仅出现一条消息就会改变行为"。

**状态：未运行，本文件夹不提供数值。**

早先对话中曾口头引用过一组"随机投递参考"的差值。
那组数字**无法从任何已记录的定义复现**，因此**不得写入论文**。
若需要这个对照，必须先固定定义再实跑。

唯一与之相关的已有数据是 MedQA 上的 `other_evidence` 条件（720 条，
用另一道题的 EGR evidence 作为消息内容）。它绑定在 EGR 上，放在
[`../parked_methods/EGR.md`](../parked_methods/EGR.md)，且其混合方向有效样本极小（CR n=3、PR n=2），
不足以作为独立对照。
