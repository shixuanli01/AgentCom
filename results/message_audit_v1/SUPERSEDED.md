# 本文件夹已被取代 —— 不要引用这里的表

2026-09-23。本文件夹是 Section 5.3 的**第一轮**材料，只覆盖 MedQA 与 GPQA-D 两个数据集。

**权威版本是 [`../../paper_materials_section53/`](../../paper_materials_section53/)**，
它覆盖 MedQA、ARC-C、GSM8K、GPQA-D 四个数据集，含配对四格表、
互斥的答案迁移分类，以及每个数据集的随机发送参照。

两轮的重叠数值一致（例如 MedQA 的 Full Text − Answer Only ΔSI = +1.72，
随机参照差 +8.22），第二轮只是范围更大、拆解更细。

## 这里仍有、第二轮没有的两份

| 文件 | 内容 |
|---|---|
| `cost.csv` | 消息构造成本与 Receiver 成本分列，缺失项标 MISSING |
| `message_audit.csv` | 逐条消息的来源、长度、模态（MedQA + GPQA-D） |

需要这两项时可以用本文件夹，但**指标一律以第二轮为准**。

`uncommitted.patch` / `uncommitted_files.txt` 是第一轮生成时的代码快照，仅作溯源。
