# 逐条记录导出

2026-09-24。本目录让 `results/` 与 `paper_materials_section5*/` 里的**每一个数字都能被重算**，
而不需要产生它们的 5.3 GB 原始生成结果——那些只存在于已销毁的 GPU 实例上。

## 文件

| 文件 | 行数 | 体积 | 内容 |
|---|---:|---:|---|
| `revisions_index.csv.gz` | 32,358 | 9.2 MB | 全部修订记录，覆盖 5 个数据集 × 全部条件 × 全部接收策略 |
| `beliefs_index.csv.gz` | 9,490 | 4.0 MB | phase-1 独立作答信念，含 StateBridge 前缀的形状与路径 |
| `coverage.csv` | 49 | 1 KB | 每个 (benchmark, condition) 的记录数，快速核对覆盖 |

## 剔除了什么，为什么

三个字段占了原始体积的 90% 以上，**未导出**：

- `raw_response` / `response` —— receiver 的完整生成文本
- `generated_token_ids` —— token 序列
- phase-1 的 `reasoning_text` 与 `question`

分析所需的一切都保留了：答案、正确性、分层、种子、hash、token 计数、终止状态、消息元数据。
**做不了的只有一件事：读回模型实际写了什么。** 任何指标、任何重新分层、任何自助法都不受影响。

## 关键字段

| 字段 | 含义 |
|---|---|
| `source_root` | 产物根目录。**同一 condition 在不同 root 下是不同的运行**，必须用它区分 |
| `replication_id` | `seed_pair_00` 或 `seed_pair_01`。MedQA StateBridge 全文采用 seed_pair_01 |
| `pair_classification` | `correction_opportunity`（CR 组）/ `destruction_risk`（PR 组）/ `both_wrong` / `both_correct` |
| `revision_seed` | 修订采样种子。**不含 condition**，这是匹配设计的基础；配对比较前应先核验它逐条相同 |
| `receiver_pre_correct` / `receiver_post_correct` | 修订前后的正确性，CR/PR/SI 由此计算 |
| `hit_eos` | `False` 表示达到 16384 token 上限被截断。按既有规则**保留并判错** |
| `msg_*` | 消息元数据：modality、token 数、字节数、sender 答案是否不可解析 |

接收策略不是字段——它由 `source_root` 决定：`runs/*/v4_verify*` 是 V4
（`icr_v4_verify_then_decide`），其余都是 V3（`icr_v3_mid_injection`）。
各 root 的完整配置在同目录的 `config.json`（已随仓库提交）。

## 重算示例

```python
import csv, gzip, sys, collections
csv.field_size_limit(sys.maxsize)          # HumanEval+ 的 gold 是代码，字段较长
rows = list(csv.DictReader(gzip.open("revisions_index.csv.gz", "rt", encoding="utf-8")))
T = lambda v: v == "True"

def si(rs):
    cr = [r for r in rs if r["pair_classification"] == "correction_opportunity"]
    pr = [r for r in rs if r["pair_classification"] == "destruction_risk"]
    a = sum(T(r["receiver_post_correct"]) for r in cr)
    b = sum(T(r["receiver_post_correct"]) for r in pr)
    return a, len(cr), b, len(pr), (100*a/len(cr) + 100*b/len(pr)) / 2

sel = [r for r in rows if r["source_root"] == "runs/medqa/replication_01"
                       and r["condition"] == "true_statebridge"]
print(si(sel))      # (63, 116, 83, 116, 62.931...)  == 主表
```

phase-1 的封闭式计数同样可复核：一题若 3 个信念中 k 个正确，则该题贡献
CR 方向数 = PR 方向数 = `k(3−k)`，SR = `(3−k)(2−k)`，SCR = `k(k−1)`。
在 `beliefs_index.csv.gz` 上按题求和，MedQA 得 116，与主表一致。

## 已验证

导出后重算并与已发布值逐一比对，**6/6 一致**：

| 检查 | 重算 | 已发布 |
|---|---|---|
| MedQA StateBridge V3 seed01 | 62.93 (63/116, 83/116) | 同 |
| MedQA StateBridge V4 seed01 | 57.33 (32/116, 101/116) | 同 |
| GPQA-D LatentMAS V4 | 59.21 (50/114, 85/114) | 同 |
| GPQA-D Full Text V4 | 58.33 (63/114, 70/114) | 同 |
| MedQA Answer Only | 59.48 (45/116, 93/116) | 同 |
| GSM8K Answer Only | 54.89 (36/92, 65/92) | 同 |

加上从 `beliefs_index.csv.gz` 重算的 MedQA 封闭式 CR 分母 = 116。

## 一处口径提醒

`artifacts/icr_v3/medqa_full_seed42` 下每个条件有 754 条，而主表用 720 条。
多出的 34 条是 `keep_one_in=10` 抽样留下的全对题记录，原本分离在
`all_correct_sample.jsonl`。本导出**包含**它们（`pair_classification == "both_correct"`
且题目为全对），重算主表时需按原口径排除。ARC-C / GSM8K / GPQA-D / HumanEval+ 同理。

## 不在这里的

StateBridge 的隐状态前缀（`.safetensors`，仅 MedQA 就 289 MB）与 LatentMAS 的 KV cache
未导出。`beliefs_index.csv.gz` 保留了它们的 `statebridge_prefix_file`、`_shape`、`_dtype`，
足以描述载荷规格，但**重跑生成需要重新构造前缀**。
