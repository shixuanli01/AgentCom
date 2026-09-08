# MP-StateBridge V1 实验计划（中文版）

## 0. 文档信息

- 文档编号：`MP-SB-V1`
- 状态：计划冻结前草案
- 日期：2026-09-05
- 基础方法：StateBridge release `0.1.0`，commit
  `3f6bf5442c6e8848555a6132516e6d36f35444fb`
- 基础模型：`Qwen/Qwen3-4B`
- 首个验证数据集：StateBridge 官方 MedQA 300 题子集
- 路径数：`m=5`
- 每条消息的最大连续前缀长度：`K=64`

英文镜像版本见 [MP_STATEBRIDGE_V1_PLAN_EN.md](MP_STATEBRIDGE_V1_PLAN_EN.md)。两份文档
具有相同的章节编号；发生歧义时，以中文版为当前实验执行依据。

## 1. 研究目标

本实验首先回答一个窄而明确的问题：

> 在不训练模型、不修改 StateBridge 对齐算法、暂不考虑额外推理成本的条件下，将单条
> Planner 推理路径扩展为 5 条独立随机路径，并让每条路径独立完成
> Planner -> Critic -> Refiner -> Judger，最后对 5 个答案进行多数投票，是否能够显著提高
> 最终任务准确率？

V1 的首要目标是验证指标增益是否存在，而不是一次性证明多层表示、多前缀融合或 latent
communication 的全部因果机制。

## 2. 核心假设

### H1：路径覆盖假设

5 条独立路径中至少有一条正确的概率，应高于单条路径正确率：

```text
Oracle@5 > Accuracy@1
```

如果该假设不成立，说明当前采样温度、prompt 或模型本身没有产生有效的答案多样性。

### H2：投票增益假设

当正确推理模式比错误模式更稳定时，5 路 plurality/majority vote 应优于单路径：

```text
Vote@5 > Accuracy@1
```

### H3：分支保真假设

每条路径独立使用 StateBridge 通信，应当能够把该路径的信息持续传递到最终 Judger，而不会
因为跨路径混合造成过早的信息坍缩。

## 3. V1 范围

### 3.1 本版本包含

1. 每个问题独立采样 5 条 Planner 路径。
2. 每条路径使用独立随机种子。
3. 每条路径独立执行完整四 Agent 链。
4. 每次 Agent 间通信沿用官方 StateBridge 的最后一层 hidden-state 对齐。
5. 分支数量始终保持为 5，不在后续 Agent 中继续扩张。
6. 最终对 5 个 Judger 答案执行预先固定的投票规则。
7. 完整保存文本、token、hidden state、对齐元数据、答案和运行统计。
8. 首先在完整 MedQA 300 题子集上做权威评估。

### 3.2 本版本明确不包含

1. 从多个 Transformer 层提取消息。
2. 把 5 条路径的 `5 x K` embeddings 拼接给同一个 Receiver。
3. 使用额外 Final Arbiter 代替投票。
4. 学习式 projector、router、gating 或 Set Transformer。
5. 动态决定路径数或动态分配 token 预算。
6. 路径树搜索、回溯或每层重新分叉。
7. 以降低成本为目标的压缩与加速。

这些内容只能在 V1 确认存在指标信号后进入后续阶段。

## 4. 方法定义

### 4.1 五路径生成

对于问题 `x_i`，Planner 使用完全相同的模型和 prompt，以 5 个独立随机种子生成：

```text
P_i,0, P_i,1, P_i,2, P_i,3, P_i,4
```

约束如下：

- 每条路径只看到原始问题，不看到其他路径。
- 不在 prompt 中要求刻意与其他路径不同。
- 沿用官方解码参数：temperature `0.6`、top-p `0.95`。
- 每个 `(dataset, item_id, branch_id, stage)` 使用稳定、可重建的独立种子。
- branch 0 同时作为严格配对的 `m=1` 基线。

这样可以把 `Vote@5` 与同一次实验中的 branch 0 逐题比较，避免使用历史运行结果造成环境和
随机性不一致。

### 4.2 固定宽度分支

每个 Planner 分支只产生一个 Critic、一个 Refiner 和一个 Judger：

```text
Planner 0 -> Critic 0 -> Refiner 0 -> Judger 0 -> answer 0
Planner 1 -> Critic 1 -> Refiner 1 -> Judger 1 -> answer 1
Planner 2 -> Critic 2 -> Refiner 2 -> Judger 2 -> answer 2
Planner 3 -> Critic 3 -> Refiner 3 -> Judger 3 -> answer 3
Planner 4 -> Critic 4 -> Refiner 4 -> Judger 4 -> answer 4
                                                   |
                                                   +-> vote
```

任意阶段都不允许一个输入分支再生成 5 个输出分支，因此总分支数恒定为 5，而不是指数增长。

### 4.3 每条分支的通信

Planner -> Critic、Critic -> Refiner、Refiner -> Judger 三次通信均保持官方配置：

- final decoder block hidden states；
- 去除 Qwen thinking 区域后保留最后 `K <= 64` 个状态；
- 每条消息独立进行 Procrustes 对齐；
- `lambda=1e-3`；
- vocabulary anchoring `alpha=0.3`；
- prefix scale `1.0`；
- aligned prefix 通过 `inputs_embeds` 注入下一 Agent。

禁止把 5 条路径共计 `5K` 个状态放入同一个 Procrustes 拟合。每条路径拥有自己的对齐统计和
旋转矩阵，防止异质路径相互干扰。

### 4.4 最终投票

MedQA 使用 A/B/C/D 标准化答案。主结果称为 `Vote@5`：

1. 对每个 Judger 输出使用同一个官方 parser。
2. 统计 5 个标准化标签的出现次数。
3. 唯一最高票标签作为最终答案。
4. 若出现 `2-2-1` 等并列，选择并列标签中最早出现的 branch 所给出的标签。
5. 非法答案记为 `INVALID`，不得静默删除或重新采样。
6. 单独报告并列率、非法率以及并列样本清单。

该规则必须在看到准确率前冻结。V1 不使用模型置信度，也不增加额外裁判推理。

除 `Vote@5` 外，利用同一批结果免费报告 `Vote@3`（branches 0-2），但它属于次要分析，不能
替代预注册的 `Vote@5` 主结果。

## 5. 对照与公平性

### 5.1 V1 必需对照

| 条件 | 定义 | 用途 |
|---|---|---|
| `Branch-0` | 只取 branch 0 的完整 StateBridge 链答案 | 配对的 `m=1` 主基线 |
| `Mean-Branch` | 5 条 branch accuracy 的平均值 | 检查某条固定分支是否异常 |
| `Oracle@5` | 5 个答案中任意一个正确即算正确 | 多样性可利用上限 |
| `Vote@3` | branches 0-2 投票 | 初步路径数趋势 |
| `Vote@5` | 全部 5 条路径投票 | V1 主方法 |

### 5.2 V1 后的因果对照

若 `Vote@5` 出现正向信号，下一阶段必须复用缓存的同一批 Sender 路径，加入：

- single-agent self-consistency@5；
- Text-MPath；
- exact-token-embedding MPath；
- no-message 五分支 pipeline；
- matched/shuffled StateBridge message；
- 将同一条路径复制 5 次的 duplicated-path 控制。

这些对照用于区分“多采样收益”“多 Agent 计算收益”和“latent communication 收益”。在它们
完成前，V1 只能声称多路径 StateBridge pipeline 提高或未提高指标，不能声称连续通信本身是
增益来源。

## 6. 数据与运行配置

### 6.1 首轮权威数据

- 数据集：仓库随附的 MedQA 300 题子集。
- 范围：必须跑完整 300 题，不使用 100 题替代最终结论。
- 模型：固定本地 `Qwen/Qwen3-4B` revision。
- 软件环境：记录 PyTorch、Transformers、CUDA、GPU 和代码 commit。
- 顺序：保持官方 item ID，不因失败或恢复运行改变样本顺序。

### 6.2 随机种子

使用一个公开的 `base_seed`，通过稳定哈希生成各阶段种子：

```text
seed = StableHash(base_seed, dataset, item_id, branch_id, stage)
```

要求：

- 禁止依赖 Python 进程内置的随机 hash。
- 同一 manifest 必须能重建所有种子。
- 恢复运行不能改变尚未运行分支的种子。
- 分布式 worker 数量变化不能改变种子分配。

### 6.3 资源策略

V1 在单张 RTX 5090 上只加载一份模型，5 条分支顺序执行。首版不为追求吞吐引入动态 batch，
避免不同长度的生成和 hook 缓存相互污染。

根据当前 MedQA 单分支约 54 秒的实测均值，顺序执行 5 分支的粗略预算约为每题 4.5 分钟、
完整 300 题约 22.5 小时。该估计不包含额外保存和故障恢复开销，只用于排期。

## 7. 产物与缓存规范

建议运行目录：

```text
artifacts/mp_statebridge_v1/
  medqa_qwen3_4b_seed42/
    manifest.yaml
    progress.json
    results.jsonl
    events.jsonl
    states/
      item_<id>_branch_<0-4>.safetensors
    traces/
      item_<id>_branch_<0-4>.json
    reports/
      authoritative_report.md
      per_item.csv
      vote_patterns.csv
```

每条 trace 至少保存：

- dataset item ID、问题 hash 和 gold label；
- branch/stage seed；
- 四个 Agent 的完整输出文本；
- generated token IDs 和截取位置；
- 三次传输的原始最后层 hidden states；
- prefix 实际长度、对齐参数与数值诊断；
- 每个 Judger 的 raw prediction 和 parsed prediction；
- token 数、时间、异常和终止原因。

原始 hidden states 使用 `safetensors` 和原生低精度保存。aligned prefix 可以由原始状态、token
IDs、固定模型 embedding 和 manifest 重建，因此默认不重复保存，但必须提供重建一致性测试。

## 8. 实施步骤

### Phase 0：协议冻结

- 冻结本计划、模型 revision、数据 hash、prompt hash 和依赖版本。
- 冻结投票及并列处理规则。
- 建立 manifest schema 和结果 schema。

完成条件：不运行正式样本也能生成完整 manifest，并通过 schema 校验。

### Phase 1：执行器实现

- 在官方 StateBridge `run_item` 外增加 branch 维度。
- 保持每个 branch 的 Agent 状态和 hidden-state hook 完全隔离。
- 实现逐 branch 原子保存、断点恢复和失败重试记录。
- 实现投票器和逐题配对统计。

完成条件：所有单元测试通过，且 `m=1` 路径与未包装的 StateBridge 在相同种子下逐字段一致。

### Phase 2：小规模冒烟测试

- 固定运行前 5 题，每题 5 branches。
- 人工检查 25 条完整链路。
- 验证不同 branch 确实使用不同随机流。
- 验证 branch 间不存在 prefix、token 或缓存串线。
- 验证停止、恢复后结果不变。

完成条件：无结构错误、无跨题/跨分支泄漏、所有结果可重建。

### Phase 3：MedQA 全集运行

- 跑完 300 题 x 5 branches。
- 每完成一个 branch 立即持久化。
- 不因中间准确率调整任何参数。
- 对系统错误只按预注册策略重试，不对普通错误答案重采样。

完成条件：1500 条 branch 结果全部进入终态，缺失与系统错误得到明确解释。

### Phase 4：权威报告

- 计算 Branch-0、每分支、Mean-Branch、Vote@3、Vote@5 和 Oracle@5。
- 输出 5 路投票模式、答案分歧率和每题 correction/harm。
- 对 `Vote@5` 与 Branch-0 做 paired bootstrap 95% CI 和 McNemar 检验。
- 输出准确率 confusion matrix、非法答案和并列样本。
- 冻结报告和结果文件 hash。

完成条件：报告能由 `results.jsonl` 一条命令完全重建。

### Phase 5：是否进入后续研究

根据第 10 节决策规则，选择进入严格通信对照、改进聚合器，或优先修复路径多样性。

## 9. 测试清单

- [ ] `m=1` wrapper equivalence test
- [ ] 5 个 branch seed 唯一且可复建
- [ ] 相同 seed 重跑结果一致
- [ ] 恢复运行不重复覆盖完成分支
- [ ] hidden-state 数量与 token 对齐
- [ ] 每次 Procrustes 只使用本 branch 数据
- [ ] branch 不读取其他 branch 输出
- [ ] 投票器覆盖 5-0、4-1、3-2、3-1-1、2-2-1 和 INVALID
- [ ] parser 与官方单路径评估一致
- [ ] 所有 300 个 item ID 唯一且齐全
- [ ] 报告指标可从原始结果重算

## 10. 决策标准

主比较为 `Vote@5 - Branch-0`。

### PASS

- `Vote@5` 点估计高于 Branch-0；
- correction 数量大于 harm 数量；
- paired 95% CI 不跨 0，且 McNemar 双侧 `p < 0.05`。

结论：存在可靠的五路径投票增益，进入通信因果对照和单次多前缀融合研究。

### PROMISING

- `Vote@5` 点估计高于 Branch-0；
- 但 95% CI 或显著性检验未通过。

结论：存在正向信号但证据不足，应扩展到 ARC-C/GSM8K 或增加独立 seed，而不能宣称方法已经
显著提高性能。

### AGGREGATION BOTTLENECK

- `Oracle@5` 明显高于 Branch-0；
- 但 `Vote@5` 没有提高。

结论：路径中存在可利用的正确答案，下一步优先研究置信度加权、验证器或 Final Arbiter。

### DIVERSITY FAILURE

- `Oracle@5` 与 Branch-0 接近；
- 路径答案和表示高度一致。

结论：当前采样没有产生有效多样性，应先调整采样或引入显式分叉，不进入复杂融合。

### FAIL

- `Vote@5` 低于 Branch-0，且 harm 多于 correction。

结论：多数错误具有更高一致性，需分析错误相关性；不得通过改变投票规则事后挽救主结果。

## 11. 报告措辞边界

V1 PASS 后允许声称：

> 在 Qwen3-4B 和当前 MedQA 协议下，5 条独立 StateBridge 推理分支的预注册投票结果显著优于
> 配对单分支结果。

V1 单独不能声称：

- StateBridge 比文本通信更有效；
- continuous hidden state 提供了 token embedding 之外的信息增益；
- 多 Agent 优于等计算量 single-agent self-consistency；
- 多层 hidden states 对应多条显式推理路径；
- `m=5` 是普遍最优路径数。

这些结论必须由第 5.2 节的严格对照或后续独立实验支持。

## 12. 后续路线（不属于 V1）

只有 V1 出现可利用信号后，按以下顺序推进：

1. 使用同一批缓存路径完成 Text、exact-token、no-message、shuffled 和 self-consistency 对照。
2. 比较固定 `K=64` 每路径预算与固定总 prefix 预算。
3. 将 5 个独立 prefix 在一个 Receiver 中做 single-pass fusion。
4. 比较 majority vote、置信度投票和 Final Arbiter。
5. 将多层 hidden states 作为“同一路径的多层视角”单独验证。
6. 只有多层视角产生额外 Oracle 增益后，再研究 multiple-subspace alignment。

该顺序保证第一次实验只改变一个核心因素：从一条随机推理链扩展为五条独立随机推理链。
