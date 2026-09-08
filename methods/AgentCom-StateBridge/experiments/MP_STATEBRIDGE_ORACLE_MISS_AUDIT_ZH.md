# MP-StateBridge V1 Oracle Miss 轨迹审计

## 1. 审计问题

本报告分析 MedQA 300 题正式运行中满足以下条件的样本：

```text
Oracle@5 = correct
Vote@5   = wrong
```

这类样本共 32 题。目标是回答：正确答案为什么只在五条路径中出现一次或两次，
以及能否从路径内容或 latent state 中找到可用于选择正确候选的信号。

本报告不重新运行模型，不修改原始结果。所有医学正确性以数据集 gold label 为准。
“高可信/部分可信/幸运命中”是对现有生成轨迹的研究性人工编码，不等同于独立临床专家复核。

## 2. 总体分布

| 正确答案出现次数 | 题数 | 占 32 题 |
|---|---:|---:|
| 1 次 | 17 | 53.1% |
| 2 次 | 15 | 46.9% |

| 票型 | 题数 |
|---|---:|
| 4-1 | 13 |
| 3-2 | 13 |
| 2-2-1 | 3 |
| 3-1-1 | 2 |
| 2-1-1-1 | 1 |

正确答案从未获得三票。13 个 `4-1` 表明错误答案在这些题上是稳定的主导模式；
13 个 `3-2` 则表明模型在两个相邻解释之间摇摆，但错误解释略占优势。

## 3. 正确轨迹的质量

逐题比较至少一条正确路径和主导错误路径后：

| 人工编码 | 题数 | 含义 |
|---|---:|---|
| 高可信正确推理 | 23 | 抓住了决定性证据，结论与解释基本一致 |
| 混合质量 | 1 | 同一正确答案的两条路径中，一条可靠、一条存在明显概念错误 |
| 部分可信 | 7 | 方向或答案正确，但诊断名称、机制或论证存在缺口 |
| 幸运命中 | 1 | 核心诊断/机制错误，但最终选项碰巧等于 gold |

因此，至少 24/32（75%）的题中存在一条内容上可辨认的正确推理路径。
Oracle gap 主要不是由随机猜标签造成的，而是由低概率但有意义的知识检索造成的。

### 3.1 逐题编码

| Item | 正确票数 | 正确路径抓住的关键区别 | 质量 |
|---:|---:|---|---|
| 4 | 2 | 局限性口腔念珠菌病：局部 nystatin，而非系统 fluconazole | 高可信 |
| 23 | 1 | 独立行走对应 12 月龄，而非 9 月龄 | 高可信 |
| 33 | 2 | PCOS 需要代谢共病筛查，而非先按 CAH 检查 | 高可信 |
| 37 | 1 | 高泌乳素抑制性腺轴并降低 progesterone | 高可信 |
| 38 | 1 | 镰状细胞肾病损害尿液浓缩能力 | 高可信 |
| 41 | 2 | 面部与身体同侧纯感觉缺失指向丘脑，而非交叉性脑干综合征 | 高可信 |
| 43 | 1 | 脑膜炎球菌败血症相关肾上腺皮质功能不足 | 部分可信 |
| 45 | 2 | CLL 与 smudge cells 的对应 | 部分可信 |
| 46 | 1 | 哭泣改善、喂养加重提示后鼻孔闭锁 | 高可信 |
| 55 | 1 | 血清阴性脊柱关节病与 psoriasis/Auspitz sign 的关联 | 高可信 |
| 60 | 2 | 初次体液免疫反应首先产生 IgM | 高可信 |
| 79 | 1 | Cocaine 阻断 dopamine/norepinephrine reuptake | 高可信 |
| 84 | 1 | 尾部退化谱系与 maternal diabetes 的关联 | 部分可信 |
| 86 | 2 | 中年女性胆汁淤积、干燥与 PBC/xanthomas | 高可信 |
| 89 | 1 | 最终选到 autosomal dominant，但生成理由误诊为 PAVM/HHT | 幸运命中 |
| 96 | 1 | Giant cell arteritis 需立即 steroid，而非先做 CT | 高可信 |
| 109 | 2 | Follicular thyroid carcinoma 的 capsular/vascular invasion | 高可信 |
| 110 | 2 | Lead-time bias 应比较 mortality，而非 diagnosis 后 survival | 高可信 |
| 127 | 1 | 性传播感染导致感染性关节炎，而非 HLA-B27 reactive arthritis | 部分可信 |
| 144 | 2 | 高度疑似 PE 时应开始 anticoagulation，而非等待低价值检查 | 高可信 |
| 145 | 2 | 非酒精戒断性 delirium：antipsychotic 优于 benzodiazepine | 高可信 |
| 151 | 2 | UC+pANCA+串珠样胆管提示 PSC 的 onion-skin fibrosis | 高可信 |
| 159 | 1 | Lymphoma 相关 nephrotic syndrome 指向 minimal change disease | 部分可信 |
| 168 | 2 | Pyloric stenosis 对应超声下 elongated/thickened pylorus | 高可信 |
| 170 | 1 | Rubella 既往免疫，当前皮疹需检查 parvovirus B19 | 高可信 |
| 206 | 1 | Salicylate 的低 PCO2、低 HCO3 混合酸碱紊乱 | 部分可信 |
| 244 | 1 | 未治疗 PID 与 Chlamydia/ectopic pregnancy 的关联 | 部分可信 |
| 251 | 1 | 无完整 mania、长期亚阈值双相症状符合 cyclothymia | 高可信 |
| 260 | 1 | 家族史、骨龄延迟和青春期启动符合 constitutional delay | 高可信 |
| 279 | 2 | ASD 男性患病/诊断率高，attention 不是核心定义 | 高可信 |
| 289 | 2 | NF2/merlin 与 meningioma；一条正确路径误称 TSC，另一条正确识别 NF2 | 混合质量 |
| 297 | 2 | Tamoxifen 的 thromboembolism/DVT 风险 | 高可信 |

## 4. 为什么正确答案只出现一到两次

### 4.1 主因：相邻知识模式之间的低概率检索

多数题并不是五条路径随机猜五个标签。26/32 题只有两个不同答案。
正确和错误路径通常同意大方向，但对一个决定性事实给出相反判断，例如：

- local vs systemic antifungal；
- 9 vs 12 月龄 milestone；
- concentration vs dilution defect；
- PSC vs PBC；
- mortality vs survival；
- treatment now vs diagnostic confirmation。

因此每道困难题内部更像存在两个语义模式。错误模式的条件概率高于 0.5，
五次同分布采样自然产生 `4-1` 或 `3-2`，正确知识只能偶尔被激活。

### 4.2 错误路径通常是连贯误区，不是随机噪声

主导错误路径往往形成完整但错误的因果故事。例如把 CLL 的 smudge cells 错配到其他疾病、
把 pyloric stenosis 错配到 triple-bubble sign，或把 NF2 错配到 vascular malformation。
同一个模型、同一个问题和同一种提示使这些高先验误区被多次重现。

这解释了为什么“错误共识”会比一条正确路径更稳定：一致性测量的是模型模式的概率质量，
不直接测量事实正确性。

### 4.3 StateBridge 链条会放大早期选中的模式

Planner、Critic、Refiner、Judger 都使用独立采样，但后续 agent 接收前一 agent 的 latent prefix。
一旦某条路径在早期选择了错误疾病或错误鉴别点，后续 agent 经常围绕该前提补充理由，
而不是重新独立求解。

对可用正则可靠提取到 Refiner 明确答案的 79 条路径，69 条（87.3%）与最终 Judger 答案一致。
该数字只覆盖 160 条审计路径中的一部分，但支持“Refiner 阶段通常已经锁定最终模式”的判断。

### 4.4 少量 Oracle 命中确实不可靠

Item 89 是最清楚的例子：正确分支最终选到 autosomal dominant，
但生成文本将核心病因误判为 PAVM/HHT。Item 289 的两条正确路径中也有一条误称 TSC，
另一条才正确识别 NF2。因此 Oracle@5 是候选覆盖上限，不等于 32 题都存在同等质量的可选证据。

## 5. Latent state 证据

使用每个阶段保存的最后 64 个 transfer hidden states，按 token 均值池化后计算 cosine distance。

| 阶段 | Correct-Wrong | Wrong-Wrong | Correct-Correct | 正确路径比错误簇更离群的题数 |
|---|---:|---:|---:|---:|
| Planner | 0.1012 | 0.0958 | 0.1044 | 23/32 |
| Critic | 0.1430 | 0.1420 | 0.1244 | 13/32 |
| Refiner | 0.1475 | 0.1383 | 0.1149 | 19/32 |

`Correct-Correct` 只在有两条正确路径的 15 题上定义。值得注意的是：

- 两条正确路径在 Planner 阶段并不相似；
- 到 Critic/Refiner 阶段，正确路径之间的距离低于 Correct-Wrong；
- Refiner 阶段形成 `Correct-Correct < Wrong-Wrong < Correct-Wrong` 的弱聚类结构。

这与“正确答案对应一个真实但低概率的语义模式”一致，而不是纯标签巧合。
但距离差异较小，而且本分析使用了 gold 划分，不能直接作为无监督 selector 的有效性证明。

单正确票的 17 题中，正确路径在 Planner state 上比错误簇更离群的有 13 题；
这进一步支持不少正确路径从最初计划阶段就进入了不同的知识模式。

## 6. LLM Judge 为什么只找回 18/32

候选均衡 Judge 在单正确票题中找回 9/17，在双正确票题中找回 9/15。
增加一条正确路径只带来有限帮助，说明失败不只是正确理由数量不足。

当前 Judge 每个答案只看到一条确定性选择的代表理由。这样隐藏了票数，但存在两个问题：

1. 同答案的多条理由可能质量不同；Item 289 就同时包含一条可靠和一条明显错误的正确答案理由。
2. Judge 与生成分支使用同一个 Qwen3-4B，可能共享完全相同的医学误区，无法可靠识别错误共识。

因此 Judge 能识别部分关键鉴别点，却也会推翻原本正确的多数答案。

## 7. 下一步实验

### A. Candidate-wise verification

不要一次比较所有候选。分别对每个候选提出相同问题：

```text
假设候选 X 是答案，列出支持证据、必要条件、最强反证和事实冲突。
```

随后使用独立仲裁器比较结构化验证结果。隐藏票数、随机候选顺序，并在独立 dev set 校准推翻阈值。

### B. 保留同答案内的多条证据

对同一答案的所有支持路径做等预算、去频率的 evidence synthesis，避免随机代表路径丢掉高质量理由。
需要与“一答案一理由”严格对照。

### C. 检验 latent self-reinforcement

对 32 题做 evaluation-only ablation：最终 Judger 分别接收 Refiner latent prefix、零 prefix、
以及其他分支 prefix。比较答案是否随 prefix 系统性改变，以验证通信是在传递有效鉴别点还是放大早期误区。

### D. Latent trajectory selector

在独立 dev split 上训练只读 Planner/Critic/Refiner state 的候选级 selector，使用：

- 候选内跨路径收敛；
- Planner 到 Refiner 的稳定性；
- 候选间分离度；
- 结构化 verifier 分数。

测试集只做一次冻结评估，不能用当前 300 题的 gold 调权重。

## 8. 结论

正确答案只出现一次或两次的首要原因不是纯随机，而是 Qwen3-4B 在困难题上存在低概率的正确知识模式
和高概率的连贯错误模式。随机 seed 偶尔激活正确鉴别点，但相同提示和 StateBridge 链条会让更高概率的
错误前提被多次自洽强化。下一阶段应从“按出现次数选答案”转向“按候选独立验证关键鉴别点”，
同时利用 Refiner latent state 中已经出现的弱聚类结构。
