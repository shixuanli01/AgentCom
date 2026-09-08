# StateBridge Qwen3-4B MedQA Analysis

## Run identity

- Method: released StateBridge implementation at commit
  `3f6bf5442c6e8848555a6132516e6d36f35444fb`
- Model: `Qwen/Qwen3-4B`, local revision
  `1cfa9a7208912126459214e8b04321603b3df60c`
- Hardware: one NVIDIA RTX 5090
- Seed: 42
- Protocol: sequential Planner, Critic, Refiner, and Judger; temperature 0.6;
  top-p 0.95; prefix length 64; snap ratio 0.3; prefix scale 1.0
- Result: `reproduction/runs/main_seed42/medqa.json`

## Primary result

| Result | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Local released-code reproduction | 210 | 300 | 70.0% |
| Public paper target | 211 (implied) | 300 | 70.3% |

The local result differs from the reported cell by one item, or 0.3 percentage
points. The Wilson 95% interval for the local accuracy is 64.6% to 74.9%, which
contains the paper value. This is a strong released-code reproduction match,
but it is not an exact stochastic replication because the paper does not
publish its seed and used two A100-80G GPUs.

## Integrity checks

- All 300 expected rows are present with contiguous indices.
- All questions are unique.
- Every row contains all four agent traces.
- No CUDA, worker, timeout, or evaluation errors occurred.
- The two released answer parsers agree on all 300 rows.
- All 300 Judger responses contain a boxed answer; 299 contain a valid A-D
  prediction and one contains a boxed out-of-set answer.

## Confusion matrix

Rows are gold labels and columns are predictions.

| Gold | A | B | C | D | Other | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A | 59 | 7 | 10 | 8 | 1 | 69.4% |
| B | 9 | 44 | 4 | 9 | 0 | 66.7% |
| C | 7 | 6 | 73 | 7 | 0 | 78.5% |
| D | 6 | 8 | 8 | 34 | 0 | 60.7% |

Gold counts are A=85, B=66, C=93, and D=56. Prediction counts are A=81,
B=65, C=95, D=58, and Other=1. The close marginal distributions provide no
evidence of a simple global answer-letter collapse. Accuracy differences by
gold letter are visible, especially C versus D, but a four-group chi-square
test is not significant at 0.05 (`p=0.119`).

## Efficiency

| Measure per item | Mean | Median | P90 | Maximum |
|---|---:|---:|---:|---:|
| Wall time | 54.1 s | 46.5 s | 82.1 s | 191.4 s |
| Prompt tokens, four agents | 1,423.6 | 1,351 | 1,893 | 4,253 |
| Generated tokens, four agents | 2,404.1 | 1,874 | 4,393 | 10,073 |
| Prefix tokens received | 190.6 | 192 | 192 | 192 |
| Total recorded tokens | 4,018.3 | 3,559 | 6,106 | 12,090 |
| Alignment time | 2.91 s | 2.91 s | 2.95 s | 2.98 s |

The run processed 1,205,481 recorded tokens in total: 427,072 prompt, 721,223
generated, and 57,186 continuous-prefix positions. Generation accounts for
most runtime. Generated-token count and wall time correlate at `r=0.977`,
whereas prompt-token count and wall time correlate at only `r=0.182`.

Planner generation is the largest component, averaging 886 tokens per item.
Critic, Refiner, and Judger average 471, 507, and 540 generated tokens,
respectively. Alignment contributes about 2.9 seconds per item and roughly
5.4% of end-to-end wall time.

## Error diagnostics

Correct items average 2,042 generated tokens and 48.9 seconds. Incorrect items
average 3,249 generated tokens and 66.3 seconds. Accuracy by generated-token
quartile falls from 88.0%, to 78.7%, to 66.7%, to 46.7%. This must not be read
causally as evidence that longer reasoning creates errors: difficult or
unstable questions can cause both longer generation and lower accuracy. It is
nevertheless a useful online warning signal for looping or indecision.

Input length is much less explanatory. Accuracy across prompt-token quartiles
is 73.3%, 70.7%, 66.7%, and 69.3%, with no comparable monotonic collapse.

Two concrete anomalies were found:

1. Item 9 asks which receptor maraviroc affects but offers gp120, gp160, p24,
   and reverse transcriptase. The Judger correctly states that maraviroc
   targets CCR5 and outputs `None of the options`; the dataset marks A/gp120.
   NIH defines CCR5 antagonists as blocking the CCR5 coreceptor. This item is
   malformed or at least materially ambiguous and should be flagged, not
   silently relabeled in the official score.
2. Item 24 reaches the 8,192-token Judger limit without EOS and repeats a
   contradictory C/D loop. It is the only non-EOS agent generation and is a
   genuine decoding-degeneration failure.

If item 9 were credited for medical correctness, the diagnostic accuracy would
be 211/300 = 70.33%, numerically equal to the rounded public target. The
authoritative benchmark score remains 210/300 = 70.0%; this coincidence does
not establish that the paper treated the same item differently.

## What this result establishes

This run provides strong evidence that the released StateBridge implementation
can reproduce the paper's Qwen3-4B MedQA absolute score under disclosed local
conditions. A one-item gap is smaller than ordinary sampling uncertainty and
is especially credible given the unpublished author seed and different GPU
topology.

This result alone does not establish that latent communication caused the
score. That claim requires matched controls under this exact model, prompt,
sampling, and evaluator: no-message, random-prefix, same-family wrong-message,
and ideally text-plan or single-agent conditions. The current run establishes
the target operating point for those causal comparisons.

## External check

- NIH CCR5 antagonist glossary:
  https://clinicalinfo.hiv.gov/en/glossary/ccr5-antagonist

