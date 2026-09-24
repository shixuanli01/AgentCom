# Answer Only baseline — four datasets

Produced 2026-09-23. Sender transmits only its normalised answer string; no
reasoning, no evidence, no latent payload. Unparseable sender answers are sent
as the literal `UNPARSEABLE`.

## Provenance

| Field | Value |
|---|---|
| Condition key | `true_answer` (`AnswerOnlyCommunicationChannel`, `icr/channels.py`) |
| Roots | `runs/medqa/cr_dnc_v1/icr_root`, `runs/{gpqa,arc_challenge,gsm8k}/hybrid_root` |
| Beliefs | symlink to `artifacts/icr_v3/<dataset>_full_seed42`, unmodified |
| replication_id | `seed_pair_00` — matches the four published channels |
| Prompt | `icr_v3_mid_injection` (V3), receiver policy `revise` |
| Model | Qwen/Qwen3-4B, temperature 0.6, top_p 0.95, max_new_tokens 16384 |
| Failures | 0 workers, 0 incomplete records |

## Scope — read this before quoting any number

Run with `--pair-classes correction_opportunity,destruction_risk`, i.e. **only
the mixed-correctness directions**. Both-wrong and both-correct directions were
not generated for this condition.

Consequences, which must be carried into the paper:

- **SR, SCR, Acc_ret and full-set accuracy are NOT MEASURED** for Answer Only.
  They are not zero and not unknown-but-estimable from these records; the
  underlying directions do not exist. Do not fill them with extrapolated values.
- CR, PR, SI, and the adoption probabilities are complete and directly
  comparable to the four published channels, which share the same beliefs,
  the same `revision_seed`, and the same mixed-direction strata.

| Dataset | CR stratum | PR stratum | Records |
|---|---:|---:|---:|
| MedQA | 116 | 116 | 232 |
| ARC-C | 98 | 98 | 196 |
| GSM8K | 92 | 92 | 184 |
| GPQA-D | 114 | 114 | 228 |

## Results

`Keep Initial` is the analytic reference (receiver never changes): CR 0, PR 100,
SI 50 by construction, zero GPU cost.

| Dataset | Channel | CR | PR | SI | SI−50 |
|---|---|---:|---:|---:|---:|
| MedQA | Keep Initial | 0.00% | 100.00% | 50.00% | +0.00 |
| MedQA | No Message | 6.90% | 98.28% | 52.59% | +2.59 |
| MedQA | **Answer Only** | 38.79% | 80.17% | 59.48% | +9.48 |
| MedQA | Full Text | 80.17% | 42.24% | 61.21% | +11.21 |
| MedQA | StateBridge | 57.76% | 75.00% | 66.38% | +16.38 |
| MedQA | LatentMAS | 69.83% | 32.76% | 51.29% | +1.29 |
| ARC-C | No Message | 8.16% | 92.86% | 50.51% | +0.51 |
| ARC-C | **Answer Only** | 38.78% | 63.27% | 51.02% | +1.02 |
| ARC-C | Full Text | 80.61% | 34.69% | 57.65% | +7.65 |
| ARC-C | StateBridge | 69.39% | 44.90% | 57.14% | +7.14 |
| ARC-C | LatentMAS | 58.16% | 45.92% | 52.04% | +2.04 |
| GSM8K | No Message | 13.04% | 92.39% | 52.72% | +2.72 |
| GSM8K | **Answer Only** | 39.13% | 70.65% | 54.89% | +4.89 |
| GSM8K | Full Text | 51.09% | 52.17% | 51.63% | +1.63 |
| GSM8K | StateBridge | 39.13% | 73.91% | 56.52% | +6.52 |
| GSM8K | LatentMAS | 59.78% | 46.74% | 53.26% | +3.26 |
| GPQA-D | No Message | 14.04% | 96.49% | 55.26% | +5.26 |
| GPQA-D | **Answer Only** | 56.14% | 63.16% | 59.65% | +9.65 |
| GPQA-D | Full Text | 74.56% | 47.37% | 60.96% | +10.96 |
| GPQA-D | StateBridge | 64.04% | 51.75% | 57.89% | +7.89 |
| GPQA-D | LatentMAS | 61.40% | 50.88% | 56.14% | +6.14 |

## Adoption decomposition

`pi_CR` = P(receiver adopts sender's answer | CR stratum); `pi_PR` likewise on
the PR stratum. Adoption rate `pi_bar` = (pi_CR + pi_PR)/2 is how often the
message moves the receiver at all; discrimination `dpi` = pi_CR - pi_PR is how
much of that movement is selective.

| Dataset | Channel | pi_CR | pi_PR | pi_bar | dpi |
|---|---|---:|---:|---:|---:|
| MedQA | No Message | 6.9% | 0.9% | 3.9% | 6.0 |
| MedQA | **Answer Only** | 38.8% | 19.8% | 29.3% | 19.0 |
| MedQA | Full Text | 80.2% | 57.8% | 69.0% | 22.4 |
| MedQA | StateBridge | 57.8% | 25.0% | 41.4% | 32.8 |
| MedQA | LatentMAS | 69.8% | 67.2% | 68.5% | 2.6 |
| ARC-C | No Message | 8.2% | 7.1% | 7.7% | 1.0 |
| ARC-C | **Answer Only** | 38.8% | 36.7% | 37.8% | 2.0 |
| ARC-C | Full Text | 80.6% | 65.3% | 73.0% | 15.3 |
| ARC-C | StateBridge | 69.4% | 53.1% | 61.2% | 16.3 |
| ARC-C | LatentMAS | 58.2% | 54.1% | 56.1% | 4.1 |
| GSM8K | No Message | 13.0% | 5.4% | 9.2% | 7.6 |
| GSM8K | **Answer Only** | 39.1% | 25.0% | 32.1% | 14.1 |
| GSM8K | Full Text | 51.1% | 46.7% | 48.9% | 4.3 |
| GSM8K | StateBridge | 39.1% | 23.9% | 31.5% | 15.2 |
| GSM8K | LatentMAS | 59.8% | 52.2% | 56.0% | 7.6 |
| GPQA-D | No Message | 14.0% | 0.9% | 7.5% | 13.2 |
| GPQA-D | **Answer Only** | 56.1% | 33.3% | 44.7% | 22.8 |
| GPQA-D | Full Text | 74.6% | 51.8% | 63.2% | 22.8 |
| GPQA-D | StateBridge | 64.0% | 46.5% | 55.3% | 17.5 |
| GPQA-D | LatentMAS | 61.4% | 49.1% | 55.3% | 12.3 |

`SI - 50` and `dpi/2` agree to within the rate at which the receiver moves to a
third answer (neither its own nor the sender's), which is why the two columns
differ slightly on ARC-C StateBridge (+7.14 vs +8.16) and GSM8K Full Text
(+1.63 vs +2.17). The identity is exact only in the two-outcome case.

## What this does and does not establish

Establishes: reasoning text raises the adoption rate on every dataset, roughly
doubling it against the bare answer (MedQA 29.3 -> 69.0, ARC-C 37.8 -> 73.0,
GSM8K 32.1 -> 48.9, GPQA-D 44.7 -> 63.2), while its effect on discrimination has
no consistent sign: equal on GPQA-D (22.8 / 22.8), higher on ARC-C (2.0 -> 15.3)
and MedQA (19.0 -> 22.4), and lower on GSM8K (14.1 -> 4.3).

Does not establish that the receiver ignores the reasoning. The receiver plainly
responds to it — adoption changes sharply. What these numbers constrain is the
joint behaviour of the sender's text and the receiver's acceptance rule under
the ICR revision prompt, on the mixed-correctness strata of these four datasets,
at one model and one sampling seed. A claim about reading behaviour would need
evidence about the receiver's processing, which this design does not collect.
