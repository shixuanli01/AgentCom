# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 27.89% | 8.16% | 92.86% | 7.14% | 1.52% | 93.75% | 13 | 11 | 4.93% |
| true_latentmas | 28.23% | 58.16% | 45.92% | 54.08% | 0.91% | 95.31% | 60 | 56 | 21.60% |
| true_statebridge | 29.93% | 69.39% | 44.90% | 55.10% | 0.30% | 98.44% | 69 | 55 | 22.62% |
| true_text | 30.27% | 80.61% | 34.69% | 65.31% | 0.30% | 100.00% | 80 | 64 | 27.04% |

## Causal comparisons

- text_ce: ΔAcc=2.38%, rescues=76, destructions=62.
- statebridge_ce: ΔAcc=2.04%, rescues=64, destructions=52.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
