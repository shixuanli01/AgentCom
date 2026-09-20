# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 42.26% | 57.14% | 92.86% | 7.14% | 9.78% | 100.00% | 25 | 2 | 27.38% |
| true_latentmas | 39.88% | 85.71% | 60.71% | 39.29% | 6.52% | 100.00% | 30 | 11 | 44.05% |
| true_statebridge | 43.45% | 67.86% | 96.43% | 3.57% | 8.70% | 95.00% | 27 | 2 | 32.14% |
| true_text | 45.24% | 75.00% | 92.86% | 7.14% | 9.78% | 100.00% | 30 | 2 | 38.69% |

## Causal comparisons

- text_ce: ΔAcc=2.98%, rescues=13, destructions=8.
- statebridge_ce: ΔAcc=1.19%, rescues=11, destructions=9.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
