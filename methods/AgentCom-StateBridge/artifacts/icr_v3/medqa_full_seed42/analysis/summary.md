# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 25.69% | 6.90% | 98.28% | 1.72% | 2.52% | 100.00% | 19 | 2 | 3.75% |
| true_latentmas | 24.72% | 69.83% | 32.76% | 67.24% | 1.61% | 100.00% | 88 | 78 | 30.14% |
| true_statebridge | 28.89% | 57.76% | 75.00% | 25.00% | 0.46% | 100.00% | 69 | 29 | 18.75% |
| true_text | 27.22% | 80.17% | 42.24% | 57.76% | 0.46% | 100.00% | 95 | 67 | 31.11% |

## Causal comparisons

- text_ce: ΔAcc=1.53%, rescues=86, destructions=75.
- statebridge_ce: ΔAcc=3.19%, rescues=61, destructions=38.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
