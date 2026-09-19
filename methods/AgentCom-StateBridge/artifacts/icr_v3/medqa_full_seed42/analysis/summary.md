# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 69.17% | 21.43% | 100.00% | 0.00% | 1.81% | 100.00% | 9 | 0 | 1.83% |
| true_latentmas | 67.50% | 71.43% | 28.57% | 71.43% | 1.81% | 98.94% | 23 | 24 | 10.83% |
| true_statebridge | 69.67% | 64.29% | 78.57% | 21.43% | 0.00% | 100.00% | 18 | 6 | 5.33% |
| true_text | 69.67% | 89.29% | 50.00% | 50.00% | 0.60% | 100.00% | 26 | 14 | 9.00% |

## Causal comparisons

- text_ce: ΔAcc=0.50%, rescues=20, destructions=17.
- statebridge_ce: ΔAcc=0.50%, rescues=12, destructions=9.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
