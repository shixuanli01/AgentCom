# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 69.59% | 54.10% | 98.36% | 1.64% | 25.00% | 100.00% | 73 | 1 | 22.24% |
| true_latentmas | 65.71% | 77.05% | 62.30% | 37.70% | 18.75% | 99.52% | 77 | 24 | 28.78% |
| true_statebridge | 69.59% | 90.16% | 77.05% | 22.95% | 19.38% | 100.00% | 86 | 14 | 26.73% |
| true_text | 70.82% | 98.36% | 77.05% | 22.95% | 20.00% | 100.00% | 92 | 14 | 28.16% |

## Causal comparisons

- text_ce: ΔAcc=1.22%, rescues=29, destructions=23.
- statebridge_ce: ΔAcc=0.00%, rescues=26, destructions=26.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
