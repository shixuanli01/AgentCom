# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 31.68% | 2.50% | 97.50% | 2.50% | 3.47% | 100.00% | 6 | 1 | 3.44% |
| true_latentmas | 30.92% | 75.00% | 27.50% | 72.50% | 1.39% | 100.00% | 32 | 29 | 29.77% |
| true_statebridge | 31.68% | 47.50% | 62.50% | 37.50% | 0.69% | 100.00% | 20 | 15 | 17.18% |
| true_text | 31.30% | 75.00% | 32.50% | 67.50% | 0.69% | 100.00% | 31 | 27 | 30.15% |

## Causal comparisons

- text_ce: ΔAcc=-0.38%, rescues=29, destructions=30.
- statebridge_ce: ΔAcc=0.00%, rescues=19, destructions=19.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
