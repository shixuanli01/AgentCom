# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 25.42% | 14.04% | 96.49% | 3.51% | 2.31% | 91.67% | 26 | 8 | 9.89% |
| true_latentmas | 25.14% | 61.40% | 50.88% | 49.12% | 0.93% | 95.83% | 74 | 58 | 27.82% |
| true_statebridge | 26.84% | 64.04% | 51.75% | 48.25% | 2.55% | 97.92% | 84 | 56 | 29.94% |
| true_text | 27.26% | 74.56% | 47.37% | 52.63% | 1.39% | 100.00% | 91 | 60 | 32.63% |

## Causal comparisons

- text_ce: ΔAcc=1.84%, rescues=78, destructions=65.
- statebridge_ce: ΔAcc=1.41%, rescues=69, destructions=59.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
