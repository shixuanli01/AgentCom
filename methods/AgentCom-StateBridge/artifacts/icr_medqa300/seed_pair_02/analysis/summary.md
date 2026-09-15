# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 71.67% | 0.00% | 100.00% | 0.00% | 2.17% | 99.75% | 3 | 1 | 1.00% |
| other_statebridge | 71.83% | 2.94% | 94.12% | 5.88% | 3.62% | 99.75% | 6 | 3 | 2.17% |
| other_text | 70.33% | 2.94% | 100.00% | 0.00% | 0.00% | 98.22% | 1 | 7 | 2.00% |
| self_statebridge | 71.33% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0 | 0 | 0.33% |
| self_text | 71.50% | 0.00% | 100.00% | 0.00% | 0.72% | 100.00% | 1 | 0 | 0.33% |
| true_statebridge | 70.33% | 79.41% | 8.82% | 91.18% | 0.00% | 99.49% | 27 | 33 | 13.00% |
| true_text | 72.50% | 85.29% | 35.29% | 64.71% | 0.00% | 100.00% | 29 | 22 | 11.17% |

## Causal comparisons

- text_ce: ΔAcc=0.83%, rescues=30, destructions=25.
- text_esv: ΔAcc=2.17%, rescues=35, destructions=22.
- text_oav: ΔAcc=1.00%, rescues=29, destructions=23.
- statebridge_ce: ΔAcc=-1.33%, rescues=28, destructions=36.
- statebridge_esv: ΔAcc=-1.50%, rescues=27, destructions=36.
- statebridge_oav: ΔAcc=-1.00%, rescues=27, destructions=33.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
