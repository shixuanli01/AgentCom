# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 72.50% | 5.71% | 97.14% | 2.86% | 0.76% | 100.00% | 3 | 1 | 1.33% |
| other_statebridge | 72.33% | 0.00% | 97.14% | 2.86% | 2.27% | 99.75% | 3 | 2 | 1.33% |
| other_text | 71.67% | 2.86% | 94.29% | 5.71% | 0.76% | 99.25% | 2 | 5 | 2.17% |
| self_statebridge | 72.50% | 2.86% | 100.00% | 0.00% | 0.76% | 100.00% | 2 | 0 | 0.67% |
| self_text | 72.17% | 0.00% | 97.14% | 2.86% | 0.76% | 100.00% | 1 | 1 | 0.50% |
| true_statebridge | 71.67% | 82.86% | 5.71% | 94.29% | 0.76% | 100.00% | 30 | 33 | 14.00% |
| true_text | 74.33% | 91.43% | 42.86% | 57.14% | 0.76% | 100.00% | 33 | 20 | 12.33% |

## Causal comparisons

- text_ce: ΔAcc=1.83%, rescues=31, destructions=20.
- text_esv: ΔAcc=2.67%, rescues=37, destructions=21.
- text_oav: ΔAcc=2.17%, rescues=33, destructions=20.
- statebridge_ce: ΔAcc=-0.83%, rescues=28, destructions=33.
- statebridge_esv: ΔAcc=-0.67%, rescues=31, destructions=35.
- statebridge_oav: ΔAcc=-0.83%, rescues=29, destructions=34.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
