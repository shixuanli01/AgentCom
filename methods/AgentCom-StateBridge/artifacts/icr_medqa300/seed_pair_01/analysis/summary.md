# ICR Analysis

Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.

| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 70.33% | 3.12% | 93.75% | 6.25% | 1.37% | 99.74% | 3 | 3 | 1.33% |
| other_latentmas | 70.50% | 3.12% | 96.88% | 3.12% | 0.68% | 100.00% | 2 | 1 | 0.67% |
| other_statebridge | 70.00% | 6.25% | 90.62% | 9.38% | 2.74% | 98.72% | 6 | 8 | 2.50% |
| other_text | 69.67% | 0.00% | 100.00% | 0.00% | 1.37% | 98.46% | 2 | 6 | 1.67% |
| self_latentmas | 70.00% | 0.00% | 96.88% | 3.12% | 0.00% | 99.74% | 0 | 2 | 0.50% |
| self_statebridge | 70.50% | 3.12% | 100.00% | 0.00% | 0.00% | 100.00% | 1 | 0 | 0.33% |
| self_text | 70.50% | 3.12% | 100.00% | 0.00% | 0.00% | 100.00% | 1 | 0 | 0.17% |
| true_latentmas | 70.50% | 31.25% | 71.88% | 28.12% | 0.68% | 99.74% | 11 | 10 | 5.33% |
| true_statebridge | 71.17% | 93.75% | 21.88% | 78.12% | 0.00% | 100.00% | 30 | 25 | 13.17% |
| true_text | 71.33% | 84.38% | 34.38% | 65.62% | 0.00% | 100.00% | 27 | 21 | 11.33% |

## Causal comparisons

- text_ce: ΔAcc=1.00%, rescues=27, destructions=21.
- text_esv: ΔAcc=1.67%, rescues=33, destructions=23.
- text_oav: ΔAcc=0.83%, rescues=26, destructions=21.
- statebridge_ce: ΔAcc=0.83%, rescues=30, destructions=25.
- statebridge_esv: ΔAcc=1.17%, rescues=33, destructions=26.
- statebridge_oav: ΔAcc=0.67%, rescues=29, destructions=25.

Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.
