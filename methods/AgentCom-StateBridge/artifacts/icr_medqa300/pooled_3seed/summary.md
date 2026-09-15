# ICR MedQA300 — pooled 3-seed analysis

Pooled 3 seed pairs and 12,600 directional-condition records. Confidence intervals resample item IDs and retain all replications and both directions.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 71.50% | 2.97% | 97.03% | 50.00% | 50.00% | 3.00% | 3.00% | 0.00% |
| Text | 72.72% | 87.13% | 37.62% | 62.38% | 62.38% | 87.00% | 63.00% | 24.00% |
| StateBridge | 71.06% | 85.15% | 11.88% | 48.51% | 48.51% | 85.00% | 89.00% | -4.00% |

## Text − StateBridge

- accuracy: 1.67%, 95% CI [0.83%, 2.50%]
- cr: 1.98%, 95% CI [-7.63%, 11.46%]
- pr: 25.74%, 95% CI [16.04%, 35.63%]
- si: 13.86%, 95% CI [6.84%, 20.98%]
- sra: 13.86%, 95% CI [6.84%, 20.98%]
- fcs: 2.00%, 95% CI [-7.69%, 11.57%]
- fws: -26.00%, 95% CI [-36.11%, -16.16%]
- follow_selectivity: 28.00%, 95% CI [13.79%, 42.42%]

## Robustness checks

- StateBridge CR > PR in every seed pair: True
- Text PR > StateBridge PR in every seed pair: True
