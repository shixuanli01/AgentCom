# ICR MedQA300 — 2-seed channel comparison

Complete pooled comparison over 12,000 directional-condition records. Confidence intervals use 10,000 item-cluster resamples and retain both directions and every included seed pair.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 71.42% | 4.48% | 95.52% | 50.00% | 50.00% | 4.55% | 4.55% | 0.00% |
| Text | 72.83% | 88.06% | 38.81% | 63.43% | 63.43% | 87.88% | 62.12% | 25.76% |
| StateBridge | 71.42% | 88.06% | 13.43% | 50.75% | 50.75% | 87.88% | 87.88% | 0.00% |
| LatentMAS | 71.17% | 31.34% | 67.16% | 49.25% | 49.25% | 30.30% | 33.33% | -3.03% |

## Pairwise channel differences

### text_minus_statebridge

- accuracy: 1.42%, 95% CI [0.42%, 2.42%]
- cr: 0.00%, 95% CI [-10.00%, 10.45%]
- pr: 25.37%, 95% CI [12.68%, 38.57%]
- si: 12.69%, 95% CI [4.23%, 21.64%]
- sra: 12.69%, 95% CI [4.23%, 21.64%]
- fcs: 0.00%, 95% CI [-10.00%, 10.61%]
- fws: -25.76%, 95% CI [-39.34%, -12.73%]
- follow_selectivity: 25.76%, 95% CI [8.47%, 44.00%]

### text_minus_latentmas

- accuracy: 1.67%, 95% CI [0.67%, 2.75%]
- cr: 56.72%, 95% CI [45.24%, 68.33%]
- pr: -28.36%, 95% CI [-42.86%, -12.90%]
- si: 14.18%, 95% CI [5.30%, 23.81%]
- sra: 14.18%, 95% CI [5.30%, 23.81%]
- fcs: 57.58%, 95% CI [45.83%, 69.23%]
- fws: 28.79%, 95% CI [13.11%, 43.48%]
- follow_selectivity: 28.79%, 95% CI [10.77%, 48.39%]

### statebridge_minus_latentmas

- accuracy: 0.25%, 95% CI [-0.67%, 1.17%]
- cr: 56.72%, 95% CI [45.45%, 67.86%]
- pr: -53.73%, 95% CI [-66.67%, -39.74%]
- si: 1.49%, 95% CI [-6.34%, 9.32%]
- sra: 1.49%, 95% CI [-6.34%, 9.32%]
- fcs: 57.58%, 95% CI [46.25%, 68.85%]
- fws: 54.55%, 95% CI [40.35%, 67.80%]
- follow_selectivity: 3.03%, 95% CI [-12.82%, 18.84%]

## Pattern evidence

- Classification: `B_directional_signature`
- StateBridge CR−PR: 74.63%
- LatentMAS CR−PR: -35.82%
- Criterion: A requires CR>PR for both latent channels; B requires CR>PR for StateBridge but not LatentMAS. This is a directional signature, not an unsupported high/low threshold claim.
