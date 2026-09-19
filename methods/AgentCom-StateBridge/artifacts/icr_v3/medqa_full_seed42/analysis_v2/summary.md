# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 69.67% | 89.29% | 50.00% | 69.64% | 69.64% | 88.46% | 53.85% | 34.62% |
| StateBridge | 69.67% | 64.29% | 78.57% | 71.43% | 71.43% | 61.54% | 23.08% | 38.46% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 0.00% | [-1.17%, 1.17%] |
| cr | 25.00% | [9.52%, 42.31%] |
| pr | -28.57% | [-48.39%, -8.70%] |
| si | -1.79% | [-14.71%, 11.12%] |
| sra | -1.79% | [-14.71%, 11.12%] |
| fcs | 26.92% | [10.53%, 45.45%] |
| fws | 30.77% | [9.52%, 52.17%] |
| follow_selectivity | -3.85% | [-31.58%, 24.00%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 28/600 answers (4.67%, 95% CI [3.00%, 6.50%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 0.50%, 95% CI [-0.83%, 2.00%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=61.54%, FWS=23.08%, FollowSelectivity=38.46%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 28 directional cases and the destruction subset contains 28, so cross-seed replication remains necessary.
