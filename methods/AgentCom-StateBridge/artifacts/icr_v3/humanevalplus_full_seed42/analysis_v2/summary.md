# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 42.26% | 57.14% | 92.86% | 75.00% | 75.00% | 0.00% | 0.00% | 0.00% |
| Full Text | 45.24% | 75.00% | 92.86% | 83.93% | 83.93% | 21.05% | 0.00% | 21.05% |
| StateBridge | 43.45% | 67.86% | 96.43% | 82.14% | 82.14% | 5.26% | 5.26% | 0.00% |
| LatentMAS | 39.88% | 85.71% | 60.71% | 73.21% | 73.21% | 26.32% | 42.11% | -15.79% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 1.79% | [-3.70%, 7.64%] |
| cr | 7.14% | [-15.62%, 30.00%] |
| pr | -3.57% | [-16.67%, 9.09%] |
| si | 1.79% | [-12.50%, 16.07%] |
| sra | 1.79% | [-12.50%, 16.07%] |
| fcs | 15.79% | [0.00%, 40.00%] |
| fws | -5.26% | [-17.65%, 0.00%] |
| follow_selectivity | 21.05% | [0.00%, 46.15%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 68/168 answers (40.48%, 95% CI [29.91%, 51.39%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 1.19%, 95% CI [-3.97%, 5.33%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=5.26%, FWS=5.26%, FollowSelectivity=0.00%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 28 directional cases and the destruction subset contains 28, so cross-seed replication remains necessary.
