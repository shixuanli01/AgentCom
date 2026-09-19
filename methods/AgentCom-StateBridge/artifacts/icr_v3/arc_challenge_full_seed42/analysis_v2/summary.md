# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 69.59% | 54.10% | 98.36% | 76.23% | 76.23% | 7.14% | 7.14% | 0.00% |
| Full Text | 70.82% | 98.36% | 77.05% | 87.70% | 87.70% | 92.86% | 78.57% | 14.29% |
| StateBridge | 69.59% | 90.16% | 77.05% | 83.61% | 83.61% | 64.29% | 50.00% | 14.29% |
| LatentMAS | 65.71% | 77.05% | 62.30% | 69.67% | 69.67% | 42.86% | 78.57% | -35.71% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 1.22% | [-0.37%, 2.90%] |
| cr | 8.20% | [0.00%, 16.98%] |
| pr | 0.00% | [-6.67%, 6.49%] |
| si | 4.10% | [-0.79%, 9.02%] |
| sra | 4.10% | [-0.79%, 9.02%] |
| fcs | 28.57% | [0.00%, 60.00%] |
| fws | 28.57% | [6.67%, 53.85%] |
| follow_selectivity | 0.00% | [-30.00%, 30.00%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 57/490 answers (11.63%, 95% CI [8.46%, 14.96%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 0.00%, 95% CI [-2.28%, 2.24%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=64.29%, FWS=50.00%, FollowSelectivity=14.29%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 61 directional cases and the destruction subset contains 61, so cross-seed replication remains necessary.
