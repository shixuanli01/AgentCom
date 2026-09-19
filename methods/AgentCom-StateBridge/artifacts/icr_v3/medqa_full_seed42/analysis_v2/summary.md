# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 31.68% | 2.50% | 97.50% | 50.00% | 50.00% | 2.50% | 0.00% | 2.50% |
| Full Text | 31.30% | 75.00% | 32.50% | 53.75% | 53.75% | 75.00% | 67.50% | 7.50% |
| StateBridge | 31.68% | 47.50% | 62.50% | 55.00% | 55.00% | 47.50% | 37.50% | 10.00% |
| LatentMAS | 30.92% | 75.00% | 27.50% | 51.25% | 51.25% | 75.00% | 72.50% | 2.50% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | -0.38% | [-4.17%, 3.31%] |
| cr | 27.50% | [8.82%, 46.15%] |
| pr | -30.00% | [-47.37%, -12.82%] |
| si | -1.25% | [-13.33%, 10.38%] |
| sra | -1.25% | [-13.33%, 10.38%] |
| fcs | 27.50% | [8.82%, 46.15%] |
| fws | 30.00% | [12.82%, 47.37%] |
| follow_selectivity | -2.50% | [-26.67%, 20.76%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 50/262 answers (19.08%, 95% CI [13.78%, 24.60%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 0.00%, 95% CI [-3.88%, 4.07%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=47.50%, FWS=37.50%, FollowSelectivity=10.00%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 40 directional cases and the destruction subset contains 40, so cross-seed replication remains necessary.
