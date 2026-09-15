# ICR Communication-First Step 1

`seed_pair_01` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 71.33% | 84.38% | 34.38% | 59.38% | 59.38% | 83.87% | 67.74% | 16.13% |
| StateBridge | 71.17% | 93.75% | 21.88% | 57.81% | 57.81% | 93.55% | 80.65% | 12.90% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 0.17% | [-1.17%, 1.50%] |
| cr | -9.38% | [-23.53%, 3.45%] |
| pr | 12.50% | [-7.14%, 31.25%] |
| si | 1.56% | [-10.61%, 13.64%] |
| sra | 1.56% | [-10.61%, 13.64%] |
| fcs | -9.68% | [-24.14%, 3.57%] |
| fws | -12.90% | [-32.14%, 7.41%] |
| follow_selectivity | 3.23% | [-21.88%, 28.12%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 80/600 answers (13.33%, 95% CI [9.83%, 17.00%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 0.83%, 95% CI [-0.33%, 2.17%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=93.55%, FWS=80.65%, FollowSelectivity=12.90%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction/destruction subsets each contain only 35 directional cases, so cross-seed replication remains necessary.
