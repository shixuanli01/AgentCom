# ICR Communication-First Step 1

Frozen `seed_pair_00` was analyzed without new model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 74.33% | 91.43% | 42.86% | 67.14% | 67.14% | 91.43% | 57.14% | 34.29% |
| StateBridge | 71.67% | 82.86% | 5.71% | 44.29% | 44.29% | 82.86% | 94.29% | -11.43% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 2.67% | [1.17%, 4.33%] |
| cr | 8.57% | [-6.06%, 23.33%] |
| pr | 37.14% | [21.43%, 53.85%] |
| si | 22.86% | [11.11%, 35.29%] |
| sra | 22.86% | [11.11%, 35.29%] |
| fcs | 8.57% | [-6.06%, 23.33%] |
| fws | -37.14% | [-53.85%, -21.43%] |
| follow_selectivity | 45.71% | [22.22%, 70.59%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 83/600 answers (13.83%, 95% CI [10.33%, 17.67%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is -0.83%, 95% CI [-1.67%, 0.00%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=82.86%, FWS=94.29%, FollowSelectivity=-11.43%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction/destruction subsets each contain only 35 directional cases, so cross-seed replication remains necessary.
