# ICR Communication-First Step 1

`seed_pair_02` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 72.50% | 85.29% | 35.29% | 60.29% | 60.29% | 85.29% | 64.71% | 20.59% |
| StateBridge | 70.33% | 79.41% | 8.82% | 44.12% | 44.12% | 79.41% | 91.18% | -11.76% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 2.17% | [0.67%, 3.83%] |
| cr | 5.88% | [-12.50%, 24.24%] |
| pr | 26.47% | [12.00%, 42.42%] |
| si | 16.18% | [4.05%, 28.12%] |
| sra | 16.18% | [4.05%, 28.12%] |
| fcs | 5.88% | [-12.50%, 24.24%] |
| fws | -26.47% | [-42.42%, -12.00%] |
| follow_selectivity | 32.35% | [8.11%, 56.25%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 82/600 answers (13.67%, 95% CI [10.17%, 17.33%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is -1.33%, 95% CI [-2.50%, -0.17%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=79.41%, FWS=91.18%, FollowSelectivity=-11.76%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction/destruction subsets each contain only 35 directional cases, so cross-seed replication remains necessary.
