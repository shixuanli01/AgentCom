# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 27.89% | 8.16% | 92.86% | 50.51% | 50.51% | 8.33% | 7.29% | 1.04% |
| Full Text | 30.27% | 80.61% | 34.69% | 57.65% | 57.65% | 80.21% | 66.67% | 13.54% |
| StateBridge | 29.93% | 69.39% | 44.90% | 57.14% | 57.14% | 68.75% | 54.17% | 14.58% |
| LatentMAS | 28.23% | 58.16% | 45.92% | 52.04% | 52.04% | 57.29% | 55.21% | 2.08% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 0.34% | [-1.48%, 2.23%] |
| cr | 11.22% | [1.19%, 21.70%] |
| pr | -10.20% | [-20.00%, 0.00%] |
| si | 0.51% | [-5.00%, 6.25%] |
| sra | 0.51% | [-5.00%, 6.25%] |
| fcs | 11.46% | [1.22%, 22.22%] |
| fws | 12.50% | [2.13%, 23.00%] |
| follow_selectivity | -1.04% | [-11.70%, 10.20%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 127/588 answers (21.60%, 95% CI [16.86%, 26.60%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 2.04%, 95% CI [-0.71%, 4.74%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=68.75%, FWS=54.17%, FollowSelectivity=14.58%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 98 directional cases and the destruction subset contains 98, so cross-seed replication remains necessary.


## Accuracy on the full benchmark

The Acc column above is conditional on the 98 mixed items phase 2 retained. Over all 1165 items (6990 directional records, of which 6402 fall on the 1067 items every agent already answered correctly):

| Channel | Receiver accuracy | 95% CI | Item accuracy (majority vote) | Measured / extrapolated |
|---|---:|---:|---:|---:|
| No message | 93.93% | [93.13%, 94.74%] | 94.16% | 796 / 6194 |
| Full Text | 94.13% | [93.33%, 94.94%] | 94.18% | 796 / 6194 |
| StateBridge | 94.11% | [93.30%, 94.91%] | 94.10% | 796 / 6194 |
| LatentMAS | 93.52% | [92.38%, 94.67%] | 94.03% | 796 / 6194 |

Pre-communication accuracy (phase 1, one independent belief per agent): 93.91%.

Receiver accuracy counts one revised answer per (item, sender to receiver) pair. The records on skipped items are extrapolated from a directly measured sample of those same items, never from the mixed ones: mixed items are harder by construction, and borrowing their rate put ARC-Challenge 5.7 points low and manufactured a 5.9-point channel gap that direct measurement puts at 0.6. The interval covers only that extrapolation, so it is common to all channels and the differences between them are tighter than the intervals suggest.
Item accuracy majority-votes the three agents' revised answers, averaged over the eight ways to give each receiver one sender.