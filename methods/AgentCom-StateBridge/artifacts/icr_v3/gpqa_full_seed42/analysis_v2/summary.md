# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 25.42% | 14.04% | 96.49% | 55.26% | 55.26% | 9.80% | 0.98% | 8.82% |
| Full Text | 27.26% | 74.56% | 47.37% | 60.96% | 60.96% | 71.57% | 56.86% | 14.71% |
| StateBridge | 26.84% | 64.04% | 51.75% | 57.89% | 57.89% | 60.78% | 51.96% | 8.82% |
| LatentMAS | 25.14% | 61.40% | 50.88% | 56.14% | 56.14% | 56.86% | 50.98% | 5.88% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | 0.42% | [-1.88%, 2.78%] |
| cr | 10.53% | [0.75%, 21.05%] |
| pr | -4.39% | [-14.81%, 5.66%] |
| si | 3.07% | [-4.30%, 10.27%] |
| sra | 3.07% | [-4.30%, 10.27%] |
| fcs | 10.78% | [0.00%, 22.54%] |
| fws | 4.90% | [-6.25%, 16.49%] |
| follow_selectivity | 5.88% | [-11.11%, 23.00%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 199/708 answers (28.11%, 95% CI [23.66%, 32.68%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 1.41%, 95% CI [-1.09%, 3.83%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=60.78%, FWS=51.96%, FollowSelectivity=8.82%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 114 directional cases and the destruction subset contains 114, so cross-seed replication remains necessary.


## Accuracy on the full benchmark

The Acc column above is conditional on the 118 mixed items phase 2 retained. Over all 198 items (1188 directional records, of which 480 fall on the 80 items every agent already answered correctly):

| Channel | Receiver accuracy | 95% CI | Item accuracy (majority vote) | Measured / extrapolated |
|---|---:|---:|---:|---:|
| No message | 54.58% | [53.67%, 55.48%] | 54.04% | 832 / 356 |
| Full Text | 56.32% | [55.68%, 56.97%] | 56.69% | 832 / 356 |
| StateBridge | 55.75% | [54.96%, 56.53%] | 55.93% | 832 / 356 |
| LatentMAS | 55.06% | [54.41%, 55.71%] | 55.43% | 831 / 357 |

Pre-communication accuracy (phase 1, one independent belief per agent): 54.04%.

Receiver accuracy counts one revised answer per (item, sender to receiver) pair. The records on skipped items are extrapolated from a directly measured sample of those same items, never from the mixed ones: mixed items are harder by construction, and borrowing their rate put ARC-Challenge 5.7 points low and manufactured a 5.9-point channel gap that direct measurement puts at 0.6. The interval covers only that extrapolation, so it is common to all channels and the differences between them are tighter than the intervals suggest.
Item accuracy majority-votes the three agents' revised answers, averaged over the eight ways to give each receiver one sender.