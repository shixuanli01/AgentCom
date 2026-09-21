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


## Accuracy on the full benchmark

The Acc column above is conditional on the 28 mixed items phase 2 retained. Over all 164 items (984 directional records, of which 816 fall on the 136 items every agent already answered correctly):

| Channel | Receiver accuracy | 95% CI | Measured / extrapolated |
|---|---:|---:|---:|
| No message | 86.69% | [73.75%, 99.63%] | 192 / 792 |
| Full Text | 90.65% | [79.54%, 100.00%] | 192 / 792 |
| StateBridge | 90.35% | [79.24%, 100.00%] | 192 / 792 |
| LatentMAS | 89.74% | [78.63%, 100.00%] | 192 / 792 |

Pre-communication accuracy (phase 1, one independent belief per agent): 87.80%.

Receiver accuracy counts one revised answer per (item, sender to receiver) pair. The records on skipped items are extrapolated from a directly measured sample of those same items, never from the mixed ones: mixed items are harder by construction, and borrowing their rate put ARC-Challenge 5.7 points low and manufactured a 5.9-point channel gap that direct measurement puts at 0.6. The interval covers only that extrapolation, so it is common to all channels and the differences between them are tighter than the intervals suggest.
Item accuracy is undefined here: answers are programs, so three distinct strings never form a majority.