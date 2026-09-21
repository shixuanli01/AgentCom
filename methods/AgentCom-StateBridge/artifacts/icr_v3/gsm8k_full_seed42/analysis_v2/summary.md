# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 10,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 26.24% | 13.04% | 92.39% | 52.72% | 52.72% | 13.04% | 5.43% | 7.61% |
| Full Text | 25.35% | 51.09% | 52.17% | 51.63% | 51.63% | 51.09% | 46.74% | 4.35% |
| StateBridge | 26.95% | 39.13% | 73.91% | 56.52% | 56.52% | 39.13% | 23.91% | 15.22% |
| LatentMAS | 25.53% | 59.78% | 46.74% | 53.26% | 53.26% | 59.78% | 52.17% | 7.61% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | -1.60% | [-4.28%, 1.11%] |
| cr | 11.96% | [2.94%, 21.67%] |
| pr | -21.74% | [-32.69%, -10.52%] |
| si | -4.89% | [-13.04%, 3.66%] |
| sra | -4.89% | [-13.04%, 3.66%] |
| fcs | 11.96% | [2.94%, 21.67%] |
| fws | 22.83% | [11.70%, 34.04%] |
| follow_selectivity | -10.87% | [-27.14%, 6.13%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 73/564 answers (12.94%, 95% CI [9.74%, 16.49%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 0.71%, 95% CI [-2.00%, 3.54%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=39.13%, FWS=23.91%, FollowSelectivity=15.22%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 92 directional cases and the destruction subset contains 92, so cross-seed replication remains necessary.


## Accuracy on the full benchmark

The Acc column above is conditional on the 94 mixed items phase 2 retained. Over all 1319 items (7914 directional records, of which 7350 fall on the 1225 items every agent already answered correctly):

| Channel | Receiver accuracy | 95% CI | Item accuracy (majority vote) | Measured / extrapolated |
|---|---:|---:|---:|---:|
| No message | 94.74% | [94.02%, 95.47%] | 94.71% | 799 / 7115 |
| Full Text | 94.68% | [93.96%, 95.40%] | 94.67% | 799 / 7115 |
| StateBridge | 94.79% | [94.07%, 95.52%] | 94.75% | 799 / 7115 |
| LatentMAS | 94.69% | [93.96%, 95.42%] | 94.60% | 797 / 7117 |

Pre-communication accuracy (phase 1, one independent belief per agent): 94.62%.

Receiver accuracy counts one revised answer per (item, sender to receiver) pair. The records on skipped items are extrapolated from a directly measured sample of those same items, never from the mixed ones: mixed items are harder by construction, and borrowing their rate put ARC-Challenge 5.7 points low and manufactured a 5.9-point channel gap that direct measurement puts at 0.6. The interval covers only that extrapolation, so it is common to all channels and the differences between them are tighter than the intervals suggest.
Item accuracy majority-votes the three agents' revised answers, averaged over the eight ways to give each receiver one sender.