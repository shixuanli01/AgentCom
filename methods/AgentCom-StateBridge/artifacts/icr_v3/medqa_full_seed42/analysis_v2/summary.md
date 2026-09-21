# ICR Communication-First Step 1

`seed_pair_00` was analyzed without additional model inference. Confidence intervals use 2,000 item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.

| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No message | 25.69% | 6.90% | 98.28% | 52.59% | 52.59% | 6.90% | 0.86% | 6.03% |
| Full Text | 27.22% | 80.17% | 42.24% | 61.21% | 61.21% | 80.17% | 57.76% | 22.41% |
| StateBridge | 28.89% | 57.76% | 75.00% | 66.38% | 66.38% | 57.76% | 25.00% | 32.76% |
| LatentMAS | 24.72% | 69.83% | 32.76% | 51.29% | 51.29% | 69.83% | 67.24% | 2.59% |

## Text − StateBridge

| Metric | Difference | 95% item-cluster bootstrap CI |
|---|---:|---:|
| accuracy | -1.67% | [-3.70%, 0.25%] |
| cr | 22.41% | [12.24%, 32.26%] |
| pr | -32.76% | [-42.54%, -23.19%] |
| si | -5.17% | [-11.11%, 0.46%] |
| sra | -5.17% | [-11.11%, 0.46%] |
| fcs | 22.41% | [12.24%, 32.26%] |
| fws | 32.76% | [23.19%, 42.54%] |
| follow_selectivity | -10.34% | [-22.22%, 0.93%] |

## Evidence-bounded answers

- **High sender influence?** Yes in this artifact: true StateBridge differs from none on 136/720 answers (18.89%, 95% CI [15.30%, 22.46%]).
- **Positive communication utility?** No on overall accuracy: StateBridge CE is 3.19%, 95% CI [0.41%, 5.99%].
- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS=57.76%, FWS=25.00%, FollowSelectivity=32.76%. This is a behavioral description of this run, not yet a cross-seed causal generalization.

The correction subset contains 116 directional cases and the destruction subset contains 116, so cross-seed replication remains necessary.


## Accuracy on the full benchmark

The Acc column above is conditional on the 120 mixed items phase 2 retained. Over all 300 items (1800 directional records, of which 1080 fall on the 180 items every agent already answered correctly):

| Channel | Receiver accuracy | 95% CI | Item accuracy (majority vote) | Measured / extrapolated |
|---|---:|---:|---:|---:|
| No message | 70.28% | [67.33%, 73.23%] | 70.25% | 754 / 1046 |
| Full Text | 70.89% | [67.94%, 73.84%] | 71.08% | 754 / 1046 |
| StateBridge | 71.56% | [68.61%, 74.51%] | 72.00% | 754 / 1046 |
| LatentMAS | 69.89% | [66.94%, 72.84%] | 69.92% | 754 / 1046 |

Pre-communication accuracy (phase 1, one independent belief per agent): 69.33%.

Receiver accuracy counts one revised answer per (item, sender to receiver) pair. The records on skipped items are extrapolated from a directly measured sample of those same items, never from the mixed ones: mixed items are harder by construction, and borrowing their rate put ARC-Challenge 5.7 points low and manufactured a 5.9-point channel gap that direct measurement puts at 0.6. The interval covers only that extrapolation, so it is common to all channels and the differences between them are tighter than the intervals suggest.
Item accuracy majority-votes the three agents' revised answers, averaged over the eight ways to give each receiver one sender.