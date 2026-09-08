# MP-StateBridge V1 Concurrency Benchmark

## Status

- Date: 2026-09-05
- GPU: NVIDIA RTX 5090 32 GB
- Model: Qwen3-4B
- Dataset: MedQA items 10 and 11
- Configuration: one complete StateBridge branch per item
- Outcome: two independent model processes on one GPU do not improve throughput

## Controlled Comparison

Both conditions used the same item IDs, prompts, stage-level seeds, generated
token counts, predictions, and correctness outcomes.

| Metric | One process, sequential | Two processes, concurrent |
|---|---:|---:|
| Total wall time | 112.60 s | 116.07 s |
| Throughput speedup | 1.000x | 0.970x |
| Item 10 branch time | 36.91 s | 75.54 s |
| Item 11 branch time | 66.89 s | 106.58 s |
| Peak GPU memory | about 10.6 GB | 20.9 GB |
| Mean sampled GPU utilization | about 85% | 81.5% |
| Peak sampled GPU utilization | about 94% | 98.0% |

The concurrent condition was approximately 3% slower in wall-clock time while
using about twice the GPU memory. Per-request latency increased by 59% to 105%.

## Bottleneck Profile

Across the first eight authoritative sequential branches before this benchmark:

| Component | Mean per branch | Share of branch wall time |
|---|---:|---:|
| Autoregressive model generation | 34.54 s | 92.1% |
| Three StateBridge alignments | 2.91 s | 7.8% |
| Total branch wall time | 37.50 s | 100% |

The branch generated 1,357 tokens on average across Planner, Critic, Refiner,
and Judger. The dominant cost is therefore four serial, batch-size-one decoding
passes. Procrustes alignment is not the primary bottleneck.

## Interpretation

Two processes duplicate the model and KV-cache memory but contend for the same
GPU execution and memory-bandwidth resources. They raise instantaneous peak
utilization without increasing completed-token throughput. This is process
concurrency, not efficient batched decoding.

The next credible optimization is one-model branch batching: run the five
Planners as a batch, align each path independently, then batch the five Critics,
Refiners, and Judgers. That requires a dedicated implementation because paths
have variable generation lengths, independent random streams, and separate
Procrustes fits. It must be validated for exact branch isolation before it can
replace the sequential authoritative runner.

## Decision

Keep the MP-StateBridge V1 authoritative MedQA run single-process and
sequential. Do not use two model replicas on the RTX 5090. Treat one-model
batched branching as a later engineering optimization after the V1 metric
question is answered.
