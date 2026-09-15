# ICR MedQA300 experiment ledger

This file preserves the decisions that would otherwise exist only in interactive work history. It distinguishes formal results from smoke tests and abandoned partial runs.

## Recovery anchors

- Repository: `shixuanli01/AgentCom`
- Formal integration branch: `icr-framework-20260915`
- Main analysis-results commit: `f53f71b30c64b947e22aca6da49fa23151d065e8`
- Chinese report commit: `88a58d4da87d22516bcac01ff1e1e888feb255df`
- Frozen artifact release/tag: `icr-medqa300-20260915`
- Release URL: <https://github.com/shixuanli01/AgentCom/releases/tag/icr-medqa300-20260915>
- Raw archive: `icr-artifacts-20260915.tar.zst`
- Raw archive SHA256: `504ba13a3038c64e0560c9e2840e826a8cb1d5bc462c9f0c874a376b87d5eb5d`

## Decision timeline

### Preliminary method work

- The repository was reduced to a focused AgentCom/StateBridge experimental workspace.
- A separate Trajectory Memory Relay (TMR) line was tested, including a `last64` run. The owner explicitly stopped the active implementation and moved to a new audit framework. TMR is historical context, not part of the formal ICR result reported here.

### ICR framework freeze

- The new evaluation objective was communication-first auditing rather than development of another communication method.
- The protocol was frozen as Independent Belief → Communication → one Revision.
- Agent A and Agent B had to solve independently before communication.
- Both A→B and B→A directions were retained.
- The receiver kept its own first-pass reasoning and answer; it was instructed to revise rather than solve from scratch.
- Prompts and generation settings were frozen before the formal replication runs. No result-conditioned prompt tuning was permitted.
- For the same replication, item, and direction, every communication condition used the same revision seed.

### Frozen seed-pair 00

- The completed artifact `icr_medqa300_v2_sourcefront_seed42` was preserved without regeneration and treated as `seed_pair_00`.
- Its StateBridge prefix is injected at the front of the receiver user turn, before the question, matching the official receiver topology used by the repository implementation.
- Step 1 performed analysis only; it did not make new model calls.
- The original approximate regression signature was reproduced: Text retained substantially more correct receiver beliefs than StateBridge, while both channels had high correction rates.

### Text/StateBridge replication

- The original plan requested four new pairs (five total), but the experiment owner later reduced the formal scope to three total seed pairs.
- `seed_pair_01` and `seed_pair_02` were completed in addition to frozen `seed_pair_00`.
- Work already produced for `seed_pair_03` and `seed_pair_04` was stopped and preserved as historical partial artifacts. It is excluded from formal pooled estimates.
- Near the end of the run, incomplete shards for seed01/02 were resumed with `--global-resume` and rebalanced across four GPUs. Completed keys were not regenerated.
- Formal Text/StateBridge scope: three seed pairs × 300 items × two directions × seven conditions = 12,600 directional-condition records.

### LatentMAS integration

- LatentMAS was evaluated only as a communication channel inside the frozen ICR protocol, not as its original four-agent application pipeline.
- The same cached first-pass sender trajectory used by Text and StateBridge was reused. Sender states were reconstructed by teacher-forcing the exact cached prompt plus generated token IDs; the sender was not resampled.
- A smoke test was required before formal execution.
- The experiment owner requested one additional LatentMAS seed after the first run. Formal LatentMAS scope therefore became `seed_pair_00` and `seed_pair_01`.
- `seed_pair_02` has complete Text/StateBridge results but no formal LatentMAS result.
- Formal four-channel scope: two seed pairs × 300 items × two directions × ten conditions = 12,000 directional-condition records.

### Completion and publication

- Formal inference finished at approximately 2026-09-15 07:51 UTC.
- Every included condition contains exactly 600 records per included seed.
- All record JSON files parsed; `(item_id, direction, condition)` keys were unique.
- Frozen seed00 and seed01/02 cache audits passed.
- The test suite passed 50/50 tests.
- Per-seed, pooled, causal-control, compute-cost, and 10,000-sample item-cluster bootstrap outputs were committed.
- The complete local artifact tree, including formal runs, logs, smoke artifacts, and stopped partial runs, was packaged in a GitHub Release with per-file and archive SHA256 checksums.
- A Chinese audit report was added to `reports/` and the complete feature history was fast-forwarded into `main`.

## Formal conclusions at freeze time

- Three-seed Text accuracy: 72.72%; StateBridge accuracy: 71.06%.
- Text − StateBridge accuracy: +1.67 percentage points, 95% item-cluster bootstrap CI `[+0.83, +2.50]`.
- Text and StateBridge have similar correction rates, but Text preservation rate exceeds StateBridge by 25.74 points, CI `[16.04, 35.63]`.
- StateBridge follows wrong senders very frequently: pooled FWS 89.00%, compared with FCS 85.00%.
- LatentMAS does not reproduce the StateBridge low-preservation signature. The predeclared directional classification is `B_directional_signature`.
- These conclusions are limited to MedQA300, three Text/StateBridge seed pairs, and two LatentMAS seed pairs. Cross-benchmark Step 4 remains pending.

## Formal and non-formal artifact boundaries

Formal outputs:

- `artifacts/icr_medqa300/seed_pair_00/`
- `artifacts/icr_medqa300/seed_pair_01/`
- `artifacts/icr_medqa300/seed_pair_02/` for Text/StateBridge only
- `artifacts/icr_medqa300/pooled_3seed/`
- `artifacts/icr_medqa300/pooled_2seed_with_latentmas/`

Historical but non-formal outputs:

- Directories with `smoke` in their names
- `pooled_1seed` and `pooled_1seed_with_latentmas`, which were diagnostic analysis checks
- Incomplete `seed_pair_03` and `seed_pair_04`
- Earlier TMR evaluations

Historical outputs are retained for provenance but must not be mixed into the published pooled estimates.

## What was deliberately not done

- No new communication method was introduced during the ICR audit.
- No prompt was tuned after observing formal outcomes.
- No gold label was supplied to generation prompts.
- No new sender trajectory was sampled specifically for LatentMAS.
- Cross-benchmark Step 4A/4B was deferred for later review.
