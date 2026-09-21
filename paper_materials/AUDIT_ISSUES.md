# AgentCom — audit issues

Read-only audit, 2026-09-21. Nothing here was repaired, and no experiment was
rerun. Each entry states the affected tables, whether it blocks writing, and what
evidence would close it.

---

## A. Affects main results

### A1 — Non-terminating generations are documented as excluded but are retained and scored wrong — **INCONSISTENT**

`excluded_item_ids` is `[]` for MedQA, GSM8K and HumanEval+, and contains only the
7 pre-registered ARC-Challenge structural exclusions. No exclusion of degenerate
repetition loops happens anywhere in the pipeline. The commit messages published
with these artifacts state "what still fails there is a degenerate repetition loop
and its item is excluded", which the artifacts contradict.

The effect is not neutral: a non-terminating generation emits no answer, is scored
wrong, and "wrong" is exactly the condition for entering the correction and
both-wrong subsets.

Contamination (`tables/truncation_contamination.csv`):

| Dataset | CR denominator touched | PR | SR |
|---|---:|---:|---:|
| MedQA | 0.00% | 0.00% | 0.92% |
| ARC-C | 2.04% | 2.04% | 1.22% |
| GSM8K | 4.35% | 4.35% | 0.00% |
| HumanEval+ | **32.14%** | **32.14%** | 21.74% |

- Tables: `main_results.csv`, `sr_scr.csv`, `paired_vs_no_message.csv`, `six_direction_detail.csv`
- **Blocking for HumanEval+ as a main-text result; not blocking for MedQA, ARC-C or GSM8K**, where the share is ≤4.35% and must be stated.
- To close: either state the retention rule in the paper and report the contamination share, or pre-register an exclusion rule and recompute. Do not change the commit-message wording retroactively without also stating which is true.

### A2 — GPQA-Diamond is incomplete — **PENDING**

79 of 198 items carry three beliefs at 01:17 UTC (agent C generating, 91/198
beliefs). `revisions/merged.jsonl` in that artifact holds 1,136 records from the
earlier **two-agent** run and is stale relative to the three-agent scope.

- Tables: all GPQA rows are absent or marked PENDING.
- **Blocking** for any GPQA claim.
- To close: let phase 1 finish, then phase 2 (six directions, `--all-correct-sample 1`), then merge with `--require-complete` and re-run analysis. Do not write a GPQA conclusion from the projected denominator or the 41.9% two-agent initial accuracy.

### A3 — No development/test split by item — **MISSING**

No config records an item-level dev/test assignment. Token budgets were raised
after observing truncation on the same items that are analysed, so there is no
held-out set.

- Tables: all.
- **Not blocking**, but must be stated as a limitation. The audit is of a single frozen run; it is not a tune-then-evaluate protocol.

### A4 — Single replication — scope limit

Everything rests on `replication_id = seed_pair_00`. The item-cluster bootstrap
quantifies item sampling variability under one seed; it says nothing about
cross-seed stability.

- Tables: `main_results.csv` (CI columns).
- **Not blocking**, but "stable across seeds" must not be written.

---

## B. Affects interpretation

### B1 — MedQA's 34 extra records per condition: provenance resolved — **VERIFIED, already handled**

The question was where MedQA's 754-record denominator and 86-record SCR
denominator came from when the retained scope implies 720 and 52.

Finding: the 34 extra records per condition (136 total) are **leftover phase-2
records from the earlier two-agent run**, generated under `keep_one_in = 10`
both-correct sampling. They land on items that became all-three-correct once agent
C was added, so the three-agent scope filter should have moved them out of the
analysis subset. MedQA's `merged.jsonl` was written at 17:29 UTC on 2026-09-20,
before the scope filter landed at 19:18, so they stayed in; ARC-C, GSM8K and
HumanEval+ were merged after the fix and never carried them.

They are **not** erroneous records. All 136 carry an accepted `config_fingerprint`
and a `receiver_prior_sha256` byte-identical to the current belief. They are the
only direct measurement of the skipped population on MedQA.

Effect: the previously published MedQA `Acc_ret` was inflated by roughly 3.3 points
uniformly across the four channels (29.05 → 25.69% for No Message). CR, PR, SI, SR,
FCS and FWS were unaffected, because all 136 are both-correct pairs.
SCR denominator 86 = 52 retained + 34 skipped-sample.

Resolution already applied (commit `4197f59`, before this audit): `icr.merge` now
splits such records into `revisions/all_correct_sample.jsonl`, MedQA's
`merged.jsonl` is 2,880 = 720 × 4, and the records feed Appendix E instead of being
discarded. They are **not** double-counted — verified in Appendix E check 4.

- Tables: `sample_statistics.csv`, `main_results.csv`, `full_set_accuracy_estimated.csv`
- **Not blocking.** Any MedQA number quoted from a pre-`4197f59` artifact must be re-read from the current one.

### B2 — Full-set accuracy is largely imputed — **PROVISIONAL**

Direct coverage of the full item × direction grid is 41.9% (MedQA), 11.4% (ARC-C),
10.1% (GSM8K), 19.5% (HumanEval+). The remainder is imputed from a sample of the
same skipped population.

- Tables: `full_set_accuracy_estimated.csv`, `accuracy_scopes.csv`
- **Not blocking** if labelled. Every row carries `ESTIMATED, NOT MEASURED FULL-SET ACCURACY`. A "measured full-set accuracy" column must not be published for these four datasets.
- To close: run the remaining skipped directions. MedQA needs ~180 more records per channel to reach ±0.5 pp; HumanEval+ ~380. ARC-C and GSM8K are already at ±0.8 and ±0.7 pp. **Not executed** — this audit starts no runs.

### B3 — Message-construction cost is not comparable across channels — **MISSING**

`communication_seconds` is recorded only for LatentMAS. StateBridge's alignment
cost lives on the phase-1 belief record (`alignment_seconds`), not per message, and
Full Text has no construction step. There is no matched-compute control run.

- Tables: `cost.csv` (`message_construction_seconds_mean = MISSING` for three of four channels)
- **Not blocking** for §5.2; **blocking** for any cost claim in §5.3.
- To close: instrument construction per message, or state the asymmetry explicitly and report only receiver-side tokens and latency. Latent payload bytes must not be converted into an equivalent token count.

### B4 — Wall-clock totals are not a throughput comparison

Worker count per GPU varied across runs (2 per GPU for phase 1, 1 per GPU for
phase 2 after OOM, and `fill` passes at different concurrency). `generation_seconds`
is measured in-process and is valid per record, but totals reflect contention.

- Tables: `cost.csv` (`wall_clock_note`)
- **Not blocking**, provided no throughput claim is made.

### B5 — StateBridge does not have the highest PR everywhere

On ARC-Challenge LatentMAS has the higher preservation rate (45.92% vs 44.90%).
Finding F3 is worded to say StateBridge preserves more than **Full Text**, which
holds on all three verified datasets.

- Tables: `main_results.csv`
- **Not blocking**; it is a wording constraint.

---

## C. Supplementary material missing

### C1 — E0–E4 development diagnostics and the 75-item MedQA pilot — **MISSING**

Nothing in this repository matches. A search for `E0`–`E4`, `75 items`, `n=75`,
`pilot` across `*.py`, `*.md` and `*.sh` returns no hit.

Consequently unavailable: the item list and stratification, overlap with the ABC
main experiment, per-variant metrics with integer counts, message lengths and
construction failures, the E1−E0 / E3−E2 / E4−E3 paired comparisons, the E2/E3
equal-length check, the E3/E4 identical-segment-and-order check, the offline
random-mixture reference curve (probability p of Full Text vs No Message), and the
E0 PR failure-case audit.

- Tables: §5.4 A has no table.
- **Blocking** for §5.4 A. To close: import the pilot artifacts from wherever they live. They must not be pooled with the ABC results.

### C2 — Final EGR method and held-out evaluation — **PENDING**

`egr/` implements V1 Contrast and Permutation, frozen 2026-09-18. Its own freeze
document states it is a receiver-policy comparison and not a fair transport
comparison. Per-record EGR artifacts are **MISSING** from this repository — only
aggregate numbers in `experiments/EGR_*_REPORT_ZH.md` survive — and those numbers
were produced under the earlier two-agent protocol with different prompts.

- Tables: §5.3 has no result table.
- **Blocking** for §5.3 and §5.4 B. To close: freeze the final method, run it under ICR-V3 on the same records as the four channels, and retain per-record artifacts.

### C3 — Model revision and attention implementation are unpinned — **MISSING**

`config.json` records `Qwen/Qwen3-4B` with no revision or commit hash, and
`attn_implementation` is never set.

- Tables: Appendix A.
- **Not blocking**, but exact reproduction cannot be guaranteed. State the transformers version from `manifest.json` and note the unpinned revision.

### C4 — HumanEval+ literal-newline evidence — **MISSING**

No recorded instance was located in this audit. The raw prompts are retained per
record and the check is runnable against them; no evidence is presented.

- Tables: Appendix D item 5.
- **Not blocking** if the claim is dropped. Do not assert the problem without an example.
