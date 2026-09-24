# AgentCom — audit issues

Read-only audit, 2026-09-21. Nothing here was repaired, and no experiment was
rerun. Each entry states the affected tables, whether it blocks writing, and what
evidence would close it.

---

## A. Affects main results

### A1 — Non-terminating generations: the documentation says excluded, the artifacts retain them. **The artifacts are authoritative.**

A phase-1 generation that consumes all of `max_new_tokens` without emitting EOS
(`hit_eos = false`) never states an answer, so `parsed_answer` is `None` and the
belief is scored wrong. Nothing is deleted: the item stays in
`selected_item_ids`, stays in `revisions/merged.jsonl`, and every revision
record that rests on it is retained and scored. `excluded_item_ids` is `[]` for
MedQA, GSM8K and HumanEval+ and holds only the 7 pre-registered ARC-Challenge
structural exclusions.

Several commit messages published with these artifacts state "what still fails
there is a degenerate repetition loop and its item is excluded". **That sentence
is wrong.** No such exclusion exists anywhere in the pipeline. The papers, the
protocol documents and any future commit message must describe the retention
rule instead. The published commit messages cannot be edited; this entry is the
correction of record.

Why it matters, and why the phase-1 rate understates it: a truncated belief is
scored wrong, and "wrong" is exactly the condition for entering the correction,
destruction and both-wrong subsets. Correct beliefs, by contrast, mostly land on
all-correct items that phase 2 skips. So a small phase-1 failure rate is
amplified in the analysed denominators.

| Dataset | Phase-1 non-termination | CR denominator touched | PR | SR |
|---|---:|---:|---:|---:|
| MedQA | 1/900 = 0.11% | 0.00% | 0.00% | 0.92% |
| ARC-C | 2/3,495 = 0.06% | 2.04% | 2.04% | 1.22% |
| GSM8K | 2/3,957 = 0.05% | 4.35% | 4.35% | 0.00% |
| GPQA-D | 13/594 = 2.19% | 10.53% | 10.53% | 4.17% |
| HumanEval+ | 12/492 = 2.44% | **32.14%** | **32.14%** | 21.74% |

**Decision: retain as the primary analysis; report exclusion as a sensitivity
analysis.** `tables/truncation_sensitivity.csv` recomputes CR, PR, SI and Acc_ret
with every item carrying a non-terminating phase-1 generation dropped whole (all
six directions), under a label-free rule that depends only on `hit_eos`.

| Dataset | SI shift on exclusion | Channel ordering |
|---|---|---|
| MedQA | 0.00 pp (all channels) | unchanged |
| ARC-C | −1.00 to +0.01 pp | unchanged |
| GSM8K | −2.20 to −1.98 pp | unchanged |
| GPQA-D | −4.46 to −2.76 pp | unchanged |
| HumanEval+ | −9.32 to +2.18 pp, denominators fall to 18 | unchanged but uninformative |

Two reasons not to promote exclusion to primary. First, the rule is label-free
but **not difficulty-neutral**: the generations that loop are the hard items, so
excluding them systematically thins the analysed subsets of their hardest cases.
Second, the exclusion would be adopted after seeing the results — the
contamination share is only knowable once computed — which is not a
pre-registration. Retaining costs nothing in conclusions: on MedQA, ARC-C, GSM8K
and GPQA-D the shift is at most 4.5 pp and moves all four channels in the same
direction.

- Tables: `main_results.csv`, `sr_scr.csv`, `paired_vs_no_message.csv`, `six_direction_detail.csv`, `truncation_sensitivity.csv`, `truncation_contamination.csv`
- **Not blocking for MedQA, ARC-C or GSM8K**, provided the retention rule and the contamination share are stated. **Blocking for HumanEval+ as a main-text result** — 32% of its CR and PR denominators, and only 18 pairs left if excluded.
- GPQA-Diamond sits between the two at 10.53%; its CR and PR must carry that figure.

### A2 — GPQA-Diamond — **RESOLVED**

Phase 1 finished with 594 of 594 beliefs and phase 2 completed at 10:47 UTC with
0 worker failures: 2,832 merged records (708 directed pairs x 4 conditions) plus
495 records on all-correct items split into `all_correct_sample.jsonl`. Every
integrity check passes — 0 duplicate keys, 0 stale receiver priors, 0 pairs whose
prior diverges across conditions, 0 records with `status != complete`, and the
closed-form counting identity holds in all four cells.

Two facts that must travel with the GPQA numbers. Its initial accuracy is
**54.04%**, not the 41.9% recorded in earlier notes and several commit messages;
that figure came from an 8,192-token run in which 28% of beliefs never terminated
and were scored wrong. And 10.53% of its CR and PR denominators touch a
non-terminating generation, the second-highest share in the study (see A1).

- Tables: all GPQA rows are populated and marked VERIFIED.
- **No longer blocking.**

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
