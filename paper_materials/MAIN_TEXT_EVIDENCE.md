# AgentCom — Main-text experimental evidence

Generated 2026-09-21 from the ICR-V3 artifacts. Read-only audit: no GPU run was
started, no configuration or historical result was modified, nothing was rerun.

Status vocabulary: **VERIFIED** (checked against per-record artifacts),
**PROVISIONAL** (computed but resting on an assumption stated inline),
**INCONSISTENT** (two sources disagree), **MISSING** (not in this repository),
**PENDING** (run not finished).

| Dataset | Status | Note |
|---|---|---|
| MedQA | VERIFIED | 300 items, three beliefs each |
| ARC-Challenge | VERIFIED | 1,165 items after 7 pre-registered structural exclusions |
| GSM8K | VERIFIED | 1,319 items |
| GPQA-Diamond | **PENDING** | agent C beliefs still generating (91/198 at 01:17 UTC); phase 2 not started |
| HumanEval+ | VERIFIED but appendix-only | 32% of its CR/PR denominators touch a non-terminating generation |

---

## 5.1 Experimental Setup

### A. Scope (2–3 sentences for the main text)

> We audit four communication channels under a single Independent–Communicate–Revise
> protocol on Qwen3-4B. For every item, three agents (A, B, C) first produce
> independent beliefs from identical prompts under distinct deterministic seeds;
> each ordered pair of agents then performs exactly one revision in which the
> receiver keeps its own prior reasoning and receives a message from the sender.
> Metrics are computed over all six ordered pairs per item, and the receiver's
> post-revision answer is the unit of evaluation.

### B. Baselines (1–2 sentences each) — VERIFIED against `icr/prompts_v3.py` and `icr/runtime.py`

- **No Message.** The receiver re-enters the revision prompt with its own question,
  its own prior reasoning and its own prior answer, and the message slot reads
  `No external message is available.` It isolates the effect of a second inference
  pass from the effect of communication.
- **Full Text.** The sender's complete phase-1 reasoning text is placed verbatim in
  the message slot. No summarisation, no extra generation call.
- **StateBridge.** The last 64 hidden states of the sender's phase-1 trajectory,
  Procrustes-aligned into the receiver's space, are spliced into the receiver's
  input embeddings at exactly the position the text message would have occupied.
- **LatentMAS.** The sender's full all-layer KV cache is prepended as a causal
  prefix, followed by 10 generated latent steps. Because RoPE is baked into the
  cache, its position is not a free parameter, so this condition cannot occupy the
  same mid-prompt slot as the other three.
- **Evidence / EGR channels.** MISSING as a frozen final method — see §5.3.

### C. Sample statistics — `tables/sample_statistics.csv` (VERIFIED)

| Dataset | Source items | Evaluated | Retained items | CR den | PR den | SR den | SCR den (retained) | Records / condition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedQA | 300 | 300 | 120 | 116 | 116 | 436 | 52 | 720 |
| ARC-C | 1,165 | 1,165 | 98 | 98 | 98 | 328 | 64 | 588 |
| GSM8K | 1,319 | 1,319 | 94 | 92 | 92 | 334 | 46 | 564 |
| GPQA-D | 198 | PENDING | — | — | — | — | — | — |
| HumanEval+ | 164 | 164 | 28 | 28 | 28 | 92 | 20 | 168 |

"Retained items" are those where the three agents did not all answer correctly;
items where all three were correct carry no correction and no destruction pair and
are skipped in phase 2 (`--skip-all-correct-items`). Skipped item counts:
MedQA 180, ARC-C 1,067, GSM8K 1,225, HumanEval+ 136.

**Counting identity — VERIFIED for all five datasets** (`tables/count_identity_check.csv`).
For an item with k of 3 correct beliefs the six ordered pairs contain k(3−k)
correction pairs, k(3−k) destruction pairs, (3−k)(2−k) both-wrong pairs and
k(k−1) both-correct pairs. Closed form equals the observed count in every cell,
over all items and over the retained scope separately.

### D. Metrics and accuracy scopes — `tables/accuracy_scopes.csv`

- **CR** — a receiver that was wrong, facing a sender that was right, ends right.
- **PR** — a receiver that was right, facing a sender that was wrong, stays right.
- **SI** — the equal-weight mean of CR and PR.
- **SR** — both were wrong; the receiver ends right.
- **SCR** — both were right; the receiver stays right.

Five accuracy scopes, which must never share a column:

| Scope | Denominator | Availability |
|---|---|---|
| Initial-answer accuracy | items × 3 beliefs | VERIFIED |
| Mixed-correctness directional accuracy | CR ∪ PR pairs | VERIFIED |
| Retained-subset accuracy `Acc_ret` | retained items × 6 | VERIFIED |
| Measured full-set accuracy | items × 6, all measured | **NOT AVAILABLE** — coverage is 42% (MedQA) to 11% (GSM8K) |
| Extrapolated full-set accuracy | items × 6, partly imputed | PROVISIONAL, Appendix E |

Initial-answer accuracy: MedQA 69.33%, ARC-C 93.91%, GSM8K 94.62%, HumanEval+ 87.80%.

**Scoring of non-terminating generations — retention, not exclusion.** A phase-1
generation that does not terminate within `max_new_tokens` states no answer and is
scored wrong. Such items are **retained**, not excluded; the only pre-registered
exclusion in the whole study is ARC-Challenge's 7 items whose option count is not
four. The phase-1 non-termination rate is 0.11% (MedQA), 0.06% (ARC-C), 0.05%
(GSM8K), 2.19% (GPQA-D) and 2.44% (HumanEval+), but because a wrong belief is what
puts a pair into the analysed subsets, the share of the CR and PR denominators
touching such a generation is larger: 0.00%, 2.04%, 4.35%, 10.53% and 32.14%
respectively. `tables/truncation_sensitivity.csv` recomputes every metric with
those items dropped; on MedQA, ARC-C and GSM8K the SI shift is at most 2.2 points
and moves all four channels together, leaving the ordering unchanged. State the
rule and the share; do not switch to exclusion post hoc.

**`Acc_mixed = SI` — VERIFIED, all 16 dataset×condition cells.** When both directions
of a mixed-correctness pair are present, the mixed-direction accuracy is
(CR·N + PR·N)/(2N) = (CR+PR)/2 = SI *identically*. They are one result, not two,
and must not be reported as independent evidence.

---

## 5.2 Auditing Communication Behavior

### A. Main results — `tables/main_results.csv` (VERIFIED)

All percentages are computed from integer counts. CIs are 2,000-resample
item-cluster paired bootstrap, analysis seed 20260921; a resample whose
denominator is zero is dropped and counted (`bootstrap_undefined_*` columns; all
zero for these datasets).

| Dataset | Condition | Acc_ret | CR | PR | SI |
|---|---|---|---|---|---|
| MedQA | No Message | 185/720 = 25.69% | 8/116 = 6.90% | 114/116 = 98.28% | 52.59% |
| MedQA | Full Text | 196/720 = 27.22% | 93/116 = 80.17% | 49/116 = 42.24% | 61.21% |
| MedQA | StateBridge | 208/720 = 28.89% | 67/116 = 57.76% | 87/116 = 75.00% | 66.38% |
| MedQA | LatentMAS | 178/720 = 24.72% | 81/116 = 69.83% | 38/116 = 32.76% | 51.29% |
| ARC-C | No Message | 164/588 = 27.89% | 8/98 = 8.16% | 91/98 = 92.86% | 50.51% |
| ARC-C | Full Text | 178/588 = 30.27% | 79/98 = 80.61% | 34/98 = 34.69% | 57.65% |
| ARC-C | StateBridge | 176/588 = 29.93% | 68/98 = 69.39% | 44/98 = 44.90% | 57.14% |
| ARC-C | LatentMAS | 166/588 = 28.23% | 57/98 = 58.16% | 45/98 = 45.92% | 52.04% |
| GSM8K | No Message | 148/564 = 26.24% | 12/92 = 13.04% | 85/92 = 92.39% | 52.72% |
| GSM8K | Full Text | 143/564 = 25.35% | 47/92 = 51.09% | 48/92 = 52.17% | 51.63% |
| GSM8K | StateBridge | 152/564 = 26.95% | 36/92 = 39.13% | 68/92 = 73.91% | 56.52% |
| GSM8K | LatentMAS | 144/564 = 25.53% | 55/92 = 59.78% | 43/92 = 46.74% | 53.26% |

GPQA-Diamond: **PENDING**. HumanEval+: see Appendix D.

CR–PR plotting data with denominators: `tables/cr_pr_plot_data.csv`.

### B. Paired comparison against No Message — `tables/paired_vs_no_message.csv` (VERIFIED)

Every channel shares the same (item, direction) records and the same revision seed
as No Message, so this is a matched comparison, not two independent rates.

Retained scope, discordant cells:

| Dataset | Channel | none✗→ch✓ | none✓→ch✗ | Net | Discordant |
|---|---|---:|---:|---:|---:|
| MedQA | Full Text | 86 | 75 | +11 | 161 |
| MedQA | StateBridge | 61 | 38 | +23 | 99 |
| MedQA | LatentMAS | 80 | 87 | −7 | 167 |
| ARC-C | Full Text | 76 | 62 | +14 | 138 |
| ARC-C | StateBridge | 64 | 52 | +12 | 116 |
| ARC-C | LatentMAS | 59 | 57 | +2 | 116 |
| GSM8K | Full Text | 40 | 45 | −5 | 85 |
| GSM8K | StateBridge | 30 | 26 | +4 | 56 |
| GSM8K | LatentMAS | 50 | 54 | −4 | 104 |

The same table stratified by CR / PR / SR / SCR subset shows where the movement
comes from. In the CR stratum the movement is almost purely gain (Full Text and
StateBridge lose 0 records on MedQA, ARC-C and GSM8K); in the PR stratum it is
almost purely loss (MedQA: Full Text −65, StateBridge −27, LatentMAS −76 of 116).

SI difference vs No Message (percentage points): MedQA Text +8.62, SB +13.79,
LatentMAS −1.29; ARC-C +7.14 / +6.63 / +1.53; GSM8K −1.09 / +3.80 / +0.54.

### C. SR and SCR — `tables/sr_scr.csv` (VERIFIED)

SR is near zero everywhere: the best cell is MedQA No Message 11/436 = 2.52%.
On the retained scope SCR is 52/52, 64/64 and 46/46 for MedQA, ARC-C and GSM8K in
most conditions (exact counts in the table). SCR measured on the *skipped*
population is reported separately in Appendix E and is the basis of the
extrapolation — it is never borrowed from the retained both-correct pairs.

### D. Answer transitions — `tables/answer_transitions.csv` (VERIFIED)

Categories: kept the receiver's prior answer / adopted the sender's initial answer /
moved to a third answer / invalid or unparsed. Normalisation for choice and numeric
tasks is the parsed answer, stripped and lowercased. For HumanEval+ the comparison
is whitespace-normalised program text and measures **string identity only**;
adoption is never inferred from pass/fail.

### E. Findings that can be written into the main text

**F1 — Every channel that corrects more than No Message also preserves less.**
> Relative to a no-message control, each of the three channels raises the
> correction rate and lowers the preservation rate; none raises both.

Support: `tables/main_results.csv`, all of MedQA, ARC-C, GSM8K (and HumanEval+ in
the appendix). CR rises from 6.90/8.16/13.04% (No Message) to 51–81% and PR falls
from 98.28/92.86/92.39% to 32–74%. The paired strata in
`tables/paired_vs_no_message.csv` show the mechanism: in the CR stratum Full Text
and StateBridge lose 0 records; in the PR stratum they gain at most 3.
Scope: one model, one replication (`seed_pair_00`), one revision per pair.

**F2 — The channels separate far more on correction/preservation than on accuracy.**
> On the retained subset the four conditions span 4.2 points of accuracy on MedQA
> while spanning 73 points of CR and 66 points of PR.

Support: `tables/main_results.csv`. MedQA Acc_ret 24.72–28.89% against CR
6.90–80.17% and PR 32.76–98.28%; ARC-C 27.89–30.27% against CR 8.16–80.61%;
GSM8K 25.35–26.95% against CR 13.04–59.78%.
Limit: this is a statement about this protocol's retained subset, not a claim that
accuracy is uninformative in general.

**F3 — StateBridge preserves more than Full Text on every verified dataset, and corrects less.**
> Across MedQA, ARC-Challenge and GSM8K, the hidden-state channel has a higher
> preservation rate and a lower correction rate than the full-text channel.

Support: PR 75.00 vs 42.24 (MedQA), 44.90 vs 34.69 (ARC-C), 73.91 vs 52.17 (GSM8K);
CR 57.76 vs 80.17, 69.39 vs 80.61, 39.13 vs 51.09.
Limit: StateBridge does **not** have the highest PR on ARC-Challenge — LatentMAS
does (45.92 vs 44.90). Do not write "highest on all datasets".

**F4 — Mixed-correctness directional accuracy is SI, not a second result.**
> When both directions of a mixed pair are evaluated, accuracy restricted to those
> pairs equals the equal-weight mean of CR and PR by construction.

Support: `tables/accuracy_scopes.csv`, `acc_mixed_equals_si = YES` in all 16 cells.
Use: prevents double-reporting; state the identity rather than two numbers.

**Not supported by current evidence — do not write:** "best on all datasets";
"significantly better than"; "latent channels cannot produce new answers"
(third-answer transitions are non-zero, see `tables/answer_transitions.csv`);
"latent representations are inherently insufficient".

---

## 5.3 Effectiveness and Cost of EGR

**PENDING: Final EGR specification and held-out evaluation.**

What exists: `egr/` implements V1 Contrast and Permutation, frozen in
`experiments/EGR_V1_METHOD_FREEZE_ZH.md` (2026-09-18), with results reported in
`experiments/EGR_MEDQA300_REPORT_ZH.md` and
`experiments/EGR_CROSS_BENCHMARK_REPORT_ZH.md`.

What is missing for the main text:

1. A frozen final EGR method. V1 is explicitly a *receiver-policy* comparison and
   its own freeze document states it is not a fair transport comparison against
   the latent baselines.
2. Per-record EGR artifacts. `find artifacts -name "*contrast*" -o -name
   "*permutation*"` returns nothing — only the aggregate numbers in the markdown
   reports survive. **MISSING.**
3. EGR under the three-agent ICR-V3 protocol. The existing EGR numbers were
   produced under the earlier two-agent protocol with different prompts.
4. A cost table. Construction-call counts and message-construction latency cannot
   be recovered from the surviving reports. **MISSING.**

Do not promote V1 Contrast/Permutation into the final method.

Cost material that *does* exist for the four audited channels is in
`tables/cost.csv`: receiver prompt and output tokens, text message tokens, latent
payload bytes, latent steps, and per-record receiver revision seconds. Latent
payload bytes are **not** converted to an equivalent token count. Message
construction time is recorded only for LatentMAS
(`communication_payload.communication_seconds`); for Full Text and StateBridge no
extra generation call is made, and for StateBridge the alignment cost is recorded
on the phase-1 belief record (`alignment_seconds`), not per message — the
per-message construction cost is therefore **MISSING** as a directly comparable
quantity. Wall-clock totals are not a controlled throughput comparison: worker
count per GPU varied across runs.

---

## 5.4 Analysis of Message Construction

### A. Completed E0–E4 development diagnostics — **MISSING**

There is no 75-item MedQA pilot and no E0–E4 variant in this repository. A
repository-wide search for `E0`, `E1`…`E4`, `75 items`, `n=75` and `pilot` across
`*.py`, `*.md` and `*.sh` returns nothing. Nothing can be assembled for this
subsection, including: item lists and stratification, per-variant metrics,
message lengths, E1−E0 / E3−E2 / E4−E3 paired comparisons, the E2/E3 equal-length
check, the E3/E4 identical-segment check, the offline random-mixture reference
curve, and the E0 PR failure-case audit.

If this pilot exists outside the repository it must be imported before this
subsection can be written. It must not be pooled with the ABC main results.

### B. Final EGR ablation — **PENDING** (blocked by §5.3).

### C. Case material that *is* available

`tables/answer_transitions.csv` distinguishes kept-prior, adopted-sender,
third-answer and unparsed outcomes per condition, which supports case selection
for: successfully resisting a wrong sender (PR stratum, none✓→ch✓), successful
correction (CR stratum, none✗→ch✓), correction loss (PR stratum, none✓→ch✗) and
other failures (third-answer and unparsed). Raw response text is retained in the
per-record JSON under `artifacts/icr_v3/<task>_full_seed42/revisions/*/records/`.
No causal narrative is written here.
