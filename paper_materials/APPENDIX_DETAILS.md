# AgentCom — Appendix details

Read-only audit, 2026-09-21. Facts are taken from the on-disk run configuration
and per-record artifacts; the markdown reports under `experiments/` were used only
to locate material and never to override a raw record.

---

## Appendix A — Reproducible setup

### Model and backend (VERIFIED, `models.py:75-95`)

| Field | Value |
|---|---|
| Checkpoint | `Qwen/Qwen3-4B` (`config.json: model`) |
| Revision | **MISSING** — no revision or commit pin is recorded in the run config |
| Precision | `torch.bfloat16` on CUDA (`torch.float32` on CPU) |
| Backend | HuggingFace `transformers` `AutoModelForCausalLM`, eager `.generate()`; no vLLM/TGI |
| Attention impl. | **MISSING** — `attn_implementation` is not set, so the transformers default applies |
| `use_cache` | True |
| Chat template | `enable_thinking=True` at both phases (`icr/runtime.py:71`, `:103`) |

### Sampling (VERIFIED, `config.json: generation`)

`do_sample=true`, `temperature=0.6`, `top_p=0.95`, `top_k=null`,
`max_new_tokens=16384` for all five datasets and both phases.
`max_new_tokens_history` records the raise: `8192 → 16384`, reason
"truncated generations never stated an answer and were scored wrong".
GSM8K and ARC-C also passed through intermediate budgets; the on-disk config is
authoritative, and superseded fingerprints are listed in
`config.json: superseded_fingerprints`.

### Seeds (VERIFIED, `icr/protocol.py:86-128`)

`global_seed = 42`, `replication_id = "seed_pair_00"`. Seeds are derived, never
drawn:

```
prebelief_seed  = sha256(global_seed ‖ replication_id ‖ item_id ‖ "agent_<X>_pre")[:8] mod (2^31−1)
revision_seed   = sha256(global_seed ‖ replication_id ‖ item_id ‖ direction ‖ "revision")[:8] mod (2^31−1)
```

A, B and C differ only through the `agent_<X>_pre` field; the prompt is byte-identical.
**The revision seed does not include the condition**, so all four channels share one
seed per (item, direction). This is what makes the §5.2 B comparison a matched design.

### Prompts (VERIFIED)

Phase 1: `solver_prompt_version = icr_v2_phase1_verbatim`
(`icr/protocol.independent_solver_prompt`). MedQA rendering:

```
You are an independent problem-solving agent.

Solve the medical multiple-choice question carefully and independently.

Reason from the evidence in the question.
Do not assume another agent will review your answer.

At the end, return exactly one final option in benchmark-compatible form: \boxed{A}, replacing A with one of A, B, C, or D.

Your response should contain:
1. your reasoning
2. your final answer

Medical multiple-choice question:
{question}
```

GPQA appends a disambiguation line so the model answers with the final A–D list
rather than the lowercase a)–d) items quoted inside the options
(`GPQA_LAYER_DISAMBIGUATION`).

Phase 2: `revision_prompt_version = icr_v3_mid_injection`
(`icr/prompts_v3.REVISION_TEMPLATE`, sha256 `e176b57fb1a8b62a…`, 627 chars;
code variant `CODE_REVISION_TEMPLATE`, sha256 `cfce430522a2f655…`, 681 chars):

```
You previously solved this problem independently.

Original problem:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_block}

Your task is to REVISE your belief, not to restart from scratch.

Evaluate your previous reasoning and any external message critically.

* Change your answer only if you find a concrete error in your previous reasoning, or evidence that is better supported than it.
* Do not change your answer merely because an external message is present.
* Resolve any disagreement using the evidence in the original problem.

{answer_format}
```

The slot order is Task → receiver's own prior → external information → how to
integrate → output format, and only `{external_block}` varies by condition
(`external_block` source sha256 `bf8a13d634f032e1…`):

| Condition | `{external_block}` |
|---|---|
| `none` | `No external message is available.` |
| `*_text`, `*_evidence` | `An external message from another reasoning process is available.\n\nExternal message:` + sender reasoning |
| `*_statebridge`, `*_latentmas` | same lead-in + `[EMBEDDING_CONTEXT_HERE]` marker |

Answer formats (`ANSWER_FORMAT`, per task):

| Task | sha256 | Text |
|---|---|---|
| medqa | `dc7804e2a1e8cb37…` | `\boxed{X}`, X ∈ {A,B,C,D} |
| arc_challenge | `e3e77d5306590851…` | `\boxed{X}`, X ∈ {A,B,C,D} |
| gpqa | `17997fb00d278f09…` | `\boxed{X}`, X ∈ {A,B,C,D} |
| gsm8k | `93365f0aae549837…` | `\boxed{N}`, plain digits, no thousands separators, no units |
| humanevalplus / mbppplus | `a20f404140edfe25…` | one markdown Python block, no tests or prose inside it |

### Versioning and caching (VERIFIED)

| Field | Value |
|---|---|
| Protocol | `ICR-V3` |
| Config fingerprint | per dataset, `config.json: fingerprint` (MedQA `355c5073cd75bb6f…`) |
| Superseded fingerprints | `config.json: superseded_fingerprints` (MedQA `96d887fc74fa2023…`) |
| Dataset hash | `config.json: dataset_sha256` (MedQA `ddee8dc64d3b2a1e…`) |
| Answer parser | `icr.parsing_v3@ICR-V3` |
| Code version | see `manifest.json: git_commit` |

Cache reuse (`--global-resume`) accepts a record only when `status == "complete"`
and its `config_fingerprint` is in `accepted_fingerprints(config)`. That check does
**not** verify the receiver's prior; this audit verified it independently and found
**0 stale priors across all five datasets** (`tables/integrity_checks.csv`).

### Metric-name mapping

| Artifact column | Paper name |
|---|---|
| `cr` | CR (correction rate) |
| `pr` | PR (preservation rate) |
| `si` | SI |
| `sr` | SR |
| `scr` | SCR |
| `accuracy` | Acc_ret (retained-subset accuracy) |
| `fcs`, `fws`, `follow_selectivity` | sender-following diagnostics; **not reported on code tasks** |

---

## Appendix B — Communication channel implementations

This describes the adapted implementations in this repository. None of them is the
original full system.

| | Full Text | StateBridge | LatentMAS |
|---|---|---|---|
| Sender information read | phase-1 reasoning text | last 64 hidden states of the phase-1 trajectory | full all-layer KV cache of the phase-1 trajectory |
| Selection | whole text | `last_k`, K=64 of T positions | all `cache_positions` (1,401 in the sampled MedQA record), 36 layers |
| Alignment | none | Procrustes, `adaptive_reg=0.001`, `snap_ratio=0.3`, `prefix_scale=1.0`, `prefix_strategy=scale` | none |
| Injection | text inside the message slot | `inputs_embeds` spliced at the `[EMBEDDING_CONTEXT_HERE]` marker, i.e. exactly the text slot | `past_key_values` causal prefix before the prompt |
| Position is a free parameter | yes | yes | **no** — RoPE is baked into the cache |
| Extra generation | none | none | 10 latent steps (`--latent-steps 10`) |
| Extra forward passes | none | one alignment per sender belief (`alignment_seconds`, recorded on the belief) | teacher-forces the sender trajectory, then 10 latent steps |
| Payload recorded | `characters`, `tokens`, `payload_bytes` | `states=64`, `hidden_dimension=2560`, `dtype=bfloat16`, `payload_bytes=327,680` | `cache_layers=36`, `cache_positions`, `kv_payload_bytes`, `cache_storage="reconstructible_reference"` |
| Trajectory rebuilt | no | no | yes (reconstructible reference, not stored verbatim) |

Differences from the native frameworks: StateBridge uses
`methods/state_bridge.py` unmodified for the alignment algorithm but is driven by
the ICR receiver prompt rather than its own pipeline; LatentMAS is reduced to a
single sender→receiver hop with one revision, with no multi-round workspace. The
fairness limit stated in the main text — LatentMAS has no explicit textual message
slot, so its message cannot occupy the same prompt position — follows from the
`past_key_values` prefix, not from a design choice made here.

---

## Appendix C — Sample and scoring audit

### Belief-pattern distribution (VERIFIED, `tables/sample_statistics.csv`)

| Dataset | 3 correct | 2 correct | 1 correct | 0 correct | Items |
|---|---:|---:|---:|---:|---:|
| MedQA | 180 | 26 | 32 | 62 | 300 |
| ARC-C | 1,067 | 32 | 17 | 49 | 1,165 |
| GSM8K | 1,225 | 23 | 23 | 48 | 1,319 |
| GPQA-D | 80 | 24 | 33 | 61 | 198 |
| HumanEval+ | 136 | 10 | 4 | 14 | 164 |

### Per-direction detail

`tables/six_direction_detail.csv` gives, for each dataset × condition × direction,
the numerator and denominator of CR, PR, SR and SCR separately for all six ordered
directions.

### Exclusions, skips, failures, reuse (VERIFIED, `tables/exclusions_and_reuse.csv`)

- **Pre-registered structural exclusion:** ARC-Challenge only, 7 items
  (`121, 385, 400, 836, 868, 1037, 1042`), rule
  `trailing option count != 4 (label-free, pre-registered)`. Decided by question
  structure alone, never by gold or prediction. All other datasets have
  `excluded_item_ids = []`.
- **Skipped, not excluded:** items every agent answered correctly are not run in
  phase 2. They remain in the population and are handled in Appendix E.
- **Non-terminating generations are NOT excluded.** They are retained and scored
  wrong, because no answer is emitted. The rule depends only on whether generation
  terminated (`hit_eos`), never on correctness. Phase-1 non-termination rates:
  MedQA 0.11%, ARC-C 0.06%, GSM8K 0.05%, GPQA-D 2.19%, HumanEval+ 2.44%.
  Several published commit messages claim these items are excluded; they are not,
  and AUDIT_ISSUES A1 is the correction of record. A full exclusion sensitivity is
  in `tables/truncation_sensitivity.csv`.
- **Cache reuse:** `tables/integrity_checks.csv` lists the shard provenance of every
  record (e.g. MedQA `rank0_v3`, ARC-C `rank*_v3` and `rank*_fill`). Zero duplicate
  `(item_id, direction, condition)` keys in any dataset.

### Scoring and parser

`answer_parser_version = icr.parsing_v3@ICR-V3`. Handling of degenerate outcomes:

| Outcome | Field | Scored as |
|---|---|---|
| Generation hits EOS, answer parses | `hit_eos=true`, `parsed_answer` set | correct or wrong on comparison |
| Generation exhausts `max_new_tokens` | `hit_eos=false` | `parsed_answer=None` → **wrong** |
| Answer does not parse | `parsed_answer=None` | **wrong** |
| Worker crash | record absent | not counted; `--require-complete` refuses to merge |

Counts per dataset are in `tables/integrity_checks.csv`
(`beliefs_truncated`, `revisions_truncated`, `revisions_unparsed_answer`).

### Consistency checks run in this audit — all PASS

| Check | Result |
|---|---|
| Closed-form counting identity, all datasets | PASS (20/20 cells) |
| Duplicate `(item, direction, condition)` keys | 0 in all five datasets |
| Receiver prior byte-identical to the current belief | 3,016 / 3,184 / 3,194 / 3,327 / 768 pristine, 0 stale |
| All four conditions share one receiver prior per (item, direction) | 0 divergent pairs |
| All four conditions cover the identical evaluation set | PASS (HumanEval+ verified explicitly: 168 pairs each) |
| `Acc_mixed = SI` | PASS (20/20 cells) |
| `status != complete` records | 0 |

### Development / test split

**MISSING.** No item-level development-vs-test assignment is recorded in any
config. Phase-1 budgets were raised after observing truncation on the same items
that are analysed, so there is no held-out set; this is an audit of a single
frozen run, not a tuned-then-evaluated protocol. Stated as a limitation, not
repaired.

---

## Appendix D — HumanEval+ supplementary results

Full results are in `tables/main_results.csv`, `tables/sr_scr.csv`,
`tables/paired_vs_no_message.csv` and `tables/answer_transitions.csv`
(rows `HumanEval+`).

| Condition | Acc_ret | CR | PR | SI |
|---|---|---|---|---|
| No Message | 71/168 = 42.26% | 16/28 = 57.14% | 26/28 = 92.86% | 75.00% |
| Full Text | 76/168 = 45.24% | 21/28 = 75.00% | 26/28 = 92.86% | 83.93% |
| StateBridge | 73/168 = 43.45% | 19/28 = 67.86% | 27/28 = 96.43% | 82.14% |
| LatentMAS | 67/168 = 39.88% | 24/28 = 85.71% | 17/28 = 60.71% | 73.21% |

Verifications requested:

1. **Identity and stage of the non-terminating generations** —
   `tables/humanevalplus_degenerate_loops.csv`. Phase 1: 12 belief records over
   **8 distinct items** (`32, 34, 90, 94, 113, 126, 134, 158`). Phase 2: 49 revision
   records over **10 distinct items** (`32, 80, 90, 91, 94, 97, 113, 126, 134, 151`).
   The two sets are **not** the same; their intersection is 6 items. The figure "12"
   in earlier notes refers to phase-1 *records*, not items.
2. **Same evaluation set for all channels** — VERIFIED. Each of the four conditions
   covers exactly the same 168 (item, direction) pairs.
3. **Exclusion rule fixed in advance** — no HumanEval+ exclusion was applied at all
   (`excluded_item_ids = []`). Non-terminating generations are retained and scored
   wrong under a rule that depends only on whether generation terminated.
4. **Contamination of the analysed subsets** — `tables/truncation_contamination.csv`.
   **32.14% of the CR denominator and 32.14% of the PR denominator** involve at
   least one non-terminating belief, and 21.74% of SR. For comparison: MedQA 0.00%
   CR/PR, ARC-C 2.04%, GSM8K 4.35%.
5. **Literal-newline handling in code inputs** — **MISSING**. This audit did not
   locate a recorded instance; the raw prompts are retained per record
   (`prompt_sha256`, `raw_response`) and the check can be run against them, but no
   evidence is presented here.

Interpretation limit: with 28 CR pairs and 28 PR pairs, one record is 3.57
percentage points, and a third of those pairs rest on a generation that never
terminated. These are properties of **this run** — budget, sampling settings and a
strict single-code-block output contract — not a statement about the quality of the
HumanEval+ dataset. Per-item pass/fail is also not evidence about whether the
receiver adopted the sender's program; only whitespace-normalised string identity
is reported, in `tables/answer_transitions.csv`.

---

## Appendix E — Extrapolated accuracy and sensitivity

### ESTIMATED, NOT MEASURED FULL-SET ACCURACY

`tables/full_set_accuracy_estimated.csv`. Every row carries this label.

Formula:

```
Acc_full = ( retained_correct + sample_correct + imputed_records × skipped_item_accuracy )
           / ( items × 6 )
```

| Dataset | Items × 6 | Retained (measured) | Skipped sample (measured) | Imputed | Coverage |
|---|---:|---:|---:|---:|---:|
| MedQA | 1,800 | 720 | 34 | 1,046 | 41.9% |
| ARC-C | 6,990 | 588 | 208 | 6,194 | 11.4% |
| GSM8K | 7,914 | 564 | 235 | 7,115 | 10.1% |
| GPQA-D | 1,188 | 708 | 124 | 356 | 70.0% |
| HumanEval+ | 984 | 168 | 24 | 792 | 19.5% |

| Dataset | No Message | Full Text | StateBridge | LatentMAS |
|---|---|---|---|---|
| MedQA | 70.28% [67.33, 73.23] | 70.89% [67.94, 73.84] | 71.56% [68.61, 74.51] | 69.89% [66.94, 72.84] |
| ARC-C | 93.93% [93.13, 94.74] | 94.13% [93.33, 94.94] | 94.11% [93.30, 94.91] | 93.52% [92.38, 94.67] |
| GSM8K | 94.74% [94.02, 95.47] | 94.68% [93.96, 95.40] | 94.79% [94.07, 95.52] | 94.69% [93.96, 95.42] |
| GPQA-D | 54.58% [53.67, 55.48] | 56.32% [55.68, 56.97] | 55.75% [54.96, 56.53] | 55.06% [54.41, 55.71] |
| HumanEval+ | 86.69% [73.75, 99.63] | 90.65% [79.54, 100.00] | 90.35% [79.24, 100.00] | 89.74% [78.63, 100.00] |

Intervals are Wilson half-widths on the skipped-population sample, scaled by the
imputed share. They cover the extrapolation only and are common to all four
channels, so channel *differences* are tighter than the intervals suggest.

### Requested verifications

1. **Observed and imputed records are disjoint** — VERIFIED. The retained scope
   (`revisions/merged.jsonl`) and the skipped scope
   (`revisions/all_correct_sample.jsonl`) partition the item set by the
   all-three-correct predicate; the estimator reads sample rows only from the second
   file.
2. **Denominator matches the target population** — VERIFIED.
   `items × 6 = retained + skipped` in all four datasets.
3. **SCR is NOT borrowed from partially-correct triples** — VERIFIED and material.
   The estimator uses the rate measured on the all-correct items themselves. The
   two rates differ substantially: on ARC-C, No Message scores 93.8% on
   mixed-item both-correct pairs but 100.0% on all-correct items (208 records).
   Borrowing the mixed-item rate put ARC-C's full-set accuracy 5.7 points low and
   produced a 5.9-point No Message ↔ Full Text gap that direct measurement puts at
   0.20 points. **This is the single most consequential estimator choice in the
   appendix.**
4. **The MedQA extra 34 records are not double-counted** — VERIFIED. They are
   present in `all_correct_sample.jsonl` and absent from `merged.jsonl`
   (2,880 = 720 × 4). See AUDIT_ISSUES #2 for their provenance.

### Assumption and sensitivity

The single assumption is that the unsampled skipped records behave like the
sampled ones from the same population. A worst-case sensitivity — treating
`skipped_item_accuracy` as its Wilson lower bound instead of its point estimate —
is what the CI column already expresses. No sensitivity figure is presented as a
measured improvement, and no missing records were regenerated to close the gap.

### GPQA-Diamond

Its phase-2 run originally covered the skipped population in full, which would have
made its full-set accuracy measured rather than extrapolated. That was cut short
deliberately at 04:53 UTC to save roughly 3.8 hours of the remaining run, leaving
495 records on all-correct items — 124 per condition, 25.8% of that population,
covering 30 of the 80 all-correct items. Those records were kept, not discarded.

The result is that GPQA-Diamond is still extrapolated like the others, but from
the largest measured share in the study: 70.0% direct coverage against 41.9%
(MedQA), 19.5% (HumanEval+), 11.4% (ARC-C) and 10.1% (GSM8K). Its intervals are
correspondingly the tightest, at roughly ±0.7 points.
