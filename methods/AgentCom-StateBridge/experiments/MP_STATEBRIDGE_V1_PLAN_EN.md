# MP-StateBridge V1 Experimental Plan (English)

## 0. Document Information

- Document ID: `MP-SB-V1`
- Status: pre-freeze draft
- Date: 2026-09-05
- Base method: StateBridge release `0.1.0`, commit
  `3f6bf5442c6e8848555a6132516e6d36f35444fb`
- Base model: `Qwen/Qwen3-4B`
- First validation dataset: the official StateBridge 300-item MedQA subset
- Number of paths: `m=5`
- Maximum continuous-prefix length per message: `K=64`

The Chinese source is [MP_STATEBRIDGE_V1_PLAN_ZH.md](MP_STATEBRIDGE_V1_PLAN_ZH.md).
Both documents use identical section numbering. If the two versions diverge,
the Chinese version is the current execution authority.

## 1. Research Objective

This experiment first answers one narrow question:

> Without training the model, changing the StateBridge alignment algorithm, or
> optimizing inference cost, does expanding one Planner reasoning path into
> five independently sampled paths, running an independent Planner -> Critic ->
> Refiner -> Judger chain for each path, and voting over the five final answers
> significantly improve task accuracy?

The primary V1 objective is to determine whether a performance signal exists.
It is not intended to establish the complete causal case for multi-layer
representations, multi-prefix fusion, or latent communication in one step.

## 2. Core Hypotheses

### H1: Path-Coverage Hypothesis

The probability that at least one of five independent paths is correct should
exceed single-path accuracy:

```text
Oracle@5 > Accuracy@1
```

If this does not hold, the current sampling configuration, prompt, or model is
not producing useful answer diversity.

### H2: Voting-Gain Hypothesis

When correct reasoning modes are more stable than incorrect modes, plurality
or majority voting over five paths should outperform one path:

```text
Vote@5 > Accuracy@1
```

### H3: Branch-Fidelity Hypothesis

Independent StateBridge communication should preserve the information in each
path through the final Judger without prematurely collapsing distinct paths.

## 3. V1 Scope

### 3.1 Included

1. Independently sample five Planner paths for every question.
2. Assign an independent random seed to every path.
3. Run a complete, independent four-agent chain for every path.
4. Use the official last-layer StateBridge alignment at every agent handoff.
5. Keep the branch count fixed at five throughout the pipeline.
6. Apply a frozen voting rule to the five Judger answers.
7. Persist text, tokens, hidden states, alignment metadata, answers, and runtime
   statistics.
8. Run the first authoritative evaluation on the complete 300-item MedQA
   subset.

### 3.2 Explicitly Excluded

1. Extracting messages from multiple Transformer layers.
2. Concatenating `5 x K` embeddings for one Receiver.
3. Replacing voting with an additional Final Arbiter.
4. Learned projectors, routers, gates, or Set Transformers.
5. Dynamic path counts or dynamic token-budget allocation.
6. Tree search, backtracking, or repeated branching at later agents.
7. Compression or acceleration whose primary objective is lower cost.

These extensions may begin only after V1 establishes a measurable signal.

## 4. Method Definition

### 4.1 Five-Path Generation

For question `x_i`, the Planner uses the same model and prompt with five
independent random seeds to generate:

```text
P_i,0, P_i,1, P_i,2, P_i,3, P_i,4
```

Constraints:

- Every path sees only the original question and cannot inspect other paths.
- The prompt does not explicitly ask a path to differ from other paths.
- Official decoding settings remain fixed: temperature `0.6`, top-p `0.95`.
- Every `(dataset, item_id, branch_id, stage)` receives a stable,
  reconstructable seed.
- Branch 0 is also the strictly paired `m=1` baseline.

This permits item-level comparison between `Vote@5` and branch 0 from the same
run, avoiding environment and randomness mismatches with historical results.

### 4.2 Fixed-Width Branching

Each Planner branch produces exactly one Critic, Refiner, and Judger output:

```text
Planner 0 -> Critic 0 -> Refiner 0 -> Judger 0 -> answer 0
Planner 1 -> Critic 1 -> Refiner 1 -> Judger 1 -> answer 1
Planner 2 -> Critic 2 -> Refiner 2 -> Judger 2 -> answer 2
Planner 3 -> Critic 3 -> Refiner 3 -> Judger 3 -> answer 3
Planner 4 -> Critic 4 -> Refiner 4 -> Judger 4 -> answer 4
                                                   |
                                                   +-> vote
```

No stage may create five new outputs from each input branch. The branch count
therefore remains five instead of growing exponentially.

### 4.3 Communication Within Each Branch

Planner -> Critic, Critic -> Refiner, and Refiner -> Judger all retain the
official configuration:

- final decoder-block hidden states;
- at most the final `K <= 64` states after removing the Qwen thinking section;
- independent Procrustes alignment for every message;
- `lambda=1e-3`;
- vocabulary anchoring `alpha=0.3`;
- prefix scale `1.0`;
- aligned-prefix injection through `inputs_embeds`.

The `5K` states from different paths must not be pooled into one Procrustes
fit. Every path has its own alignment statistics and rotation so that
heterogeneous paths cannot compromise one another's mapping.

### 4.4 Final Vote

MedQA answers are canonicalized to A/B/C/D. The primary result is `Vote@5`:

1. Parse every Judger output with the same official parser.
2. Count occurrences of the five canonicalized labels.
3. Select the unique label with the highest count.
4. For a `2-2-1` or another tie, select the tied label produced by the earliest
   branch.
5. Record an unparsable answer as `INVALID`; do not silently drop or resample
   it.
6. Report the tie rate, invalid rate, and complete tied-item list.

This rule must be frozen before accuracy is inspected. V1 does not use model
confidence or an additional adjudication call.

The same outputs provide `Vote@3` over branches 0-2 at no extra cost. It is a
secondary analysis and cannot replace the preregistered `Vote@5` result.

## 5. Controls and Fairness

### 5.1 Required V1 Comparisons

| Condition | Definition | Purpose |
|---|---|---|
| `Branch-0` | Complete StateBridge answer from branch 0 | Paired `m=1` primary baseline |
| `Mean-Branch` | Mean accuracy across the five branches | Detect an abnormal fixed branch |
| `Oracle@5` | Correct if any of five answers is correct | Usable diversity upper bound |
| `Vote@3` | Vote over branches 0-2 | Preliminary path-count trend |
| `Vote@5` | Vote over all five paths | Primary V1 method |

### 5.2 Post-V1 Causal Controls

If `Vote@5` produces a positive signal, the next phase must reuse the same
cached Sender paths and add:

- single-agent self-consistency@5;
- Text-MPath;
- exact-token-embedding MPath;
- a five-branch no-message pipeline;
- matched and shuffled StateBridge messages;
- a duplicated-path control that repeats one path five times.

These controls separate gains due to repeated sampling, additional agent
computation, and latent communication. Until they are complete, V1 may only
claim that the multi-path StateBridge pipeline did or did not improve the
metric. It cannot attribute the gain specifically to continuous communication.

## 6. Data and Run Configuration

### 6.1 First Authoritative Dataset

- Dataset: the repository's 300-item MedQA subset.
- Coverage: all 300 items; a 100-item subset cannot support the final result.
- Model: a fixed local `Qwen/Qwen3-4B` revision.
- Environment: record PyTorch, Transformers, CUDA, GPU, and code commit.
- Ordering: retain official item IDs; failures and resumes must not change the
  sample order.

### 6.2 Random Seeds

Use one public `base_seed` and derive stage-level seeds through a stable hash:

```text
seed = StableHash(base_seed, dataset, item_id, branch_id, stage)
```

Requirements:

- Do not use Python's process-randomized built-in hash.
- The manifest must reconstruct every seed.
- Resuming a run must not alter seeds for unfinished branches.
- Changing the number of workers must not alter seed assignment.

### 6.3 Resource Strategy

V1 loads one model on one RTX 5090 and executes the five branches
sequentially. The first implementation will not introduce dynamic batching,
which could mix variable-length generation and hook buffers across branches.

The current measured MedQA mean is approximately 54 seconds per complete
single branch. Sequential execution therefore has a rough budget of 4.5
minutes per item and 22.5 hours for all 300 items. This excludes persistence
and recovery overhead and is intended only for scheduling.

## 7. Artifact and Cache Contract

Recommended run layout:

```text
artifacts/mp_statebridge_v1/
  medqa_qwen3_4b_seed42/
    manifest.yaml
    progress.json
    results.jsonl
    events.jsonl
    states/
      item_<id>_branch_<0-4>.safetensors
    traces/
      item_<id>_branch_<0-4>.json
    reports/
      authoritative_report.md
      per_item.csv
      vote_patterns.csv
```

Every trace must include at least:

- dataset item ID, question hash, and gold label;
- branch and stage seeds;
- complete outputs from all four agents;
- generated token IDs and truncation boundaries;
- raw final-layer hidden states for all three handoffs;
- actual prefix lengths, alignment parameters, and numerical diagnostics;
- raw and parsed Judger predictions;
- token counts, timing, exceptions, and termination reasons.

Store raw hidden states in their native low precision with `safetensors`.
Aligned prefixes are reproducible from raw states, token IDs, the fixed model
embedding, and the manifest, so they need not be duplicated by default. A
reconstruction-equivalence test is mandatory.

## 8. Implementation Phases

### Phase 0: Protocol Freeze

- Freeze this plan, model revision, data hash, prompt hash, and dependency
  versions.
- Freeze voting and tie-breaking behavior.
- Define manifest and result schemas.

Completion: a complete, schema-valid manifest can be generated without
running a formal sample.

### Phase 1: Runner Implementation

- Add a branch dimension around the official StateBridge `run_item` flow.
- Fully isolate agent state and hidden-state hooks between branches.
- Implement atomic per-branch persistence, resume behavior, and explicit retry
  records.
- Implement voting and item-level paired statistics.

Completion: all unit tests pass, and `m=1` produces field-level equivalence
with unwrapped StateBridge under the same seed.

### Phase 2: Smoke Test

- Run the first five fixed items with five branches each.
- Manually inspect all 25 complete chains.
- Confirm that branches use distinct random streams.
- Confirm that prefixes, tokens, and caches never cross branch boundaries.
- Confirm stop/resume invariance.

Completion: no structural error or cross-item/branch leakage, and every output
is reconstructable.

### Phase 3: Full MedQA Run

- Complete 300 items x 5 branches.
- Persist immediately after each branch.
- Do not tune any parameter based on intermediate accuracy.
- Retry only system failures under the preregistered policy; never resample an
  ordinary incorrect answer.

Completion: all 1,500 branch results reach a terminal state, with every missing
or system-error result explicitly explained.

### Phase 4: Authoritative Report

- Compute Branch-0, each branch, Mean-Branch, Vote@3, Vote@5, and Oracle@5.
- Report vote patterns, answer disagreement, and per-item correction/harm.
- Compare `Vote@5` with Branch-0 using a paired bootstrap 95% interval and
  McNemar's test.
- Produce the accuracy confusion matrix, invalid outputs, and tied-item list.
- Freeze hashes for the report and result files.

Completion: one command can fully rebuild the report from `results.jsonl`.

### Phase 5: Next-Research Decision

Use the decision rules in Section 10 to choose causal communication controls,
aggregation research, or diversity repair.

## 9. Test Checklist

- [ ] `m=1` wrapper equivalence test
- [ ] Five branch seeds are unique and reconstructable
- [ ] Same-seed reruns are identical
- [ ] Resume does not overwrite a completed branch
- [ ] Hidden-state and token counts remain aligned
- [ ] Every Procrustes fit uses only its own branch
- [ ] No branch reads another branch's output
- [ ] Voting covers 5-0, 4-1, 3-2, 3-1-1, 2-2-1, and INVALID cases
- [ ] Parser behavior matches official single-path evaluation
- [ ] All 300 item IDs are unique and complete
- [ ] Every report metric can be recomputed from raw results

## 10. Decision Rules

The primary comparison is `Vote@5 - Branch-0`.

### PASS

- The `Vote@5` point estimate exceeds Branch-0.
- Corrections outnumber harms.
- The paired 95% interval excludes zero and the two-sided McNemar test has
  `p < 0.05`.

Conclusion: reliable five-path voting gain exists. Proceed to causal
communication controls and single-pass multi-prefix fusion.

### PROMISING

- The `Vote@5` point estimate exceeds Branch-0.
- The confidence interval or significance test does not pass.

Conclusion: a positive but insufficient signal exists. Extend to ARC-C/GSM8K
or additional independent seeds; do not claim a significant improvement.

### AGGREGATION BOTTLENECK

- `Oracle@5` clearly exceeds Branch-0.
- `Vote@5` does not improve.

Conclusion: correct paths exist but voting cannot identify them. Prioritize
confidence weighting, a verifier, or a Final Arbiter.

### DIVERSITY FAILURE

- `Oracle@5` remains close to Branch-0.
- Path answers and representations are highly similar.

Conclusion: current sampling does not create useful diversity. Repair sampling
or introduce explicit branching before attempting complex fusion.

### FAIL

- `Vote@5` is below Branch-0 and harms outnumber corrections.

Conclusion: incorrect answers are more consistent than correct answers.
Analyze correlated errors and do not retroactively change the voting rule to
rescue the primary result.

## 11. Claim Boundary

After a V1 PASS, the supported claim is:

> Under the Qwen3-4B MedQA protocol, preregistered voting over five independent
> StateBridge reasoning branches significantly outperforms the paired
> single-branch result.

V1 alone does not establish that:

- StateBridge is more effective than text communication;
- continuous hidden states add information beyond exact token embeddings;
- multi-agent inference beats compute-matched single-agent self-consistency;
- hidden states from multiple layers are explicit reasoning paths;
- `m=5` is a universally optimal path count.

Those claims require the strict controls in Section 5.2 or separate follow-up
experiments.

## 12. Follow-Up Roadmap (Outside V1)

Only after V1 exposes a usable signal, proceed in this order:

1. Reuse the cached paths for Text, exact-token, no-message, shuffled, and
   single-agent self-consistency controls.
2. Compare fixed `K=64` per-path budgets with a fixed total-prefix budget.
3. Fuse all five independent prefixes in a single Receiver pass.
4. Compare majority voting, confidence voting, and a Final Arbiter.
5. Evaluate multi-layer hidden states separately as multiple views of one path.
6. Study multiple-subspace alignment only if layer views add Oracle diversity.

This ordering ensures that the first experiment changes one core factor only:
one stochastic reasoning chain becomes five independent stochastic chains.
