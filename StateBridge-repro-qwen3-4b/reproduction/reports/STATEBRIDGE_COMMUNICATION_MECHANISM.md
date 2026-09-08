# StateBridge Communication Mechanism Audit

## Scope

This note distinguishes the behavior of the released implementation from
causal explanations that still require controlled experiments. The inspected
upstream commit is `3f6bf5442c6e8848555a6132516e6d36f35444fb`.

## End-to-end channel

StateBridge reuses one frozen model in a homogeneous four-stage pipeline:

```text
Planner --continuous prefix--> Critic
Critic  --continuous prefix--> Refiner
Refiner --continuous prefix--> Judger
```

Every stage also sees the original question. Only the immediately preceding
stage's message is injected directly; earlier information can persist only
after being integrated into the next stage's output. No model parameter or
projector is trained.

For each non-final sender, a forward hook records one final-transformer-layer
hidden vector per decoding step. The implementation searches for the end of
the Qwen thinking block and retains at most the last 64 post-thinking vectors
and the corresponding generated token IDs. If no valid thinking boundary is
found, it retains the last 64 generated states.

### Exact hook semantics

With the default `use_hook=True`, the hook is attached to
`model.model.layers[-1]`. During the initial generation forward pass it keeps
only the last prompt-position output; during cached autoregressive steps it
keeps the single newest position. The resulting sequence contains the states
whose LM-head logits produce the successive sampled tokens. It is therefore a
predictive pairing `h_i -> y_i`, not a post-hoc lookup of a state after `y_i`
has already been emitted.

For Qwen3 in Transformers 4.51.3, the last decoder block runs before the
model-level final RMSNorm. The hook therefore captures the last block's
pre-final-norm output. By contrast, the released `use_hook=False` fallback
collects `outputs.hidden_states[-1]`, which is after the final RMSNorm. The two
paths are not numerically identical even though the release presents the hook
as a memory-saving extraction route. The main published-style path uses the
hook, and later norm calibration reduces but does not mathematically erase this
difference.

## Per-message alignment

Let `H` be the retained final-layer states and `E` the input embeddings of the
corresponding generated tokens. Both have shape `K x d`, with `K <= 64`.
StateBridge computes a new mapping for every message:

1. Center `H` and `E` independently.
2. Form regularized covariances with `lambda=1e-3`.
3. Whiten both point sets.
4. Solve an orthogonal Procrustes problem by SVD.
5. Recolor the rotated states with the covariance and mean of `E`.
6. Rescale each vector to the global mean vocabulary-embedding norm.
7. Find its nearest vocabulary embedding by cosine similarity and interpolate
   30% toward that anchor.

In compact form:

```text
H_w = (H - mu_H) C_H^(-1/2)
E_w = (E - mu_E) C_E^(-1/2)
SVD(H_w^T E_w) = U Sigma V^T
R = U V^T
Z = H_w R C_E^(1/2) + mu_E
Z_final = 0.7 norm_calibrate(Z) + 0.3 nearest_vocab(Z)
```

The code forces `det(R)=+1`, so the Procrustes component is a proper rotation
rather than a reflection. Orthogonality preserves distances and angles in the
whitened space; whitening, recoloring, norm calibration, and anchoring mean
that the complete map does not literally preserve the geometry of the raw
hidden space.

On Qwen3-4B, `d=2560` and the current batch size is one, so a full message uses
only 64 paired observations to form regularized `2560 x 2560` covariance and
cross-correlation matrices. The unregularized sample covariances have rank at
most 63. Adding `1e-3 I` makes the eigendecompositions invertible, but the map
outside the observed message subspace is necessarily weakly identified. The
same K pairs are used both to fit the per-message map and to produce the vectors
that are sent; this is alignment of the current point cloud, not training or
validation of a reusable global projector.

## Receiver injection

The aligned sequence is inserted at a marker inside the receiver prompt and
passed through Hugging Face generation as `inputs_embeds`. Its attention-mask
entries are one and its positions are sequential with the surrounding prompt,
so the receiver processes the vectors as continuous pseudo-token positions.

This is input-layer communication, not KV-cache transfer. It uses `O(Kd)`
message storage and does not depend on matching or modifying every transformer
layer. In the main MedQA run, 282 of 300 items transferred exactly 64 positions
at all three transitions; the remaining 18 had at least one shorter suffix.

## Why the channel can help

### 1. It corrects a representation-interface mismatch

Final-layer states are optimized for the LM head and do not follow the norm,
mean, covariance, or orientation of token embeddings. Raw-state insertion at
the input is therefore out of distribution. Whitening plus Procrustes provides
a local coordinate bridge, while recoloring, norm calibration, and vocabulary
anchoring make the resulting vectors less surprising to the receiver.

### 2. It can retain information beyond discrete token identity

A final-layer state reflects the context and predictive computation that
produced a token, not only that token's ID. Aligning it near the corresponding
input embedding can preserve a continuous residual containing uncertainty,
relations, exclusions, or task context that hard text communication discards.
The 30% anchor keeps the vector readable without fully quantizing that residual.

### 3. It creates a conclusion-focused bottleneck

Only the final post-thinking suffix is transferred. This often contains the
sender's diagnosis, selected strategy, or corrected plan rather than its whole
reasoning trajectory. The receiver receives a compact conditioning sequence
instead of a long textual transcript.

### 4. Role decomposition adds iterative test-time computation

Planner, Critic, Refiner, and Judger offer repeated opportunities to propose,
challenge, repair, and execute a solution. This can improve accuracy even when
the communication channel itself is not uniquely beneficial. Any claim about
StateBridge-specific gain must therefore compare against compute- and
role-matched controls.

### 5. Input-level injection is comparatively portable

Every causal language model consumes input embeddings, while KV layouts and
layer semantics vary across architectures. Restricting communication to the
input interface avoids many architecture-specific failure modes of cross-layer
KV injection.

## Evidence and limits

The paper's direct Qwen3-4B comparison is:

| Task | TextMAS | StateBridge | StateBridge - TextMAS |
|---|---:|---:|---:|
| ARC-Challenge | 90.0 | 93.7 | +3.7 |
| MedQA | 65.3 | 70.3 | +5.0 |
| GSM8K | 89.8 | 89.8 | 0.0 |
| MBPP+ | 69.8 | 75.9 | +6.1 |
| HumanEval+ | 79.7 | 82.3 | +2.6 |
| Macro average | 78.9 | 82.4 | +3.5 |

The paper states that TextMAS and StateBridge use the same four roles,
hyperparameters, and evaluator, with the communication-modality wording as the
prompt difference. This is the relevant text-channel comparison. The local B0
and B1 text results use a different number of agents, greedy decoding, disabled
thinking, and different prompts, so they must not be substituted into this
table.

The authors report that, on the Qwen3-4B task average, full StateBridge scores
82.4, ridge alignment 74.9, no norm calibration 79.5, no vocabulary anchoring
80.2, and a random-noise prefix 48.8. They also report a 2.4-point gain over
the best Qwen3-4B baseline and a 4.0-point gain on MedQA. These results support
the importance of semantic, geometry-compatible prefixes, but the released
repository omits the paper baseline and ablation scripts, so the controls
cannot yet be locally audited.

Random noise is not a sufficient no-communication control: an out-of-
distribution prefix can actively damage the receiver. Likewise, matching the
paper's absolute MedQA score demonstrates executable reproducibility but does
not identify the cause of the score.

The paper does not include a matched-message versus shuffled-message test, an
exact last-64 token-embedding control, or a length-matched text-suffix control.
Consequently, its experiments do not fully separate semantic matching,
continuous residual information, message length, and soft-prompt perturbation.
The released repository also omits TextMAS and ablation implementations, so the
claimed control equivalence cannot be independently audited from code alone.

The alignment also uses generated token IDs as paired anchors and for nearest-
vocabulary snapping. StateBridge therefore does not avoid discretization
entirely; it uses token identity to construct a continuous message. A direct
token-embedding suffix control is necessary to show that the continuous
residual contributes beyond the same last 64 token identities.

Finally, this is query-specific communication: every sender and receiver sees
the same test question. It does not by itself demonstrate a reusable policy
message inferred from support examples and applied to held-out receiver states.

## Decisive local audit

Cache each sender's generated states and token IDs once, then evaluate the same
receiver items with per-item paired seeds under these conditions:

1. matched aligned prefix;
2. shuffled aligned prefix from another question;
3. answer-label-matched shuffled prefix;
4. no prefix with an otherwise identical receiver prompt;
5. length- and norm-matched random prefix;
6. raw hidden-state prefix;
7. exact token-embedding suffix for the same 64 generated tokens;
8. nearest-token-only prefix, equivalent to full quantization;
9. explicit text suffix; and
10. compute-matched single-agent or text-role decomposition.

Use paired rescue/harm counts, McNemar tests, bootstrap confidence intervals,
gold-answer logit-margin changes, and accuracy. The interpretation is then:

- matched greater than shuffled/random/no-message: message semantics matter;
- aligned greater than raw hidden: the representation bridge matters;
- aligned greater than token embeddings/text/nearest-only: continuous residual
  information matters;
- full pipeline greater than compute-matched role controls: the gain is not
  explained only by additional inference and prompting.
