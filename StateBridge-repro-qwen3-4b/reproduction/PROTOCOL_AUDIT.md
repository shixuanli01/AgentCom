# StateBridge Qwen3-4B Reproduction Audit

## Public target

The paper reports Qwen3-4B StateBridge results on five core tasks only:

| Task | Rows | Paper score | Implied count |
|---|---:|---:|---:|
| ARC-Challenge | 1,172 | 93.7% | 1,098 |
| MedQA | 300 | 70.3% | 211 |
| GSM8K | 1,319 | 89.8% | 1,184 |
| MBPP+ | 378 | 75.9% | 287 |
| HumanEval+ | 164 | 82.3% | 135 |

AIME24, AIME25, and GPQA are not Qwen3-4B targets in Table 1. They are only
reported for Qwen3-8B and Qwen3-32B.

## Released method path

Each item runs Planner, Critic, Refiner, and Judger sequentially with one
shared frozen Qwen3-4B model. After each of the first three agents, the code:

1. captures final-transformer-layer states from generation using a hook;
2. keeps at most the last 64 post-thinking states and their generated tokens;
3. aligns states to the corresponding token-embedding points through
   regularized whitening and orthogonal Procrustes;
4. calibrates each vector to the vocabulary mean norm;
5. interpolates it by 0.3 toward its nearest vocabulary embedding; and
6. inserts the resulting continuous prefix at the prompt marker for the next
   agent through `inputs_embeds`.

The released default prefix scale is 1.0. Commit `f259374` explicitly says
that this changed the public default to match the paper.

## Evaluation

- Multiple choice: parse the final boxed answer and compare its normalized
  letter to gold.
- GSM8K: parse the last number (preferring a boxed number) and exact-match it.
- Code: extract the last Python Markdown block, append the dataset's `test`
  program, and execute it in a fresh subprocess with a ten-second timeout.

The EvalPlus Hugging Face rows used by the loader contain large generated
`test` programs. The released runner executes those programs directly; it does
not invoke the external `evalplus` CLI.

## Reproducibility gaps

The released package is a core-method release, not a complete reproduction
package. In particular:

1. Sampling is enabled at temperature 0.6 and top-p 0.95, but neither the
   paper nor README states the seed or whether scores aggregate repeated runs.
2. The paper used two A100-80G GPUs. The worker seed, when supplied, is
   `seed + gpu_id`, while dynamic queue scheduling determines which example
   reaches each RNG stream. Two-GPU per-example outputs are therefore not
   guaranteed deterministic across launches.
3. PyTorch is unpinned and Transformers is constrained only to `>=4.51.0`.
   Generation behavior may vary across compatible releases.
4. Runtime Hugging Face dataset revisions are unpinned.
5. The CLI passes `input_len + task_max_new_tokens` as `max_new_tokens`.
   This exceeds the output limits described by the paper, although normal
   generations may stop at EOS before reaching that bound.
6. Resume reconstructs sample indices from progress counters in the log. It
   is reliable for this reproduction's one-GPU FIFO track, but can associate
   completed outputs with the wrong dataset rows after an interrupted
   multi-GPU run that completed examples out of order.

## Frozen local track

The main local track uses one RTX 5090, seed 42, Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, Transformers 4.51.3, and
PyTorch 2.7.1+cu128. CUDA 12.8 is a hardware compatibility substitution for
the upstream cu124 requirement; no method code is changed.

Because the author seed and software lock are unavailable, matching the five
paper cells is an empirical outcome, not something the public artifacts make
uniquely reproducible. We will not search seeds against test accuracy.
