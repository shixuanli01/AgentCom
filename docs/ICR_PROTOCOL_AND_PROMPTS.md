# Frozen ICR protocol and prompt record

This document makes the experiment reconstructable without relying on conversational history. The executable source of truth remains `methods/AgentCom-StateBridge/icr/protocol.py` and `runtime.py` at the frozen Git commit.

## System prompt

```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

The Qwen chat template is rendered with `add_generation_prompt=True` and `enable_thinking=True`, after which `<think>` is appended.

## Independent solver user prompt

```text
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

## Revision user prompt

```text
You previously solved this question independently.

Original question:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_section}

Your task is to REVISE your belief, not simply restart from scratch.

Evaluate your previous reasoning and the external information critically.

* Do not change your answer merely because another message exists.
* If the external information provides stronger evidence or identifies a real error, revise.
* If your original reasoning remains better supported, keep it.
* Resolve disagreements using the evidence in the original question.

Return:
1. concise revised reasoning
2. exactly one final benchmark-compatible answer in the form \boxed{A}, replacing A with one of A, B, C, or D.
```

## Communication rendering

For `none`:

```text
No external message is available.
```

For a Text message:

```text
You may now have access to an external message from another reasoning process.

External message:
{message_text}
```

For StateBridge or LatentMAS, the following modality-neutral preamble is placed at the beginning of the user turn:

```text
You may now have access to an external message from another reasoning process.

External message:
[EMBEDDING_CONTEXT_HERE]
```

The visible revision template then uses:

```text
Use the external message above as evidence if it is relevant.
```

`[EMBEDDING_CONTEXT_HERE]` is removed from tokenized text. Its token position defines where the continuous prefix/KV communication is injected. The marker appears before the original question and revision instructions.

## StateBridge configuration

```text
max_prefix_tokens = 64
selection_method = last_k
prefix_strategy = scale
adaptive_reg = 1e-3
snap_ratio = 0.3
use_hook = true
enable_thinking = true
```

The handoff is generated from the cached sender first-pass hidden states. The saved `.safetensors` object contains `statebridge_prefix`.

## LatentMAS reconstruction

- The exact cached sender prompt is rebuilt and checked against its saved token count and SHA256.
- Cached `generated_token_ids` are concatenated to that prompt.
- The resulting sequence is teacher-forced through Qwen; no sender generation is resampled.
- The formal run uses 10 latent steps.
- KV payload size, layers, dtype, communication time, and forward-step counts are saved in per-record diagnostics.
- The receiver still performs exactly one revision generation.

## Conditions and controls

```text
none
true_text, self_text, other_text
true_statebridge, self_statebridge, other_statebridge
true_latentmas, self_latentmas, other_latentmas
```

`other_*` uses the deterministic cyclic item mapping with offset 137. It is not selected using correctness, answer, similarity, or gold labels.

## Seed construction

Seeds are derived with SHA256 from null-separated fixed fields. Conceptually:

```text
prebelief seed = hash(global_seed, replication_id, item_id, agent_A_or_B_pre)
revision seed  = hash(global_seed, replication_id, item_id, direction, revision)
```

The first eight digest bytes are interpreted as a big-endian integer modulo `2^31 - 1`. All communication conditions for a given replication, item, and direction receive the same revision seed.

## Generation configuration

```text
model = Qwen/Qwen3-4B
model revision = 1cfa9a7208912126459214e8b04321603b3df60c
do_sample = true
temperature = 0.6
top_p = 0.95
top_k = null
max_new_tokens = 8192
global_seed = 42
```
