"""Model-coupled half of CR-DNC: counterfactual encoding and payload building.

`decision_nullspace` stays free of the model so its tests run in a second. This
module owns everything that needs a forward pass.

The counterfactual is applied at the token level, not the text level. On MedQA
each option is a single token and the trajectory ends with the assistant-end
token, so the four variants are the cached token ids with exactly one id
substituted. That is stronger than rewriting text and re-tokenising: every other
token is bit-identical by construction, the four sequences have the same length
so batching needs no padding, and the anchor sits at the same index in all four.

The anchor states are read through a forward hook on the last decoder layer,
the same tensor StateBridge's payload comes from. `hidden_states[-1]` is that
tensor after the final norm; estimating a direction there and projecting it out
of the pre-norm payload would mix two spaces.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from communication.decision_nullspace import (
    CHOICES,
    estimate_decision_direction,
    project_out_direction,
)

BOXED_TOKEN = "boxed"
OPEN_BRACE = "{"


def locate_answer_token(token_ids: list[int], tokenizer) -> Optional[int]:
    """Index of the single token holding the final formal answer.

    Scans backwards for the last ``boxed`` immediately followed by ``{``; the
    next token is the answer. Returns None when the pattern is absent or the
    answer is not one of the four options, so the caller falls back to the raw
    payload instead of dropping the sender.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index] != BOXED_TOKEN:
            continue
        if index + 2 >= len(tokens) or tokens[index + 1] != OPEN_BRACE:
            continue
        candidate = tokens[index + 2]
        if candidate.strip().upper() in CHOICES:
            return index + 2
    return None


def build_token_counterfactuals(
    token_ids: list[int], answer_position: int, tokenizer
) -> dict[str, list[int]]:
    """The same trajectory with the answer token replaced by each option."""
    variants: dict[str, list[int]] = {}
    for choice in CHOICES:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"option {choice!r} is not a single token")
        ids = list(token_ids)
        ids[answer_position] = encoded[0]
        variants[choice] = ids
    lengths = {len(v) for v in variants.values()}
    if lengths != {len(token_ids)}:
        raise AssertionError("counterfactual variants changed the sequence length")
    for choice, ids in variants.items():
        differing = [i for i, (a, b) in enumerate(zip(ids, token_ids)) if a != b]
        if differing not in ([], [answer_position]):
            raise AssertionError(
                f"variant {choice} differs at {differing}, not only at the answer"
            )
    return variants


@torch.no_grad()
def encode_anchor_states(
    runtime, *, prompt_ids: torch.Tensor, variants: dict[str, list[int]],
    anchor_offset_from_end: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Last-layer state at the shared anchor token, one row per option.

    All four variants are encoded in a single batched call. That is four
    sequence-equivalent encodings, not "one forward" in the sense of one
    sequence, and the cost accounting says so.
    """
    order = list(CHOICES)
    batch = torch.tensor([variants[c] for c in order], dtype=torch.long,
                         device=runtime.device)
    prompts = prompt_ids.repeat(len(order), 1)
    full = torch.cat([prompts, batch], dim=1)
    anchor_index = full.shape[1] - 1 - anchor_offset_from_end

    anchor_ids = full[:, anchor_index].tolist()
    if len(set(anchor_ids)) != 1:
        raise AssertionError(f"anchor token ids differ across variants: {anchor_ids}")

    grabbed: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = runtime.model.model.model.layers[-1].register_forward_hook(hook)
    try:
        runtime.model.model(
            input_ids=full, attention_mask=torch.ones_like(full), use_cache=False
        )
    finally:
        handle.remove()

    states = grabbed["h"][:, anchor_index, :]
    return (
        {choice: states[i].detach().float().cpu() for i, choice in enumerate(order)},
        {
            "anchor_index": int(anchor_index),
            "anchor_token_id": int(anchor_ids[0]),
            "sequence_length": int(full.shape[1]),
            "batched_sequences": len(order),
        },
    )


@torch.no_grad()
def build_payload(
    runtime, *, raw_payload: torch.Tensor, token_ids: list[int],
    prompt_ids: torch.Tensor, sender_answer: str, alpha: float,
    direction_override: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """CR-DNC payload for one sender, falling back to the raw payload on failure.

    `direction_override` exists for the controls: a random direction, or one
    estimated on a different question. It bypasses estimation only.
    """
    diagnostics: dict[str, Any] = {
        "sender_answer": sender_answer, "alpha": float(alpha),
        "counterfactual_status": "ok", "fallback_reason": None,
    }
    tokenizer = runtime.model.tokenizer

    if direction_override is None:
        position = locate_answer_token(token_ids, tokenizer)
        if position is None:
            diagnostics.update(counterfactual_status="fallback",
                               fallback_reason="no single-token boxed answer found")
            return raw_payload, diagnostics
        try:
            variants = build_token_counterfactuals(token_ids, position, tokenizer)
        except (ValueError, AssertionError) as error:
            diagnostics.update(counterfactual_status="fallback",
                               fallback_reason=f"counterfactual build failed: {error}")
            return raw_payload, diagnostics
        diagnostics["answer_token_position"] = position
        diagnostics["answer_offset_from_end"] = len(token_ids) - 1 - position

        anchors, anchor_info = encode_anchor_states(
            runtime, prompt_ids=prompt_ids, variants=variants
        )
        diagnostics.update(anchor_info)
        direction, direction_info = estimate_decision_direction(anchors, sender_answer)
        diagnostics.update(direction_info)
        if direction is None:
            diagnostics.update(counterfactual_status="fallback")
            return raw_payload, diagnostics
    else:
        direction = direction_override
        diagnostics["counterfactual_status"] = "override"

    suppressed, projection_info = project_out_direction(
        raw_payload.float(), direction, alpha
    )
    diagnostics.update(projection_info)
    return suppressed.to(raw_payload.dtype), diagnostics
