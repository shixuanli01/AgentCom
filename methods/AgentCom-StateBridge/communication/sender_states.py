"""Recover the pre-alignment StateBridge payload for a cached sender trajectory.

CR-DNC has to modify the sender hidden states *before* StateBridge aligns them,
but only the aligned prefix is cached on disk. The raw [1, 64, hidden] tensor is
discarded after `_prepare_handoff` runs.

It can be recovered exactly, without regenerating any reasoning, because the
belief record stores `generated_token_ids`. Teacher-forcing that exact token
sequence reproduces the same states the generation loop saw, provided the
off-by-one is right:

During generation a forward hook captures the last layer's last position at each
step. Step 0 is the prefill, whose final position is the last prompt token and
whose logits produced generated token 0; step t produces generated token t. So
the captured `hidden[t]` is the state that *produced* token t, which in a
teacher-forced pass over [prompt, generated] sits at index `prompt_len - 1 + t`.

The recovery is verified end to end rather than assumed: the recovered states are
pushed through the unmodified alignment and compared against the cached prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SenderStates:
    """The payload StateBridge consumes, plus what is needed to audit it."""

    selected_hidden: torch.Tensor      # [1, K, hidden]; K is 64 unless the trajectory is shorter
    selected_token_ids: torch.Tensor   # [1, K]
    selected_indices: list[int]        # positions within the post-think slice
    generated_length: int
    post_think_length: int


@torch.no_grad()
def replay_sender_states(runtime, *, question: str, generated_token_ids: list[int]) -> SenderStates:
    """Reproduce the raw sender payload from a cached trajectory."""
    from methods.state_bridge import select_hidden_states
    from icr.protocol import independent_solver_prompt

    prompt = runtime._render(independent_solver_prompt(runtime.task, question))
    input_ids, attention_mask, _, _ = runtime._encode_prompt(prompt, None)
    prompt_len = int(input_ids.shape[1])

    gen = torch.tensor([generated_token_ids], dtype=torch.long, device=runtime.device)
    full = torch.cat([input_ids, gen], dim=1)
    mask = torch.ones_like(full)

    out = runtime.model.model(
        input_ids=full, attention_mask=mask, output_hidden_states=True, use_cache=False
    )
    states = out.hidden_states[-1]
    # hidden[t] produced token t, so it lives one position to the left.
    hidden = states[:, prompt_len - 1 : prompt_len - 1 + gen.shape[1], :]

    filtered_hidden, filtered_ids = runtime._post_think_slice(hidden, gen)
    if filtered_hidden.shape[1] == 0:
        raise RuntimeError("replayed trajectory has no post-think states")

    selected_hidden, selected_token_ids, selected_indices = select_hidden_states(
        filtered_hidden,
        filtered_ids,
        k=runtime.bridge.max_prefix_tokens,
        method=runtime.bridge.selection_method,
        window_size=runtime.bridge.turning_point_window_size,
    )
    return SenderStates(
        selected_hidden=selected_hidden,
        selected_token_ids=selected_token_ids,
        selected_indices=list(selected_indices),
        generated_length=int(gen.shape[1]),
        post_think_length=int(filtered_hidden.shape[1]),
    )


@torch.no_grad()
def align_to_prefix(runtime, states: SenderStates) -> torch.Tensor:
    """Push recovered states through the UNMODIFIED StateBridge alignment."""
    aligned = runtime.bridge._align_hidden_sequence(
        states.selected_hidden, states.selected_token_ids
    )
    return runtime.bridge._process_prefix(aligned)


def verify_against_cache(
    runtime, *, question: str, record: dict[str, Any], cached_prefix: torch.Tensor
) -> dict[str, Any]:
    """Check that the recovered payload reproduces the cached aligned prefix."""
    states = replay_sender_states(
        runtime, question=question, generated_token_ids=record["generated_token_ids"]
    )
    prefix = align_to_prefix(runtime, states).detach().cpu().float()
    cached = cached_prefix.detach().cpu().float()
    if prefix.shape != cached.shape:
        return {
            "ok": False, "reason": f"shape {tuple(prefix.shape)} != {tuple(cached.shape)}",
            "selected_indices_match": None,
        }
    diff = (prefix - cached).abs()
    scale = cached.abs().max().clamp_min(1e-6)
    recorded = record.get("statebridge", {}).get("selected_indices")
    return {
        "ok": bool((diff.max() / scale) < 1e-2),
        "max_abs_diff": float(diff.max()),
        "relative_max_diff": float(diff.max() / scale),
        "cached_abs_max": float(cached.abs().max()),
        "selected_indices_match": (recorded == states.selected_indices) if recorded else None,
        "K": int(states.selected_hidden.shape[1]),
        "post_think_length": states.post_think_length,
        "generated_length": states.generated_length,
    }
