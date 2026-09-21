"""Counterfactual Readout Decision-Nullspace Communication (CR-DNC).

The hypothesis: a sender's latent payload carries a decision-commitment
component that can be isolated by varying only the sender's final answer, and
suppressing it before communication should reduce harmful answer adoption while
leaving useful reasoning transfer intact.

Estimation, per sender, ground-truth-free and sender-only:

1. Take the sender's own reasoning and rewrite ONLY its final ``\\boxed{...}``
   span into each of the four options. Every other character is identical, and
   that is asserted rather than assumed.
2. Teacher-force all four through the same frozen model and read the hidden
   state at a shared anchor token that sits AFTER the answer. The anchor has to
   come after the answer because the model is causal: comparing positions before
   the answer cannot see it, and such a comparison would degenerate into answer
   masking.
3. With the sender's own answer a_s, set u_pos = u_{a_s} and u_neg = mean of the
   other three; d = normalize(u_pos - u_neg).
4. Remove d from every position of the payload: H_tilde = H - alpha (H d) d^T.

The anchor states and the payload must live in the same space or the projection
is meaningless. The payload StateBridge consumes is the output of the last
decoder layer, captured by a forward hook -- NOT ``hidden_states[-1]``, which is
that tensor after the final norm. The anchor is read through the same hook.

Nothing here sees a reference answer, a correctness flag, the receiver, or the
receiver's states. The only answer it reads is the sender's own parsed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from icr.parsing_v3 import iter_boxed_spans

CHOICES = ("A", "B", "C", "D")
DIRECTION_EPSILON = 1e-8


@dataclass
class CounterfactualSet:
    """Four sender responses differing only in the final formal answer."""

    variants: dict[str, str]
    answer_span: tuple[int, int]
    original_answer: str
    prefix_text: str
    suffix_text: str

    def assert_controlled(self) -> None:
        """Everything outside the rewritten span must be byte-identical."""
        for choice, text in self.variants.items():
            if not text.startswith(self.prefix_text):
                raise AssertionError(f"variant {choice} altered text before the answer")
            if not text.endswith(self.suffix_text):
                raise AssertionError(f"variant {choice} altered text after the answer")
            head = len(self.prefix_text)
            tail = len(text) - len(self.suffix_text)
            if text[head:tail] not in {f"\\boxed{{{c}}}" for c in CHOICES}:
                raise AssertionError(f"variant {choice} has an unexpected answer span")


def build_counterfactuals(reasoning: str, sender_answer: str) -> CounterfactualSet:
    """Rewrite only the final ``\\boxed{...}`` into each of the four options.

    Earlier mentions of A-D in the reasoning are deliberately left alone. The
    intervention is meant to vary the formal decision while holding the argument
    that led to it fixed.
    """
    choice = str(sender_answer).strip().upper()
    if choice not in CHOICES:
        raise ValueError(f"sender answer {sender_answer!r} is not one of {CHOICES}")
    spans = iter_boxed_spans(reasoning)
    if not spans:
        raise ValueError("no \\boxed{...} span in the sender reasoning")
    start, end, _ = spans[-1]
    prefix, suffix = reasoning[:start], reasoning[end:]
    built = CounterfactualSet(
        variants={c: f"{prefix}\\boxed{{{c}}}{suffix}" for c in CHOICES},
        answer_span=(start, end),
        original_answer=choice,
        prefix_text=prefix,
        suffix_text=suffix,
    )
    built.assert_controlled()
    return built


def estimate_decision_direction(
    anchor_states: dict[str, torch.Tensor], sender_answer: str
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    """d = normalize(u_{a_s} - mean(u_a for a != a_s)), in float32.

    Returns ``(None, diagnostics)`` when the contrast is numerically empty, so
    the caller can fall back to the raw payload rather than divide by ~0.
    """
    choice = str(sender_answer).strip().upper()
    if choice not in anchor_states:
        raise ValueError(f"no anchor state for sender answer {choice!r}")
    stacked = {k: v.float().reshape(-1) for k, v in anchor_states.items()}
    positive = stacked[choice]
    others = [v for k, v in stacked.items() if k != choice]
    negative = torch.stack(others).mean(dim=0)
    raw = positive - negative
    norm = float(raw.norm())

    keys = sorted(stacked)
    cosines = {
        f"{a}|{b}": float(
            torch.nn.functional.cosine_similarity(stacked[a], stacked[b], dim=0)
        )
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }
    diagnostics = {
        "anchor_norms": {k: float(v.norm()) for k, v in stacked.items()},
        "anchor_pairwise_cosine": cosines,
        "raw_direction_norm": norm,
        "anchor_states_identical": max(cosines.values()) > 1 - 1e-6
        and min(cosines.values()) > 1 - 1e-6,
    }
    if norm < DIRECTION_EPSILON:
        diagnostics["fallback_reason"] = f"direction norm {norm:.3e} below epsilon"
        return None, diagnostics
    return raw / norm, diagnostics


def project_out_direction(
    payload: torch.Tensor, direction: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    """H_tilde = H - alpha (H d) d^T, computed in float32.

    Shape and dtype are preserved: no state is zeroed, no position is dropped,
    and StateBridge downstream is untouched.
    """
    original_dtype = payload.dtype
    flat = payload.reshape(-1, payload.shape[-1]).float()
    unit = direction.float().reshape(-1)
    unit = unit / unit.norm().clamp_min(DIRECTION_EPSILON)

    coefficients = flat @ unit
    suppressed = flat - alpha * coefficients[:, None] * unit[None, :]

    residual = float((suppressed @ unit).abs().max())
    norms = flat.norm(dim=1)
    diagnostics = {
        "alpha": float(alpha),
        "raw_payload_frobenius": float(flat.norm()),
        "suppressed_payload_frobenius": float(suppressed.norm()),
        "relative_intervention_norm": float(
            (suppressed - flat).norm() / (flat.norm() + DIRECTION_EPSILON)
        ),
        "projection_magnitude_by_position": [float(x) for x in coefficients.abs()],
        "projection_ratio_by_position": [
            float(x) for x in (coefficients.abs() / norms.clamp_min(DIRECTION_EPSILON))
        ],
        "total_projection_energy": float((coefficients**2).sum()),
        "residual_after_projection": residual,
        "finite": bool(torch.isfinite(suppressed).all()),
    }
    return suppressed.reshape(payload.shape).to(original_dtype), diagnostics
