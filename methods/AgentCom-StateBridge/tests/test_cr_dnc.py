"""Unit tests for CR-DNC (plan section 15, the parts that need no model).

Anchor identity, causal sanity and anchor separability need a forward pass and
are covered by the smoke test instead.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from communication import decision_nullspace as dnc
from icr.parsing_v3 import iter_boxed_spans

REASONING = (
    "The patient has bloody diarrhea, so option C is plausible and option A is not.\n"
    "Considering \\boxed{B} would be wrong here because the CRP is elevated.\n"
    "Therefore the answer is \\boxed{C}"
)


# --- A. Counterfactual integrity -------------------------------------------

def test_variants_differ_only_in_the_final_answer_span():
    built = dnc.build_counterfactuals(REASONING, "C")
    assert set(built.variants) == set(dnc.CHOICES)
    for label, text in built.variants.items():
        assert text.endswith(f"\\boxed{{{label}}}")
        # Everything before the rewritten span is byte-identical.
        assert text[: built.answer_span[0]] == REASONING[: built.answer_span[0]]

    # Pairwise, the variants differ in exactly one character position.
    a, b = built.variants["A"], built.variants["B"]
    assert len(a) == len(b)
    assert sum(1 for x, y in zip(a, b) if x != y) == 1


def test_only_the_final_boxed_span_is_rewritten():
    built = dnc.build_counterfactuals(REASONING, "C")
    # The earlier \boxed{B} inside the reasoning must survive untouched.
    for text in built.variants.values():
        spans = iter_boxed_spans(text)
        assert len(spans) == 2
        assert spans[0][2] == "B"


def test_assert_controlled_rejects_a_tampered_variant():
    built = dnc.build_counterfactuals(REASONING, "C")
    built.variants["A"] = "totally different reasoning \\boxed{A}"
    with pytest.raises(AssertionError):
        built.assert_controlled()


def test_missing_boxed_span_raises_so_the_caller_can_fall_back():
    with pytest.raises(ValueError):
        dnc.build_counterfactuals("no formal answer here", "C")


def test_non_choice_sender_answer_raises():
    with pytest.raises(ValueError):
        dnc.build_counterfactuals(REASONING, "17")


# --- direction estimation ---------------------------------------------------

def _anchors(hidden=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {c: torch.randn(hidden, generator=g) for c in dnc.CHOICES}


def test_direction_is_unit_norm_and_matches_the_definition():
    anchors = _anchors()
    d, diag = dnc.estimate_decision_direction(anchors, "C")
    assert d is not None
    assert d.norm().item() == pytest.approx(1.0, abs=1e-5)
    expected = anchors["C"].float() - torch.stack(
        [anchors[c].float() for c in dnc.CHOICES if c != "C"]
    ).mean(0)
    assert torch.allclose(d, expected / expected.norm(), atol=1e-6)
    assert diag["raw_direction_norm"] > 0
    assert set(diag["anchor_norms"]) == set(dnc.CHOICES)


def test_identical_anchors_fall_back_instead_of_dividing_by_zero():
    shared = torch.ones(16)
    d, diag = dnc.estimate_decision_direction({c: shared.clone() for c in dnc.CHOICES}, "A")
    assert d is None
    assert "fallback_reason" in diag
    assert diag["anchor_states_identical"] is True


def test_unknown_sender_answer_raises():
    with pytest.raises(ValueError):
        dnc.estimate_decision_direction(_anchors(), "E")


# --- E/F/G. Projection ------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_alpha_one_removes_the_direction(dtype):
    g = torch.Generator().manual_seed(1)
    payload = torch.randn(1, 64, 32, generator=g).to(dtype)
    d = torch.randn(32, generator=g)
    d = d / d.norm()
    out, diag = dnc.project_out_direction(payload, d, alpha=1.0)
    # Checked in float32 before the dtype round-trip, as the plan requires.
    assert diag["residual_after_projection"] < 1e-3
    assert out.shape == payload.shape          # F. shape invariance
    assert out.dtype == payload.dtype
    assert diag["finite"] is True              # G. finite values
    assert torch.isfinite(out).all()


def test_alpha_zero_is_the_identity():
    payload = torch.randn(1, 8, 32)
    d = torch.randn(32); d = d / d.norm()
    out, diag = dnc.project_out_direction(payload, d, alpha=0.0)
    assert torch.allclose(out, payload, atol=1e-6)
    assert diag["relative_intervention_norm"] == pytest.approx(0.0, abs=1e-6)


def test_intervention_norm_grows_with_alpha():
    payload = torch.randn(1, 64, 32)
    d = torch.randn(32); d = d / d.norm()
    norms = [
        dnc.project_out_direction(payload, d, alpha=a)[1]["relative_intervention_norm"]
        for a in (0.25, 0.5, 0.75, 1.0)
    ]
    assert norms == sorted(norms)
    assert norms[0] > 0


def test_projection_diagnostics_cover_every_position():
    payload = torch.randn(1, 64, 32)
    d = torch.randn(32); d = d / d.norm()
    _, diag = dnc.project_out_direction(payload, d, alpha=1.0)
    assert len(diag["projection_magnitude_by_position"]) == 64
    assert len(diag["projection_ratio_by_position"]) == 64


def test_an_unnormalised_direction_is_normalised_internally():
    payload = torch.randn(1, 16, 32)
    d = torch.randn(32) * 37.0
    out_scaled, _ = dnc.project_out_direction(payload, d, alpha=1.0)
    out_unit, _ = dnc.project_out_direction(payload, d / d.norm(), alpha=1.0)
    assert torch.allclose(out_scaled, out_unit, atol=1e-5)


# --- H. Ground-truth isolation ---------------------------------------------

def test_module_cannot_reach_a_reference_label():
    """No identifier, attribute or literal in executable code names a label.

    Scanning the raw source would fail on the docstring, which says in prose
    that the method never touches correctness or the receiver. The guarantee
    that matters is about code, so this walks the AST and ignores docstrings
    and comments.
    """
    import ast

    tree = ast.parse(inspect.getsource(dnc))
    forbidden = {"gold", "is_correct", "correct", "correctness", "receiver",
                 "receiver_state", "reference_answer", "label"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.arg):
            seen.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            seen.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Skip docstrings; any other string literal is real code.
            if node.value.strip() and "\n" not in node.value:
                seen.add(node.value)
    leaked = {name for name in seen if name.lower() in forbidden}
    assert not leaked, f"CR-DNC code references {sorted(leaked)}"


def test_public_entry_points_take_no_correctness_argument():
    for fn in (dnc.build_counterfactuals, dnc.estimate_decision_direction,
               dnc.project_out_direction):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"gold", "correct", "is_correct", "receiver", "receiver_state"}
