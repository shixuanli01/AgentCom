import json

import numpy as np
import pytest
import torch

from icr.analysis import clustered_bootstrap, condition_metrics
from icr.protocol import (
    INDEPENDENT_SOLVER_PROMPT,
    REVISION_BASE,
    classify_pair,
    other_item_id,
    parse_conditions,
    prebelief_seed,
    revision_seed,
)
from icr.channels import CommunicationMessage
from prompts import EMBEDDING_HINT_MARKER


def test_prompt_templates_render_literal_boxed_answer_example():
    solver = INDEPENDENT_SOLVER_PROMPT.format(question="Question?")
    revision = REVISION_BASE.format(
        question="Question?",
        receiver_prior_reasoning="Reasoning",
        receiver_prior_answer="a",
        external_section="No external message is available.",
    )
    assert "\\boxed{A}" in solver
    assert "\\boxed{A}" in revision


def test_statebridge_marker_sits_in_the_shared_message_slot():
    # ICR-V3 places the message slot after the receiver's prior and before the
    # integration rules, identically for every condition.
    # Avoid loading a model: build_revision_prompt only needs the pure render
    # method, which is replaced here by an identity function.
    from icr.runtime import ICRRuntime

    runtime = object.__new__(ICRRuntime)
    runtime._render = lambda content: content
    prompt = runtime.build_revision_prompt(
        question="QUESTION_SENTINEL",
        receiver_reasoning="prior reasoning",
        receiver_answer="a",
        message=CommunicationMessage(
            condition="true_statebridge",
            source_item_id=0,
            source_agent_id="A",
            prefix=torch.zeros(1, 64, 8),
        ),
    )
    assert prompt.count(EMBEDDING_HINT_MARKER) == 1
    assert (
        prompt.index("QUESTION_SENTINEL")
        < prompt.index("Your previous answer:")
        < prompt.index(EMBEDDING_HINT_MARKER)
        < prompt.index("Your task is to REVISE")
    )


def _row(item, direction, condition, sender, pre, post, sender_answer="a", pre_answer="b"):
    return {
        "item_id": item,
        "direction": direction,
        "condition": condition,
        "sender_correct": sender,
        "receiver_pre_correct": pre,
        "receiver_post_correct": post,
        "sender_pre_answer": sender_answer,
        "receiver_pre_answer": pre_answer,
        "receiver_post_answer": sender_answer if post else pre_answer,
        "answer_changed": post != pre,
        "pair_classification": classify_pair(sender, pre),
        "communication_payload": {"payload_bytes": 0},
        "generation_seconds": 1.0,
    }


def test_seeds_are_independent_then_paired_across_conditions():
    assert prebelief_seed(42, 7, "A") != prebelief_seed(42, 7, "B")
    seed = revision_seed(42, 7, "A_to_B")
    assert len({condition: seed for condition in parse_conditions("none,true_text")}.values()) == 2
    assert len(set({condition: seed for condition in parse_conditions("none,true_text")}.values())) == 1


def test_replication_ids_produce_distinct_stable_seed_pairs():
    legacy = prebelief_seed(42, 7, "A")
    seed_01 = prebelief_seed(42, 7, "A", "seed_pair_01")
    assert seed_01 == prebelief_seed(42, 7, "A", "seed_pair_01")
    assert len({legacy, seed_01, prebelief_seed(42, 7, "A", "seed_pair_02")}) == 3
    assert revision_seed(42, 7, "A_to_B", "seed_pair_01") != revision_seed(
        42, 7, "A_to_B", "seed_pair_02"
    )


def test_other_mapping_is_deterministic_and_never_self():
    ids = list(range(10))
    assert other_item_id(0, ids) == 7
    assert all(other_item_id(item, ids) != item for item in ids)


@pytest.mark.parametrize("value", ["", "none,unknown", "none,none"])
def test_condition_parser_rejects_invalid_sets(value):
    with pytest.raises(ValueError):
        parse_conditions(value)


def test_conditional_metrics_measure_correction_and_preservation():
    rows = [
        _row(0, "A_to_B", "true_text", True, False, True),
        _row(0, "B_to_A", "true_text", False, True, True),
        _row(1, "A_to_B", "true_text", False, False, False),
        _row(1, "B_to_A", "true_text", True, True, True),
    ]
    metrics = condition_metrics(rows)
    assert metrics["post_accuracy"] == 0.75
    assert metrics["cr"] == 1.0
    assert metrics["pr"] == 1.0
    assert metrics["sr"] == 0.0
    assert metrics["scr"] == 1.0
    assert metrics["rescues"] == 1
    assert metrics["destructions"] == 0


def test_bootstrap_clusters_both_directions_by_item():
    rows = [
        _row(item, direction, "none", True, False, item == 0)
        for item in range(2)
        for direction in ("A_to_B", "B_to_A")
    ]
    result = clustered_bootstrap({"none": rows}, num_bootstrap=100, seed=1)
    assert result["cluster"] == "item"
    assert result["conditions"]["none"]["post_accuracy"]["ci95"] is not None
