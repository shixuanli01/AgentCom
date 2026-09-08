from agentcom.llm_judge import (
    aggregate_judge_items,
    build_candidate_packet,
    deterministic_candidate_order,
    parse_judge_choice,
)


def _source_item(predictions, gold="c"):
    return {
        "item_index": 7,
        "gold": gold,
        "branch_predictions": predictions,
    }


def _branch(branch_id, prediction, rationale=None):
    return {
        "branch_id": branch_id,
        "prediction": prediction,
        "final_response": rationale or f"reasoning for {prediction}",
    }


def test_candidate_order_is_reproducible_and_count_independent():
    first = deterministic_candidate_order(["a", "c"], base_seed=42, item_index=3)
    repeated = deterministic_candidate_order(
        ["c", "a", "a", "a"], base_seed=42, item_index=3
    )
    assert first == repeated
    assert set(first) == {"a", "c"}


def test_candidate_packet_contains_one_rationale_per_valid_answer():
    item = _source_item(["c", "a", "a", "a", "c"])
    branches = [
        _branch(0, "c"),
        _branch(1, "a"),
        _branch(2, "a"),
        _branch(3, "not-an-option"),
        _branch(4, "c"),
    ]
    packet = build_candidate_packet(item, branches, base_seed=42)
    assert {candidate["label"] for candidate in packet} == {"a", "c"}
    assert len(packet) == 2
    assert all("vote" not in candidate for candidate in packet)


def test_parse_judge_choice_is_strict_and_candidate_constrained():
    assert parse_judge_choice("Analysis. \\boxed{C}", ["a", "c"]) == "c"
    assert parse_judge_choice("Analysis. \\boxed{B}", ["a", "c"]) is None
    assert parse_judge_choice("I choose C", ["a", "c"]) is None


def test_aggregate_judge_items_reports_paired_effects():
    rows = [
        {
            "judge_correct": True,
            "vote_correct": False,
            "branch_0_correct": False,
            "decision_mode": "llm_judge",
            "judge_prediction": "a",
            "oracle_at_m": True,
            "judge_seconds": 2.0,
            "judge_generated_tokens": 20,
        },
        {
            "judge_correct": False,
            "vote_correct": True,
            "branch_0_correct": True,
            "decision_mode": "llm_judge",
            "judge_prediction": "b",
            "oracle_at_m": True,
            "judge_seconds": 4.0,
            "judge_generated_tokens": 40,
        },
        {
            "judge_correct": True,
            "vote_correct": True,
            "branch_0_correct": True,
            "decision_mode": "unanimous_passthrough",
            "judge_prediction": "c",
            "oracle_at_m": True,
            "judge_seconds": 0.0,
            "judge_generated_tokens": 0,
        },
    ]
    metrics = aggregate_judge_items(rows, base_seed=42)
    assert metrics["llm_judge"]["correct"] == 2
    assert metrics["judge_vs_vote"]["corrections"] == 1
    assert metrics["judge_vs_vote"]["harms"] == 1
    assert metrics["judge_calls"] == 2
    assert metrics["oracle_miss_recovery"]["recovered"] == 1
    assert metrics["mean_judge_seconds"] == 3.0
