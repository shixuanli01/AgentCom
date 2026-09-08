from agentcom.refiner_causal import (
    aggregate_items,
    paired_metrics,
    parse_boxed_choice,
    select_source_items,
)


def test_parse_boxed_choice_is_strict():
    assert parse_boxed_choice("reasoning \\boxed{C}") == "c"
    assert parse_boxed_choice("answer C") is None


def test_select_source_items_filters_oracle_misses():
    source = {
        "items": [
            {"item_index": 0, "oracle_at_m": True, "vote_at_m_correct": False},
            {"item_index": 1, "oracle_at_m": True, "vote_at_m_correct": True},
            {"item_index": 2, "oracle_at_m": False, "vote_at_m_correct": False},
        ]
    }
    assert [
        item["item_index"]
        for item in select_source_items(source, "oracle-misses")
    ] == [0]
    assert len(select_source_items(source, "all")) == 3


def test_paired_metrics_counts_corrections_and_harms():
    result = paired_metrics(
        [True, False, True], [False, True, True], seed=42
    )
    assert result["corrections"] == 1
    assert result["harms"] == 1
    assert result["delta_accuracy"] == 0.0


def test_aggregate_reports_prefix_effect_and_path_retention():
    item = {
        "fixed_branch_correct": [True, False, True, False, False],
        "fixed_branch_predictions": ["a", "b", "a", "b", "b"],
        "fixed_oracle_at_m": True,
        "fixed_vote_correct": False,
        "no_prefix_correct": False,
        "zero_prefix_correct": True,
        "original_vote_correct": False,
        "original_branch_correct": [True, False, False, False, False],
        "fixed_unique_prediction_count": 2,
        "fixed_pairwise_disagreement": 0.6,
        "any_branch_differs_from_no_prefix": True,
        "any_branch_differs_from_zero_prefix": True,
        "no_prefix_prediction": "b",
        "zero_prefix_prediction": "a",
    }
    result = aggregate_items([item], base_seed=42)
    assert result["fixed_oracle_at_m"]["correct"] == 1
    assert result["items_with_prefix_dependent_branch_answers"] == 1
    assert result["original_correct_path_retention"]["retained"] == 1
    assert result["original_wrong_path_repair"]["repaired"] == 1
