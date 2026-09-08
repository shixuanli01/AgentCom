from agentcom.multipath import (
    INVALID_LABEL,
    aggregate_items,
    branch_seeds,
    canonical_prediction,
    exact_mcnemar_p,
    pairwise_disagreement,
    plurality_vote,
    stable_seed,
    summarize_item,
)


def _branch(prediction, gold="a"):
    return {"prediction": prediction, "correct": prediction == gold}


def test_stable_seed_is_reproducible_and_stage_specific():
    first = stable_seed(42, "medqa", 7, 2, "planner")
    assert first == stable_seed(42, "medqa", 7, 2, "planner")
    assert first != stable_seed(42, "medqa", 7, 3, "planner")
    assert first != stable_seed(42, "medqa", 7, 2, "critic")


def test_all_branch_stage_seeds_are_unique():
    values = {
        seed
        for branch in range(5)
        for seed in branch_seeds(42, "medqa", 0, branch).values()
    }
    assert len(values) == 20


def test_prediction_canonicalization_preserves_invalid_as_category():
    assert canonical_prediction(None) == INVALID_LABEL
    assert canonical_prediction("  ") == INVALID_LABEL
    assert canonical_prediction(" A ") == "a"


def test_plurality_vote_handles_majority_and_tie_by_earliest_branch():
    majority = plurality_vote(["a", "b", "a", "c", "a"])
    assert majority["winner"] == "a"
    assert not majority["tied"]

    tied = plurality_vote(["b", "a", "a", "b", "c"])
    assert tied["winner"] == "b"
    assert tied["tied"]
    assert tied["tied_labels"] == ["a", "b"]


def test_pairwise_disagreement_extremes():
    assert pairwise_disagreement(["a"] * 5) == 0.0
    assert pairwise_disagreement(["a", "b", "c", "d", None]) == 1.0


def test_item_summary_records_correction():
    item = summarize_item(
        3,
        "a",
        [_branch("b"), _branch("a"), _branch("a"), _branch("a"), _branch("c")],
    )
    assert not item["branch_0_correct"]
    assert item["vote_at_m"]["winner"] == "a"
    assert item["vote_at_m_correct"]
    assert item["oracle_at_m"]
    assert item["correction"]
    assert not item["harm"]


def test_aggregate_items_reports_paired_effects():
    correction = summarize_item(
        0,
        "a",
        [_branch("b"), _branch("a"), _branch("a"), _branch("a"), _branch("b")],
    )
    harm = summarize_item(
        1,
        "a",
        [_branch("a"), _branch("b"), _branch("b"), _branch("b"), _branch("a")],
    )
    stable = summarize_item(2, "a", [_branch("a")] * 5)
    aggregate = aggregate_items([correction, harm, stable], base_seed=42)

    assert aggregate["total"] == 3
    assert aggregate["corrections"] == 1
    assert aggregate["harms"] == 1
    assert aggregate["branch_0"]["correct"] == 2
    assert aggregate["vote_at_m"]["correct"] == 2
    assert aggregate["delta_vote_minus_branch_0"] == 0.0
    assert aggregate["mcnemar_exact_two_sided_p"] == 1.0


def test_exact_mcnemar_handles_no_discordant_pairs():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 0) == 0.0625
