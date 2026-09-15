from icr.analysis_v2 import condition_metrics, safe_delta


def _row(category, pre, post, sender_answer="a", receiver_answer="b", post_answer="a"):
    return {
        "pair_classification": category,
        "sender_correct": category in {"correction_opportunity", "both_correct"},
        "receiver_pre_correct": pre,
        "receiver_post_correct": post,
        "sender_pre_answer": sender_answer,
        "receiver_pre_answer": receiver_answer,
        "receiver_post_answer": post_answer,
        "answer_changed": post_answer != receiver_answer,
        "generation_length": 10,
        "prompt_tokens": 20,
        "generation_seconds": 1.0,
        "communication_payload": {"payload_bytes": 0},
    }


def test_selectivity_and_sra_are_computed_independently():
    rows = [
        _row("correction_opportunity", False, True),
        _row("correction_opportunity", False, False, post_answer="b"),
        _row("destruction_risk", True, True, post_answer="b"),
        _row("both_correct", True, True, sender_answer="a", receiver_answer="a", post_answer="a"),
    ]
    metrics = condition_metrics(rows)
    assert metrics["cr"] == 0.5
    assert metrics["pr"] == 1.0
    assert metrics["si"] == 0.75
    assert metrics["sra"] == 2 / 3


def test_follow_selectivity_distinguishes_correct_and_wrong_sender_following():
    rows = [
        _row("correction_opportunity", False, True, post_answer="a"),
        _row("destruction_risk", True, False, post_answer="a"),
        _row("destruction_risk", True, True, post_answer="b"),
    ]
    metrics = condition_metrics(rows)
    assert metrics["fcs"] == 1.0
    assert metrics["fws"] == 0.5
    assert metrics["follow_selectivity"] == 0.5


def test_sparse_smoke_deltas_remain_nullable():
    assert safe_delta(None, 0.5) is None
    assert safe_delta(0.5, None) is None
    assert safe_delta(0.75, 0.5) == 0.25
