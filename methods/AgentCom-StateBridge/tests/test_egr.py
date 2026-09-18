from egr.analyze_gate_crossfit import threshold_grid
from egr.analyze_contrast import verdict_audit
from egr.analyze_permutation import permutation_audit
from egr.evidence import build_evidence_packet, summarize_leakage
from egr.prompts import (
    build_contrast_adjudication_prompt,
    build_dual_scoring_prompt,
    build_falsification_prompt,
    build_evidence_ledger_prompt,
    build_ledger_decision_prompt,
    build_hypothesis_adjudication_prompt,
    build_scoring_prompt,
)
from egr.scoring import egr_gate


def _record(reasoning, answer="c"):
    return {
        "replication_id": "seed_pair_00",
        "item_id": 7,
        "agent_id": "A",
        "reasoning_text": reasoning,
        "parsed_answer": answer,
    }


def _metadata():
    return {
        "benchmark": "medqa300",
        "answer_options": {
            "a": "A. Distractor one",
            "b": "B. Distractor two",
            "c": "C. Nystatin",
            "d": "D. Distractor four",
        },
    }


def test_evidence_packet_removes_explicit_claim_but_keeps_medical_evidence():
    packet = build_evidence_packet(
        _record(
            "Nystatin is effective for local candidiasis.\n"
            "Thus, the correct answer is C.\n"
            "\\boxed{C}"
        ),
        _metadata(),
    )
    assert "Nystatin is effective" in packet["evidence_packet"]
    assert "correct answer" not in packet["evidence_packet"]
    assert "boxed" not in packet["evidence_packet"]
    assert packet["sender_answer_text_present_after_filter"] is True
    assert packet["explicit_answer_cue_present_after_filter"] is False


def test_answer_bearing_conclusion_is_removed_without_deleting_option_analysis():
    packet = build_evidence_packet(
        _record(
            "- C. Nystatin: It treats local candidiasis.\n\n"
            "**Conclusion**: Nystatin is the best treatment."
        ),
        _metadata(),
    )
    assert "It treats local candidiasis" in packet["evidence_packet"]
    assert "Conclusion" not in packet["evidence_packet"]
    assert "answer_bearing_conclusion" in packet["removed_answer_cues"]


def test_leakage_summary_is_explicit_about_answer_text_presence():
    packets = [
        build_evidence_packet(_record("Nystatin treats candidiasis."), _metadata()),
        build_evidence_packet(_record("The organism is fungal.", answer="a"), _metadata()),
    ]
    summary = summarize_leakage(packets)
    assert summary["packets"] == 2
    assert summary["answer_text_present"] == 1
    assert summary["explicit_answer_cue_present"] == 0


def test_egr_gate_requires_positive_evidence_margin_and_gain():
    assert egr_gate(0.1, 0.2)
    assert not egr_gate(0.0, 0.2)
    assert not egr_gate(0.1, 0.0)
    assert not egr_gate(-0.1, 0.2)
    assert not egr_gate(0.2, 0.1, tau=0.1)
    assert egr_gate(0.2, 0.10001, tau=0.1)


def test_threshold_grid_uses_positive_gain_quantiles():
    rows = [{"G": value} for value in (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    grid = threshold_grid(rows)
    assert grid[0] == 0.0
    assert grid[-1] == 3.7
    assert len(grid) == 5


def test_scoring_prompt_never_inserts_an_explicit_sender_claim():
    prompt = build_scoring_prompt("Question?", "My prior", "External evidence")
    assert "External evidence" in prompt
    assert "sender answer" not in prompt.lower()
    assert "the answer best supported" in prompt


def test_dual_scoring_prompt_has_no_agent_identity():
    prompt = build_dual_scoring_prompt("Question?", "Observed evidence")
    assert "Observed evidence" in prompt
    assert "sender" not in prompt.lower()
    assert "receiver" not in prompt.lower()


def test_contrast_prompt_anonymizes_evidence_sources():
    prompt = build_contrast_adjudication_prompt("Question?", "first", "second")
    assert "Evidence set 1" in prompt
    assert "Evidence set 2" in prompt
    assert "sender" not in prompt.lower()
    assert "receiver" not in prompt.lower()


def test_gsm8k_contrast_prompt_requests_numeric_answer():
    prompt = build_contrast_adjudication_prompt(
        "2 + 3?", "2 + 3 = 4", "2 + 3 = 5", task="gsm8k"
    )
    assert "math word problem" in prompt
    assert "\\boxed{NUMBER}" in prompt
    assert "A, B, C, or D" not in prompt


def test_code_evidence_preserves_executable_implementation():
    record = _record("```python\ndef f(x):\n    return x + 1\n```", answer="unused")
    metadata = {
        "benchmark": "humanevalplus_test",
        "answer_type": "code",
        "answer_options": {},
    }
    packet = build_evidence_packet(record, metadata)
    assert "def f(x):" in packet["evidence_packet"]
    assert "return x + 1" in packet["evidence_packet"]
    assert packet["removed_spans"] == []


def test_code_contrast_prompt_requests_one_python_block():
    prompt = build_contrast_adjudication_prompt(
        "Implement f.", "def f(): return 1", "def f(): return 2", task="humanevalplus"
    )
    assert "programming problem" in prompt
    assert "markdown Python code block" in prompt
    assert "A, B, C, or D" not in prompt


def test_verdict_audit_separates_selection_and_third_answer_attempt():
    rows = [
        {
            "decision_reason": "contrast_adjudication",
            "answer_A": "a",
            "answer_B": "b",
            "correct_A": True,
            "correct_B": False,
            "verdict": "a",
        },
        {
            "decision_reason": "contrast_adjudication",
            "answer_A": "a",
            "answer_B": "b",
            "correct_A": False,
            "correct_B": False,
            "verdict": "c",
        },
        {
            "decision_reason": "answers_agree",
            "answer_A": "d",
            "answer_B": "d",
            "correct_A": True,
            "correct_B": True,
            "verdict": "d",
        },
    ]
    audit = verdict_audit(rows)
    assert audit["agreements"] == 1
    assert audit["one_correct_selected_correct"] == 1
    assert audit["selected_third"] == 1
    assert audit["both_wrong_selected_third"] == 1


def test_falsification_prompt_marks_preliminary_answer_as_untrusted():
    prompt = build_falsification_prompt("Question?", "first", "second", "Maybe B")
    assert "Maybe B" in prompt
    assert "may contain a confident medical error" in prompt
    assert "Try to falsify it" in prompt


def test_evidence_ledger_separates_fact_work_from_answer_selection():
    ledger = build_evidence_ledger_prompt("Question?", "first", "second")
    decision = build_ledger_decision_prompt("Question?", "verified facts")
    assert "Do not choose" in ledger
    assert "Remaining uncertainty:" in ledger
    assert "raw candidate conclusions were intentionally withheld" in decision
    assert "verified facts" in decision


def test_hypothesis_prompt_binds_evidence_without_granting_authority():
    prompt = build_hypothesis_adjudication_prompt(
        "Question?", "a", "First answer", "first evidence", "c", "Second answer", "second evidence"
    )
    assert "Hypothesis 1 proposes option A: First answer" in prompt
    assert "Hypothesis 2 proposes option C: Second answer" in prompt
    assert "receive no weight merely because they were proposed" in prompt


def test_permutation_audit_counts_caught_and_rejected_verdicts():
    rows = [
        {
            "decision_reason": "permutation_check",
            "order_stable": False,
            "answer_A": "a",
            "answer_B": "b",
            "correct_A": True,
            "correct_B": False,
            "original_verdict": "b",
        },
        {
            "decision_reason": "permutation_check",
            "order_stable": True,
            "answer_A": "c",
            "answer_B": "d",
            "correct_A": False,
            "correct_B": True,
            "original_verdict": "d",
        },
    ]
    audit = permutation_audit(rows)
    assert audit["order_unstable"] == 1
    assert audit["unstable_original_wrong"] == 1
    assert audit["one_correct_stable_correct"] == 1
