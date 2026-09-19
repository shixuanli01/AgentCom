"""ICR-V3 prompt alignment, per-task output contracts, and scoring fixes."""

import pytest

from icr.parsing_v3 import (
    extract_numeric_gold,
    numeric_equal,
    parse_choice_answer,
    parse_numeric_answer,
)
from icr.prompts_v3 import (
    ANSWER_FORMAT,
    PROMPT_VERSION,
    external_block,
    independent_solver_prompt,
    revision_prompt,
)
from prompts import EMBEDDING_HINT_MARKER


PRIOR = {
    "question": "<<QUESTION>>",
    "receiver_prior_reasoning": "<<PRIOR>>",
    "receiver_prior_answer": "a",
}


def _revision(task, condition, sender_reasoning=None):
    return revision_prompt(
        task,
        external_block_text=external_block(
            condition, sender_reasoning=sender_reasoning
        ),
        **PRIOR,
    )


# --- Scoring fixes S1-S5 -------------------------------------------------


def test_s1_numeric_gold_keeps_thousands_separator_digits():
    # Legacy utils.extract_gold truncated "#### 2,125" to "2".
    assert extract_numeric_gold("Work.\n#### 2,125") == "2125"
    assert extract_numeric_gold("#### 18") == "18"
    assert extract_numeric_gold("no gold here") is None


def test_s2_boxed_number_survives_thousands_separator():
    # Legacy parsing returned "2" for the comma form, inverting correctness.
    assert parse_numeric_answer(r"So the answer is \boxed{2,125}.") == "2125"
    assert parse_numeric_answer(r"So the answer is \boxed{2125}.") == "2125"
    assert numeric_equal(
        parse_numeric_answer(r"\boxed{2125}"), extract_numeric_gold("#### 2,125")
    )
    assert numeric_equal(
        parse_numeric_answer(r"\boxed{2,125}"), extract_numeric_gold("#### 2,125")
    )


def test_s3_boxed_label_with_trailing_option_text():
    assert parse_choice_answer(r"\boxed{C. Colorectal cancer}") == "c"
    assert parse_choice_answer(r"\boxed{(B)}") == "b"
    assert parse_choice_answer(r"\boxed{d}") == "d"


def test_s4_boxed_label_inside_latex_wrapper():
    assert parse_choice_answer(r"\boxed{\text{C}}") == "c"
    assert parse_numeric_answer(r"\boxed{\text{42}}") == "42"


def test_s5_numeric_equality_ignores_pure_formatting():
    assert numeric_equal("18.0", "18")
    assert numeric_equal("+18", "18")
    assert not numeric_equal("18", "19")
    assert not numeric_equal(None, "18")


def test_choice_parser_rejects_out_of_contract_answers():
    assert parse_choice_answer(r"\boxed{E}") is None
    assert parse_choice_answer("The answer is clearly C.") is None
    assert parse_choice_answer(r"\boxed{}") is None


def test_numeric_parser_falls_back_to_last_number_without_boxed():
    assert parse_numeric_answer("first 3 then 7") == "7"


# --- Condition alignment -------------------------------------------------


def test_statebridge_and_latentmas_prompts_are_byte_identical():
    assert _revision("medqa", "true_statebridge") == _revision(
        "medqa", "true_latentmas"
    )


def test_text_differs_from_latent_only_in_the_message_body():
    latent = _revision("medqa", "true_statebridge")
    text = _revision("medqa", "true_text", sender_reasoning="<<SENDER>>")
    assert text == latent.replace(EMBEDDING_HINT_MARKER, "<<SENDER>>")


def test_none_differs_from_latent_only_in_the_external_block():
    latent = _revision("medqa", "true_statebridge")
    none = _revision("medqa", "none")
    assert none == latent.replace(
        external_block("true_statebridge"), external_block("none")
    )


def test_marker_appears_exactly_once_and_only_for_latent_conditions():
    assert _revision("medqa", "true_statebridge").count(EMBEDDING_HINT_MARKER) == 1
    assert EMBEDDING_HINT_MARKER not in _revision("medqa", "none")
    assert EMBEDDING_HINT_MARKER not in _revision(
        "medqa", "true_text", sender_reasoning="<<SENDER>>"
    )


def test_message_slot_sits_between_prior_and_integration_rules():
    prompt = _revision("medqa", "true_statebridge")
    assert (
        prompt.index("Your previous answer:")
        < prompt.index(EMBEDDING_HINT_MARKER)
        < prompt.index("Your task is to REVISE")
    )


def test_self_and_other_controls_share_the_true_condition_rendering():
    for suffix in ("statebridge", "latentmas"):
        assert external_block(f"self_{suffix}") == external_block(f"true_{suffix}")
        assert external_block(f"other_{suffix}") == external_block(f"true_{suffix}")


def test_external_block_rejects_unknown_conditions_and_missing_text():
    with pytest.raises(ValueError):
        external_block("true_telepathy")
    with pytest.raises(ValueError):
        external_block("true_text")


# --- Per-task output contracts -------------------------------------------


@pytest.mark.parametrize(
    "task", ["medqa", "gpqa", "arc_challenge", "gsm8k", "humanevalplus"]
)
def test_output_contract_closes_both_phases(task):
    contract = ANSWER_FORMAT[task]
    assert independent_solver_prompt(task, "Q?").endswith(contract)
    assert _revision(task, "none").endswith(contract)


def test_answer_contracts_match_the_labels_each_dataset_displays():
    assert "one of A, B, C, or D" in ANSWER_FORMAT["medqa"]
    assert "a, b, c, or d" in ANSWER_FORMAT["arc_challenge"]
    assert "final labeled list" in ANSWER_FORMAT["gpqa"]
    assert "no thousands separators" in ANSWER_FORMAT["gsm8k"]
    assert "markdown Python code block" in ANSWER_FORMAT["humanevalplus"]


def test_code_tasks_use_the_implementation_revision_template():
    prompt = _revision("humanevalplus", "true_text", sender_reasoning="<<SENDER>>")
    assert "Your previous reasoning and implementation:" in prompt
    assert "REVISE your implementation" in prompt


def test_prompt_version_is_recorded():
    assert PROMPT_VERSION == "icr_v3_mid_injection"


# --- Pre-registered structural exclusion ---------------------------------


def test_arc_structural_exclusion_counts_trailing_options():
    from icr.benchmarks import structural_exclusions, trailing_option_labels

    four = "Stem?\na: one\nb: two\nc: three\nd: four"
    five = "Stem?\na: one\nb: two\nc: three\nd: four\ne: five"
    three = "Stem?\na: one\nb: two\nc: three"
    assert trailing_option_labels(four) == ["a", "b", "c", "d"]
    assert trailing_option_labels(five) == ["a", "b", "c", "d", "e"]

    data = [{"question": four}, {"question": five}, {"question": three}]
    assert structural_exclusions("arc_challenge", data) == [1, 2]
    assert structural_exclusions("gsm8k", data) == []
