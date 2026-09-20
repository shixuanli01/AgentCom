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


def test_evidence_uses_the_same_message_slot_as_full_text():
    latent = _revision("medqa", "true_statebridge")
    evidence = _revision("medqa", "true_evidence", sender_reasoning="<<EVIDENCE>>")
    assert evidence == latent.replace(EMBEDDING_HINT_MARKER, "<<EVIDENCE>>")


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
    assert EMBEDDING_HINT_MARKER not in _revision(
        "medqa", "true_evidence", sender_reasoning="<<EVIDENCE>>"
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
    with pytest.raises(ValueError):
        external_block("true_evidence")


def test_evidence_channel_removes_explicit_claim_without_rewriting_support():
    from icr.channels import EvidenceCommunicationChannel

    sender = {
        "item_id": 0,
        "agent_id": "A",
        "replication_id": "seed_pair_00",
        "reasoning_text": (
            "Nystatin treats local candidiasis.\n"
            "Therefore, the correct answer is C.\n"
            "\\boxed{C}"
        ),
        "parsed_answer": "c",
    }
    receiver = {**sender, "agent_id": "B"}

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": text.split()}

    message = EvidenceCommunicationChannel("true_evidence", "true").build_message(
        sender,
        receiver,
        {
            "tokenizer": Tokenizer(),
            "other_sender_record": sender,
            "benchmark_metadata_by_item": {
                0: {
                    "benchmark": "medqa300",
                    "answer_type": "choice",
                    "answer_options": {"c": "Nystatin"},
                }
            },
        },
    )
    assert message.text == "Nystatin treats local candidiasis."
    assert message.source_agent_id == "A"
    assert message.diagnostics["modality"] == "claim_suppressed_evidence_v1"
    assert message.diagnostics["removed_answer_cue_count"] == 2
    assert not message.diagnostics["explicit_answer_cue_present_after_filter"]


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


# --- Phase 1 is V2 verbatim -----------------------------------------------


def test_phase1_prompts_are_byte_identical_to_v2():
    """V3 rewrote phase 1 and asked for concise reasoning, which V2 never did.

    Independent-solve accuracy fell and the A/B disagreement subset shrank, so
    phase 1 is restored verbatim. These assertions keep it that way.
    """
    from icr import protocol

    assert protocol.independent_solver_prompt(
        "medqa", "Q?"
    ) == protocol.INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert protocol.independent_solver_prompt(
        "gsm8k", "Q?"
    ) == protocol.NUMERIC_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert protocol.independent_solver_prompt(
        "arc_challenge", "Q?"
    ) == protocol.GENERAL_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert protocol.independent_solver_prompt(
        "humanevalplus", "Q?"
    ) == protocol.CODE_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")


def test_phase1_never_asks_for_concise_reasoning():
    from icr import protocol

    for task in ("medqa", "gpqa", "arc_challenge", "gsm8k", "humanevalplus"):
        assert "concise" not in protocol.independent_solver_prompt(task, "Q?").lower()


def test_gpqa_is_the_only_phase1_deviation_and_disambiguates_its_layers():
    from icr import protocol

    gpqa = protocol.independent_solver_prompt("gpqa", "Q?")
    general = protocol.GENERAL_INDEPENDENT_SOLVER_PROMPT.format(question="Q?")
    assert gpqa != general
    assert gpqa == general.replace(
        "replacing A with one of A, B, C, or D.",
        "replacing A with one of A, B, C, or D." + protocol.GPQA_LAYER_DISAMBIGUATION,
        1,
    )
    assert "final A-D list" in gpqa


def test_jsonl_readers_survive_unicode_line_separators(tmp_path):
    """A record may contain U+2028, which str.splitlines() treats as a break.

    json.dumps leaves that character unescaped, so a merged.jsonl written from
    GPQA prebeliefs splits into more "lines" than it has records and the
    fragments do not parse. Every jsonl reader must split on newlines only.
    """
    import json

    record = {"item_id": 0, "agent_id": "A", "reasoning_text": "before after"}
    path = tmp_path / "merged.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    assert len([l for l in text.splitlines() if l]) == 2  # the trap
    rows = [json.loads(l) for l in text.split("\n") if l]
    assert len(rows) == 1
    assert rows[0]["reasoning_text"] == "before after"


# --- Selective rerun of truncated records --------------------------------


def test_raised_budget_keeps_finished_records_and_drops_truncated_ones(tmp_path):
    """max_new_tokens decides when generation stops, not how it samples.

    A record that reached EOS is what the same seed would produce under any
    larger budget, so raising the budget must keep it. A truncated record never
    stated its answer and was scored wrong, so it must be regenerated.
    """
    import json
    import subprocess
    import sys

    from icr.prebeliefs import accepted_fingerprints
    from icr.protocol import sha256_json

    stable = {"generation": {"max_new_tokens": 2048}, "dataset": "arc_challenge"}
    config = {**stable, "fingerprint": sha256_json(stable)}
    old = config["fingerprint"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/raise_token_budget.py", str(tmp_path), "4096"],
        check=True,
        capture_output=True,
    )
    updated = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))

    assert updated["generation"]["max_new_tokens"] == 4096
    assert updated["fingerprint"] != old
    assert old in updated["superseded_fingerprints"]
    # Records written under the old fingerprint remain acceptable.
    assert old in accepted_fingerprints(updated)
    assert updated["fingerprint"] in accepted_fingerprints(updated)
    assert updated["max_new_tokens_history"][-1]["from"] == 2048


def test_budget_cannot_be_lowered(tmp_path):
    import json
    import subprocess
    import sys

    from icr.protocol import sha256_json

    stable = {"generation": {"max_new_tokens": 4096}}
    (tmp_path / "config.json").write_text(
        json.dumps({**stable, "fingerprint": sha256_json(stable)}), encoding="utf-8"
    )
    done = subprocess.run(
        [sys.executable, "scripts/raise_token_budget.py", str(tmp_path), "2048"],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert "Refusing to lower" in done.stderr


def test_raised_budget_fingerprint_matches_a_rebuilt_config(tmp_path):
    """A raised budget must hash the same fields build_config hashes.

    config.json accumulates keys after it is first written -- which conditions
    ran, how both-correct items were sampled -- and hashing those made the
    on-disk fingerprint unreachable: the rebuilt candidate never covered them,
    so every worker refused to start on a mismatch.
    """
    import json
    import subprocess
    import sys

    from icr.prebeliefs import fingerprint_payload
    from icr.protocol import sha256_json

    stable = {"dataset": "arc_challenge", "generation": {"max_new_tokens": 2048}}
    config = {
        **stable,
        "fingerprint": sha256_json(stable),
        # Runtime bookkeeping written after the first fingerprint.
        "completed_revision_conditions": ["none", "true_text"],
        "both_correct_sampling": {"keep_one_in": 10},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/raise_token_budget.py", str(tmp_path), "4096"],
        check=True,
        capture_output=True,
    )
    raised = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))

    rebuilt = sha256_json({**stable, "generation": {"max_new_tokens": 4096}})
    assert raised["fingerprint"] == rebuilt
    assert raised["fingerprint"] == sha256_json(fingerprint_payload(raised))
    assert raised["completed_revision_conditions"] == ["none", "true_text"]


def test_raising_to_the_current_budget_is_a_no_op(tmp_path):
    """A repair can be rerun after failing part-way, so reaching the target
    budget again is success. Treating it as an error once aborted a repair that
    had already deleted the revisions it was about to regenerate."""
    import json
    import subprocess
    import sys

    from icr.protocol import sha256_json

    stable = {"generation": {"max_new_tokens": 4096}}
    config = {**stable, "fingerprint": sha256_json(stable)}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    done = subprocess.run(
        [sys.executable, "scripts/raise_token_budget.py", str(tmp_path), "4096"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0
    assert "nothing to raise" in done.stdout
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == config
