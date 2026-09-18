"""Frozen prompts, seeds, parsing, and atomic artifact helpers for ICR."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from utils import (
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
    set_seed,
)

from . import CONDITIONS, DIRECTIONS


SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

INDEPENDENT_SOLVER_PROMPT = """You are an independent problem-solving agent.

Solve the medical multiple-choice question carefully and independently.

Reason from the evidence in the question.
Do not assume another agent will review your answer.

At the end, return exactly one final option in benchmark-compatible form: \\boxed{{A}}, replacing A with one of A, B, C, or D.

Your response should contain:
1. your reasoning
2. your final answer

Medical multiple-choice question:
{question}"""

GENERAL_INDEPENDENT_SOLVER_PROMPT = """You are an independent problem-solving agent.

Solve the multiple-choice question carefully and independently.

Reason from the evidence in the question.
Do not assume another agent will review your answer.

At the end, return exactly one final option in benchmark-compatible form: \\boxed{{A}}, replacing A with one of A, B, C, or D.

Your response should contain:
1. your reasoning
2. your final answer

Multiple-choice question:
{question}"""

NUMERIC_INDEPENDENT_SOLVER_PROMPT = """You are an independent problem-solving agent.

Solve the math word problem carefully and independently.

Show the reasoning needed to verify the calculation.
Do not assume another agent will review your answer.

At the end, return exactly one final numeric answer in the form \\boxed{{NUMBER}}.

Math word problem:
{question}"""

CODE_INDEPENDENT_SOLVER_PROMPT = """You are an independent programming agent.

Solve the programming problem carefully and independently.
Check the function signature, edge cases, and examples in the problem.
Do not assume another agent will review your answer.

Return the complete implementation in exactly one markdown Python code block.

Programming problem:
{question}"""

REVISION_BASE = """You previously solved this question independently.

Original question:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_section}

Your task is to REVISE your belief, not simply restart from scratch.

Evaluate your previous reasoning and the external information critically.

* Do not change your answer merely because another message exists.
* If the external information provides stronger evidence or identifies a real error, revise.
* If your original reasoning remains better supported, keep it.
* Resolve disagreements using the evidence in the original question.

Return:
1. concise revised reasoning
2. exactly one final benchmark-compatible answer in the form \\boxed{{A}}, replacing A with one of A, B, C, or D."""

NUMERIC_REVISION_BASE = """You previously solved this math word problem independently.

Original problem:
{question}

Your previous reasoning:
{receiver_prior_reasoning}

Your previous answer:
{receiver_prior_answer}

{external_section}

Your task is to REVISE your belief, not simply restart from scratch.

Evaluate your previous reasoning and the external information critically.

* Do not change your answer merely because another message exists.
* If the external information identifies a real calculation or reasoning error, revise.
* If your original reasoning remains better supported, keep it.
* Resolve disagreements by checking the calculation against the original problem.

Return concise revised reasoning and exactly one final numeric answer in the form
\\boxed{{NUMBER}}."""

CODE_REVISION_BASE = """You previously solved this programming problem independently.

Original problem:
{question}

Your previous reasoning and implementation:
{receiver_prior_reasoning}

{external_section}

Your task is to REVISE the implementation, not simply copy the external message.

Check the required function signature, examples, edge cases, imports, and algorithmic
correctness. Keep your original implementation when it is better supported; revise only
when the external information identifies a real defect or provides a sound improvement.

Return the complete final implementation in exactly one markdown Python code block.
Do not place tests or explanatory prose inside that code block."""


def stable_seed(global_seed: int, item_id: int, *parts: str) -> int:
    payload = "\0".join((str(global_seed), str(item_id), *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def replicated_stable_seed(
    global_seed: int, replication_id: str, item_id: int, *parts: str
) -> int:
    """Stable seed with replication identity included in a fixed field order."""
    payload = "\0".join((str(global_seed), replication_id, str(item_id), *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def prebelief_seed(
    global_seed: int,
    item_id: int,
    agent_id: str,
    replication_id: Optional[str] = None,
) -> int:
    if agent_id not in ("A", "B"):
        raise ValueError(f"Unknown agent: {agent_id}")
    if replication_id is None:
        return stable_seed(global_seed, item_id, f"agent_{agent_id}_pre")
    return replicated_stable_seed(
        global_seed, replication_id, item_id, f"agent_{agent_id}_pre"
    )


def revision_seed(
    global_seed: int,
    item_id: int,
    direction: str,
    replication_id: Optional[str] = None,
) -> int:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    if replication_id is None:
        return stable_seed(global_seed, item_id, direction, "revision")
    return replicated_stable_seed(
        global_seed, replication_id, item_id, direction, "revision"
    )


def reset_rng(seed: int) -> None:
    set_seed(seed)


def parse_medqa_answer(text: str) -> Optional[str]:
    value = normalize_answer(extract_gsm8k_answer(text))
    return value if value in set("abcd") else None


def parse_multiple_choice_answer(text: str) -> Optional[str]:
    """Parse the shared A-D output contract used by all current ICR tasks."""
    return parse_medqa_answer(text)


def parse_task_answer(task: str, text: str) -> Optional[str]:
    if task == "gsm8k":
        return normalize_answer(extract_gsm8k_answer(text))
    if task in {"mbppplus", "humanevalplus"}:
        return extract_markdown_python_block(text)
    return parse_multiple_choice_answer(text)


def canonical_gold(task: str, value: Any) -> Optional[str]:
    if task in {"mbppplus", "humanevalplus"}:
        return None if value is None else str(value)
    if task == "gsm8k":
        return normalize_answer(None if value is None else str(value))
    return canonical_answer(value)


def answer_is_correct(task: str, prediction: Any, gold: Any) -> bool:
    if task in {"mbppplus", "humanevalplus"}:
        if prediction is None or gold is None:
            return False
        passed, _ = run_with_timeout(f"{prediction}\n{gold}", timeout=10)
        return passed
    normalized_prediction = canonical_gold(task, prediction)
    normalized_gold = canonical_gold(task, gold)
    return (
        normalized_prediction is not None
        and normalized_gold is not None
        and normalized_prediction == normalized_gold
    )


def independent_solver_prompt(task: str, question: str) -> str:
    if task == "medqa":
        template = INDEPENDENT_SOLVER_PROMPT
    elif task == "gsm8k":
        template = NUMERIC_INDEPENDENT_SOLVER_PROMPT
    elif task in {"mbppplus", "humanevalplus"}:
        template = CODE_INDEPENDENT_SOLVER_PROMPT
    else:
        template = GENERAL_INDEPENDENT_SOLVER_PROMPT
    return template.format(question=question)


def revision_prompt(
    task: str,
    *,
    question: str,
    receiver_prior_reasoning: str,
    receiver_prior_answer: Optional[str],
    external_section: str,
) -> str:
    if task == "gsm8k":
        template = NUMERIC_REVISION_BASE
    elif task in {"mbppplus", "humanevalplus"}:
        template = CODE_REVISION_BASE
    else:
        template = REVISION_BASE
    return template.format(
        question=question,
        receiver_prior_reasoning=receiver_prior_reasoning,
        receiver_prior_answer=receiver_prior_answer or "UNPARSEABLE",
        external_section=external_section,
    )


def canonical_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in set("abcd") else None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(payload)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_conditions(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(requested) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if not requested:
        raise ValueError("At least one condition is required")
    if len(requested) != len(set(requested)):
        raise ValueError("Conditions must not be repeated")
    return requested


def other_item_id(item_id: int, selected_ids: Sequence[int], offset: int = 137) -> int:
    ordered = list(selected_ids)
    if len(ordered) < 2:
        raise ValueError("Other-message controls require at least two selected items")
    try:
        position = ordered.index(item_id)
    except ValueError as error:
        raise ValueError(f"Item {item_id} is not in the selected set") from error
    other = ordered[(position + offset) % len(ordered)]
    if other == item_id:
        other = ordered[(position + 1) % len(ordered)]
    return int(other)


def classify_pair(sender_correct: bool, receiver_correct: bool) -> str:
    if sender_correct and not receiver_correct:
        return "correction_opportunity"
    if not sender_correct and receiver_correct:
        return "destruction_risk"
    if not sender_correct and not receiver_correct:
        return "both_wrong"
    return "both_correct"
