"""Frozen prompts, seeds, parsing, and atomic artifact helpers for ICR."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from utils import extract_gsm8k_answer, normalize_answer, set_seed

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
