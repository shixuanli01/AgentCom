"""Deterministic claim-suppressed evidence extraction and leakage diagnostics."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional


ANSWER_CUE_PATTERNS = (
    r"final\s+answer",
    r"(?:the\s+)?answer\s+is",
    r"correct\s+answer",
    r"therefore\s*,?\s*(?:the\s+)?(?:correct\s+)?answer",
    r"thus\s*,?\s*(?:the\s+)?(?:correct\s+)?answer",
    r"choose\s+(?:option\s+)?[a-d]\b",
    r"i\s+(?:would\s+)?choose\s+(?:option\s+)?[a-d]\b",
    r"best\s+answer\s+is",
    r"correct\s+option\s+is",
    r"most\s+appropriate\s+answer\s+is",
    r"\\boxed\s*\{",
)
ANSWER_CUE_RE = re.compile("|".join(f"(?:{value})" for value in ANSWER_CUE_PATTERNS), re.I)
STANDALONE_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:(?:option|choice|answer)\s*[:=-]?\s*)?"
    r"[\[(]?([a-d])[\])?.:]?(?:\*\*)?\s*$",
    re.I,
)
CONCLUSION_PREFIX_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:\*\*)?(?:conclusion|in\s+summary|thus|therefore|overall)"
    r"(?:\*\*)?\s*[:,]?",
    re.I,
)
LABEL_MENTION_TEMPLATE = (
    r"(?:\\boxed\s*\{{\s*{label}\s*\}}|"
    r"\b(?:option|choice|answer)\s*[:=-]?\s*{label}\b|"
    r"^\s*(?:[-*]\s*)?(?:\*\*)?[\[(]?{label}[\])?.:](?:\*\*)?(?:\s|$))"
)


def _normalized_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _answer_text(metadata: Mapping[str, Any], answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    options = metadata.get("answer_options") or {}
    value = options.get(str(answer).lower())
    if value is None:
        return None
    value = re.sub(r"^\s*[A-Da-d]\s*[.):]\s*", "", str(value)).strip()
    return value or None


def _contains_answer_commitment(
    line: str, answer: Optional[str], answer_text: Optional[str]
) -> bool:
    normalized_line = _normalized_text(line)
    normalized_answer = _normalized_text(answer_text or "")
    if normalized_answer and normalized_answer in normalized_line:
        return True
    if answer in {"a", "b", "c", "d"}:
        return bool(
            re.search(
                LABEL_MENTION_TEMPLATE.format(label=re.escape(answer)),
                line,
                re.I | re.M,
            )
        )
    return False


def _remove_claim_lines(
    reasoning: str, answer: Optional[str], answer_text: Optional[str]
) -> tuple[str, list[str], list[str]]:
    lines = reasoning.splitlines()
    removed_spans: list[str] = []
    removed_cues: list[str] = []
    kept: list[str] = []

    last_nonempty = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    suppress_remainder = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if suppress_remainder:
            if stripped:
                removed_spans.append(line)
            continue
        if not stripped:
            kept.append("")
            continue

        cue = ANSWER_CUE_RE.search(line)
        is_final_label_line = index == last_nonempty and bool(
            re.match(
                r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:[\[(]?[A-Da-d][\])?.:]\s+)?"
                r".{0,160}(?:\*\*)?\s*$",
                line,
            )
        ) and bool(re.search(r"(?:\\boxed|\*\*[A-Da-d][.):]|^[A-Da-d][.):])", line))

        if cue:
            prefix = line[: cue.start()].rstrip(" :-—–,;*")
            removed = line[cue.start() :]
            removed_spans.append(removed)
            removed_cues.append(cue.group(0))
            if prefix and len(_normalized_text(prefix).split()) >= 4:
                kept.append(prefix)
            if re.search(r"final\s+answer", cue.group(0), re.I) and stripped.endswith(":"):
                suppress_remainder = True
            continue
        if CONCLUSION_PREFIX_RE.search(line) and _contains_answer_commitment(
            line, answer, answer_text
        ):
            removed_spans.append(line)
            removed_cues.append("answer_bearing_conclusion")
            continue
        if STANDALONE_LABEL_RE.match(line) or is_final_label_line:
            removed_spans.append(line)
            removed_cues.append("standalone_answer_declaration")
            continue
        kept.append(line.rstrip())

    packet = "\n".join(kept)
    packet = re.sub(r"\n{3,}", "\n\n", packet).strip()
    return packet, removed_spans, removed_cues


def build_evidence_packet(
    sender_record: Mapping[str, Any],
    benchmark_metadata: Mapping[str, Any],
    *,
    token_counter: Optional[Callable[[str], int]] = None,
) -> dict[str, Any]:
    """Build deterministic evidence while recording residual claim leakage."""

    reasoning = str(sender_record.get("reasoning_text") or "")
    answer = sender_record.get("parsed_answer")
    answer = str(answer).lower() if answer is not None else None
    answer_text = _answer_text(benchmark_metadata, answer)
    answer_type = str(benchmark_metadata.get("answer_type", "choice"))
    if answer_type == "code":
        # In code tasks the implementation is the evidence itself. Removing
        # answer-bearing lines would silently corrupt indentation or behavior.
        packet, removed_spans, removed_cues = reasoning.strip(), [], []
    else:
        packet, removed_spans, removed_cues = _remove_claim_lines(
            reasoning, answer, answer_text
        )

    label_present = False
    if answer in {"a", "b", "c", "d"}:
        label_present = bool(
            re.search(
                LABEL_MENTION_TEMPLATE.format(label=re.escape(answer)),
                packet,
                re.I | re.M,
            )
        )
    normalized_packet = _normalized_text(packet)
    normalized_answer_text = _normalized_text(answer_text or "")
    answer_text_present = bool(
        normalized_answer_text
        and len(normalized_answer_text) >= 3
        and normalized_answer_text in normalized_packet
    )
    explicit_cue_present = bool(ANSWER_CUE_RE.search(packet))
    count = token_counter or (lambda value: len(value.split()))

    return {
        "benchmark": str(benchmark_metadata.get("benchmark", "unknown")),
        "answer_type": answer_type,
        "replication_id": str(sender_record.get("replication_id", "unknown")),
        "item_id": int(sender_record["item_id"]),
        "agent_id": str(sender_record["agent_id"]),
        "raw_sender_reasoning": reasoning,
        "sender_answer": answer,
        "sender_answer_text": answer_text,
        "evidence_packet": packet,
        "original_token_count": int(count(reasoning)),
        "evidence_token_count": int(count(packet)),
        "removed_spans": removed_spans,
        "removed_answer_cues": removed_cues,
        "sender_answer_label_present_after_filter": label_present,
        "sender_answer_text_present_after_filter": answer_text_present,
        "explicit_answer_cue_present_after_filter": explicit_cue_present,
    }


def summarize_leakage(packets: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(packets)

    def count(field: str) -> int:
        return sum(bool(packet.get(field)) for packet in packets)

    original = sum(int(packet["original_token_count"]) for packet in packets)
    evidence = sum(int(packet["evidence_token_count"]) for packet in packets)
    empty = sum(not str(packet.get("evidence_packet") or "").strip() for packet in packets)
    return {
        "packets": total,
        "answer_label_present": count("sender_answer_label_present_after_filter"),
        "answer_label_present_rate": count("sender_answer_label_present_after_filter") / total if total else None,
        "answer_text_present": count("sender_answer_text_present_after_filter"),
        "answer_text_present_rate": count("sender_answer_text_present_after_filter") / total if total else None,
        "explicit_answer_cue_present": count("explicit_answer_cue_present_after_filter"),
        "explicit_answer_cue_present_rate": count("explicit_answer_cue_present_after_filter") / total if total else None,
        "empty_packets": empty,
        "empty_packet_rate": empty / total if total else None,
        "original_tokens": original,
        "evidence_tokens": evidence,
        "token_retention_rate": evidence / original if original else None,
    }
