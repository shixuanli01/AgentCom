"""EGR V3: one suppression rule for every answer type.

V2 needed a different rule per answer type -- remove every sentence naming an
option on a choice task, remove every sentence stating the number on a numeric
one -- and the numeric rule turned out to delete the closing arithmetic, leaving
the receiver a derivation with every intermediate quantity and no landing point.
Fixing that per type invites the obvious objection that the method is tuned per
dataset.

V3 states one criterion instead:

    remove a sentence that announces the answer without showing why;
    keep a sentence that shows why.

That is answer-type agnostic, because "showing why" has the same meaning in both
places. In arithmetic it is operands and an operator: "Total = 14 + 20 = 34"
derives its number. In a differential it is an evidential connective:
"Pseudopolyps are characteristic of inflammatory bowel disease" derives its
claim. "Therefore the answer is C" derives nothing, and neither does
``\\boxed{C}``.

Suppression is symmetric over the answer space, as in V2 and for the same
reason: removing conclusions only about the chosen answer leaves the others
discussed and the chosen one conspicuously absent, which on MedQA recovered the
answer by elimination for 52% of packets.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional

from egr.evidence_v2 import _mentions, _normalise, _is_matchable

METHOD_VERSION = "egr_v3_assert_not_derive"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_BOXED = re.compile(r"\\boxed\s*\{", re.I)

# Announcing: the sentence presents something as the answer.
_ANNOUNCE = re.compile(
    r"\b(?:the\s+)?(?:answer|final\s+answer|result|conclusion|choice|option)\b"
    r"|\b(?:therefore|thus|hence|so|in\s+conclusion|overall|altogether|in\s+all)\b"
    r"|\b(?:most\s+likely|best\s+answer|correct\s+answer|the\s+greatest|is\s+the\s+most)\b"
    r"|答案|因此|所以|综上|结论",
    re.I,
)

# Deriving: the sentence shows the work or the grounds.
_DERIVE = re.compile(
    r"[0-9)\}]\s*(?:[-+*/=]|\\times|\\div|\\cdot)\s*[0-9(\\]"          # arithmetic
    r"|\b(?:because|since|due\s+to|given\s+that|as\s+evidenced|evidence|"
    r"indicat\w*|suggest\w*|consistent\s+with|characteristic\s+of|"
    r"associated\s+with|supports?|rules?\s+out|unlikely|differential|"
    r"presents?\s+with|reveals?|shows?|finding)\b"
    r"|因为|由于|提示|表明|支持",
    re.I,
)


def _answer_space(metadata: Mapping[str, Any], answer: Optional[str]) -> list[str]:
    """Every surface form the answer space can take, for symmetric removal."""
    answer_type = str(metadata.get("answer_type", "choice"))
    if answer_type == "number":
        if not answer:
            return []
        return [re.sub(r"[^0-9.\-]", "", str(answer))]
    options = metadata.get("answer_options") or {}
    return [text for text in options.values() if _is_matchable(text)]


def _references(sentence: str, space: list[str], numeric: bool) -> bool:
    if numeric:
        return any(
            re.search(rf"(?<![0-9.]){re.escape(value)}(?![0-9.])", sentence)
            for value in space if value
        )
    return any(_mentions(sentence, text) for text in space)


def build_evidence_packet_v3(
    sender_record: Mapping[str, Any],
    benchmark_metadata: Mapping[str, Any],
    *,
    token_counter: Optional[Callable[[str], int]] = None,
) -> dict[str, Any]:
    reasoning = str(sender_record.get("reasoning_text") or "")
    answer = sender_record.get("parsed_answer")
    answer = str(answer).strip().lower() if answer is not None else None
    answer_type = str(benchmark_metadata.get("answer_type", "choice"))
    count = token_counter or (lambda text: len(text.split()))

    if answer_type == "code":
        packet = reasoning.strip()
        return {
            "method_version": METHOD_VERSION, "evidence_packet": packet,
            "original_token_count": count(reasoning), "evidence_token_count": count(packet),
            "sentences_total": 0, "sentences_removed": 0, "removed_sentences": [],
            "removal_reasons": {}, "skipped_reason": "code task",
        }

    numeric = answer_type == "number"
    space = _answer_space(benchmark_metadata, answer)
    sentences = [s for s in _SENTENCE_SPLIT.split(reasoning) if s.strip()]
    kept, removed, reasons = [], [], {}

    for sentence in sentences:
        why = None
        if _BOXED.search(sentence):
            why = "boxed_span"
        elif _references(sentence, space, numeric):
            announces = bool(_ANNOUNCE.search(sentence))
            derives = bool(_DERIVE.search(sentence))
            if announces and not derives:
                why = "announces_without_deriving"
        if why:
            removed.append(sentence.strip())
            reasons[why] = reasons.get(why, 0) + 1
        else:
            kept.append(sentence.strip())

    packet = " ".join(kept).strip()
    chosen = None
    if not numeric:
        chosen = (benchmark_metadata.get("answer_options") or {}).get(answer)
    still = [
        label for label, text in (benchmark_metadata.get("answer_options") or {}).items()
        if _is_matchable(text) and _mentions(packet, text)
    ]
    return {
        "method_version": METHOD_VERSION,
        "evidence_packet": packet,
        "original_token_count": count(reasoning),
        "evidence_token_count": count(packet),
        "sentences_total": len(sentences),
        "sentences_removed": len(removed),
        "removed_sentences": removed,
        "removal_reasons": reasons,
        "answer_space_size": len(space),
        "sender_answer_text_present_after_filter": bool(
            _references(packet, [space[0]] if numeric and space else ([chosen] if chosen else []),
                        numeric)
        ),
        "explicit_answer_cue_present_after_filter": bool(_BOXED.search(packet)),
        "options_still_mentioned": sorted(still),
        "skipped_reason": None,
    }
