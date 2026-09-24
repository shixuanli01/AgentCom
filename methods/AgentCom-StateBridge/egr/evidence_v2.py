"""EGR V2: sentence-level claim suppression.

V1 removed the formal answer span and left everything else. On a benchmark whose
options are noun phrases rather than symbols that is close to a no-op: on MedQA
it dropped a median of 8 tokens, kept over 95% of the text in 78% of packets, and
the sender's chosen option still appeared in 79.3% of them against 81.3% in the
raw reasoning. The resulting channel behaved like Full Text, agreeing with it on
658 of 720 revisions.

V2 works at sentence granularity. A sentence that asserts the answer is removed
whole, because the claim is carried by the sentence, not by the span:

    "Given the colonoscopic findings, the greatest risk is colorectal cancer."

Deleting only "colorectal cancer" leaves a sentence that still points at it.

A sentence is removed when it carries any of:
  * a ``\\boxed{...}`` span;
  * the chosen option's text, or a distinctive content word from it;
  * the chosen option's label in an answer-asserting context ("the answer is C",
    "option (C)", "选 C");
  * a conclusion cue together with any option's text.

Removing only the chosen option's sentences leaks by omission, and badly:
measured on MedQA, 491 of 900 packets end up missing exactly one option, and in
95.3% of those the missing one is the sender's choice against a 25% chance
baseline. Reading off the absent option recovers the answer for 52% of all
packets, so suppressing the claim this way mostly changes how it is transmitted.

``policy`` controls that. ``"chosen"`` removes only the chosen option's
sentences and carries the omission leak. ``"all_options"`` removes any sentence
that names any option, which costs more of the argument but leaves nothing to
eliminate against. Both are measured rather than assumed.

This is a new method version. V1 stays in ``egr/evidence.py`` and its results are
not superseded.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional, Sequence

METHOD_VERSION = "egr_v2_sentence_claim_suppression"

# A sentence ends at ., ?, ! or a newline. Abbreviations are not special-cased:
# an over-eager split removes a shorter fragment, which is the safe direction.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_BOXED = re.compile(r"\\boxed\s*\{", re.I)
_CONCLUSION_CUES = (
    "therefore", "thus", "hence", "so the answer", "the answer is", "in conclusion",
    "conclusion:", "final answer", "most likely", "best answer", "correct answer",
    "answer:", "因此", "所以", "答案", "综上",
)
# "the answer is 18", "Final Answer: 18", "so we get 18", "答案是 18"
_NUMERIC_ASSERTION = (
    r"(?:answer|result|total(?:s|ing)?|altogether|in all|final|therefore|thus|hence|"
    r"so\s+(?:the|we|she|he|they)|答案|结果|总共|因此|所以)"
    r"[^.\n]{{0,40}}?{n}"
)
# A sentence showing operands and an operator is a derivation step, not an
# announcement. "Total = 14 + 20 = 34" states the answer but is also the
# arithmetic that produces it; removing it leaves the receiver every
# intermediate quantity and no landing point. "The answer is 34" is not.
_ARITHMETIC = re.compile(r"[0-9)\}]\s*(?:[-+*/=]|\\times|\\div|\\cdot|x)\s*[0-9(\\]")
_LABEL_ASSERTION = (
    r"(?:answer|option|choice|select|choose|pick|响应|选项|答案)"
    r"[^.\n]{{0,24}}\b\(?{label}\)?\b"
)
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "with", "for", "by",
    "syndrome", "disease", "disorder", "cancer", "acute", "chronic", "primary",
}


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _content_words(option_text: str) -> list[str]:
    """Distinctive words of an option, used when the full phrase is paraphrased."""
    words = re.findall(r"[a-z]{4,}", _normalise(option_text))
    return [w for w in words if w not in _STOPWORDS]


def _is_matchable(option_text: str) -> bool:
    """Whether an option's text can be searched for as content.

    Some GPQA items carry the A-D to a-d mapping layer in place of option text,
    so an "option" is the single character "d". Substring-matching that would
    delete almost every sentence, since most contain the letter. Options that
    short are left to the label-assertion rule instead.
    """
    return len(_normalise(option_text)) >= 4


def _mentions(sentence: str, option_text: str) -> bool:
    if not _is_matchable(option_text):
        return False
    low = _normalise(sentence)
    if _normalise(option_text) in low:
        return True
    words = _content_words(option_text)
    # A single distinctive word is enough only when the option has just one.
    if not words:
        return False
    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}", low))
    return hits == len(words) if len(words) == 1 else hits >= max(2, len(words) - 1)


def build_evidence_packet_v2(
    sender_record: Mapping[str, Any],
    benchmark_metadata: Mapping[str, Any],
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    policy: str = "all_options",
    numeric_policy: str = "strict",
) -> dict[str, Any]:
    """Remove whole sentences that assert the sender's answer.

    ``policy`` is ``"all_options"`` (default, symmetric) or ``"chosen"``.
    """
    if policy not in ("chosen", "all_options"):
        raise ValueError(f"unknown policy {policy!r}")
    if numeric_policy not in ("strict", "assertion_only"):
        raise ValueError(f"unknown numeric_policy {numeric_policy!r}")
    reasoning = str(sender_record.get("reasoning_text") or "")
    answer = sender_record.get("parsed_answer")
    answer = str(answer).strip().lower() if answer is not None else None
    options: Mapping[str, str] = benchmark_metadata.get("answer_options") or {}
    answer_type = str(benchmark_metadata.get("answer_type", "choice"))
    count = token_counter or (lambda text: len(text.split()))

    if answer_type == "code":
        # The implementation is the evidence; deleting lines would break it.
        packet = reasoning.strip()
        return {
            "method_version": METHOD_VERSION, "evidence_packet": packet,
            "original_token_count": count(reasoning), "evidence_token_count": count(packet),
            "sentences_total": 0, "sentences_removed": 0, "removed_sentences": [],
            "removal_reasons": {}, "policy": policy,
            "sender_answer_text_present_after_filter": False,
            "explicit_answer_cue_present_after_filter": False,
            "options_still_mentioned": [], "skipped_reason": "code task",
        }

    chosen_text = options.get(answer) if answer else None
    numeric_answer = None
    if answer_type == "number" and answer:
        # A numeric benchmark has no option set, so the claim is the number
        # itself. Sentences stating it are removed the way option-naming
        # sentences are; there are no alternatives to stay symmetric with.
        numeric_answer = re.sub(r"[^0-9.\-]", "", str(answer))
    sentences = [s for s in _SENTENCE_SPLIT.split(reasoning) if s.strip()]
    kept, removed, reasons = [], [], {}

    for sentence in sentences:
        why = None
        if _BOXED.search(sentence):
            why = "boxed_span"
        elif numeric_answer and re.search(
            rf"(?<![0-9.]){re.escape(numeric_answer)}(?![0-9.])", sentence
        ) and (
            numeric_policy == "strict"
            or (
                re.search(_NUMERIC_ASSERTION.format(n=re.escape(numeric_answer)),
                          sentence, re.I)
                and not _ARITHMETIC.search(sentence)
            )
        ):
            # strict removes every sentence stating the number, which in
            # arithmetic is the closing step of the derivation: deleting
            # "48 x 120 = 5760" leaves the receiver a reasoning chain with every
            # intermediate quantity and no landing point. assertion_only removes
            # the sentences that announce the number as the answer and keeps the
            # arithmetic that produced it.
            why = "numeric_answer" if numeric_policy == "strict" else "numeric_assertion"
        elif chosen_text and _mentions(sentence, chosen_text):
            why = "chosen_option_text"
        elif policy == "all_options" and any(
            _mentions(sentence, text) for text in options.values()
        ):
            # Symmetry is the point: an option left standing while the chosen one
            # is gone is exactly what makes the omission readable.
            why = "any_option_text"
        elif answer and re.search(
            _LABEL_ASSERTION.format(label=re.escape(answer)), sentence, re.I
        ):
            why = "label_assertion"
        elif policy == "all_options" and re.search(
            _LABEL_ASSERTION.format(label="[a-d]"), sentence, re.I
        ):
            why = "any_label_assertion"
        else:
            low = _normalise(sentence)
            if any(cue in low for cue in _CONCLUSION_CUES) and chosen_text and _mentions(
                sentence, chosen_text
            ):
                why = "conclusive_about_chosen"
        if why:
            removed.append(sentence.strip())
            reasons[why] = reasons.get(why, 0) + 1
        else:
            kept.append(sentence.strip())

    packet = " ".join(kept).strip()
    still = [label for label, text in options.items() if _mentions(packet, text)]
    return {
        "method_version": METHOD_VERSION,
        "evidence_packet": packet,
        "original_token_count": count(reasoning),
        "evidence_token_count": count(packet),
        "sentences_total": len(sentences),
        "sentences_removed": len(removed),
        "removed_sentences": removed,
        "removal_reasons": reasons,
        "policy": policy,
        "numeric_policy": numeric_policy,
        "sender_answer_text_present_after_filter": bool(
            chosen_text and _mentions(packet, chosen_text)
        ),
        "explicit_answer_cue_present_after_filter": bool(_BOXED.search(packet)),
        "options_still_mentioned": sorted(still),
        "skipped_reason": None,
    }
