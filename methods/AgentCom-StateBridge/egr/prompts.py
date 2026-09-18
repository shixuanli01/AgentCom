"""Frozen EGR-M1 message and continuation-scoring templates."""

from __future__ import annotations


CLAIM_ONLY_MESSAGE = """The other reasoning process selected:
{sender_answer}"""

SCORING_TEMPLATE = """Question:
{question}

My previous reasoning:
{receiver_prior_reasoning}
{evidence_section}
Based only on the question, my prior reasoning, and the available evidence,
the answer best supported by the evidence is:
"""


def build_scoring_prompt(
    question: str, receiver_prior_reasoning: str, evidence_packet: str | None
) -> str:
    evidence_section = ""
    if evidence_packet is not None:
        evidence_section = (
            "\nClaim-suppressed evidence from another reasoning process:\n"
            f"{evidence_packet}\n"
        )
    return SCORING_TEMPLATE.format(
        question=question,
        receiver_prior_reasoning=receiver_prior_reasoning,
        evidence_section=evidence_section,
    )


DUAL_SCORING_TEMPLATE = """Question:
{question}

{evidence_section}Based only on the question and the available evidence,
the answer best supported by the evidence is:
"""


def build_dual_scoring_prompt(question: str, evidence_packet: str | None) -> str:
    evidence_section = ""
    if evidence_packet is not None:
        evidence_section = (
            "Candidate evidence (explicit final claim suppressed):\n"
            f"{evidence_packet}\n\n"
        )
    return DUAL_SCORING_TEMPLATE.format(
        question=question,
        evidence_section=evidence_section,
    )


CONTRAST_ADJUDICATION_TEMPLATE = """You are adjudicating two independent attempts at the same medical multiple-choice problem.

Original question:
{question}

Evidence set 1 (its explicit final-answer declaration has been suppressed):
{evidence_a}

Evidence set 2 (its explicit final-answer declaration has been suppressed):
{evidence_b}

Solve the question by comparing the evidence, not by voting or trusting either source.
Identify the decisive clinical facts, check each evidence set for factual or causal errors,
and resolve contradictions against the original question.

Return:
1. a concise evidence adjudication;
2. exactly one final answer in the form \\boxed{{A}}, replacing A with A, B, C, or D.
"""

GENERAL_CONTRAST_ADJUDICATION_TEMPLATE = """You are adjudicating two independent attempts at the same multiple-choice problem.

Original question:
{question}

Evidence set 1 (its explicit final-answer declaration has been suppressed):
{evidence_a}

Evidence set 2 (its explicit final-answer declaration has been suppressed):
{evidence_b}

Solve the question by comparing the evidence, not by voting or trusting either source.
Identify the decisive facts, check each evidence set for factual or causal errors,
and resolve contradictions against the original question.

Return:
1. a concise evidence adjudication;
2. exactly one final answer in the form \\boxed{{A}}, replacing A with A, B, C, or D.
"""

NUMERIC_CONTRAST_ADJUDICATION_TEMPLATE = """You are adjudicating two independent attempts at the same math word problem.

Original problem:
{question}

Evidence set 1 (its explicit final-answer declaration has been suppressed):
{evidence_a}

Evidence set 2 (its explicit final-answer declaration has been suppressed):
{evidence_b}

Solve the problem by comparing the evidence, not by voting or trusting either source.
Recompute the decisive quantities, identify arithmetic or interpretation errors, and resolve
contradictions against the original problem.

Return a concise evidence adjudication followed by exactly one final numeric answer in the
form \\boxed{{NUMBER}}.
"""

CODE_CONTRAST_ADJUDICATION_TEMPLATE = """You are adjudicating two independent implementations of the same programming problem.

Original programming problem:
{question}

Candidate attempt 1:
{evidence_a}

Candidate attempt 2:
{evidence_b}

Treat both attempts as untrusted. Check the required function signature, algorithm,
imports, examples, and edge cases. Preserve correct ideas, repair concrete defects, and
produce one self-contained final implementation. Do not vote based on surface agreement.

Return concise reasoning followed by exactly one markdown Python code block containing
the complete implementation. Do not put tests or prose inside the code block.
"""


def build_contrast_adjudication_prompt(
    question: str, evidence_a: str, evidence_b: str, *, task: str = "medqa"
) -> str:
    if task == "medqa":
        template = CONTRAST_ADJUDICATION_TEMPLATE
    elif task == "gsm8k":
        template = NUMERIC_CONTRAST_ADJUDICATION_TEMPLATE
    elif task in {"mbppplus", "humanevalplus"}:
        template = CODE_CONTRAST_ADJUDICATION_TEMPLATE
    else:
        template = GENERAL_CONTRAST_ADJUDICATION_TEMPLATE
    return template.format(
        question=question,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
    )


FALSIFICATION_TEMPLATE = """You are the final verifier for a medical multiple-choice problem.

Original question:
{question}

Evidence set 1 (explicit final-answer declaration suppressed):
{evidence_a}

Evidence set 2 (explicit final-answer declaration suppressed):
{evidence_b}

A preliminary adjudicator proposed the following analysis and answer:
{preliminary_adjudication}

The preliminary answer is only a hypothesis and may contain a confident medical error.
Try to falsify it before accepting it:
1. restate the decisive clinical clue or task requirement from the original question;
2. identify the single factual or causal statement in the preliminary analysis most likely to be wrong;
3. check the proposed answer against the strongest competing option using established medical knowledge;
4. keep the proposed answer only if it survives that check.

Do not defer to the preliminary adjudicator or vote between evidence sets.
Return a concise verification followed by exactly one final answer in the form
\\boxed{{A}}, replacing A with A, B, C, or D.
"""


def build_falsification_prompt(
    question: str,
    evidence_a: str,
    evidence_b: str,
    preliminary_adjudication: str,
) -> str:
    return FALSIFICATION_TEMPLATE.format(
        question=question,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        preliminary_adjudication=preliminary_adjudication,
    )


EVIDENCE_LEDGER_TEMPLATE = """You are constructing a verified evidence ledger for a medical multiple-choice problem.

Original question:
{question}

Independent reasoning trace 1 (explicit final-answer declaration suppressed):
{evidence_a}

Independent reasoning trace 2 (explicit final-answer declaration suppressed):
{evidence_b}

Do not choose, recommend, or state a final answer. Do not vote between the traces.
Build a compact ledger containing only:
1. the exact clinical findings and task requirement stated in the question;
2. established medical facts needed to solve it;
3. agreements between the traces;
4. contradictions or unsupported claims, with the medically justified resolution when possible.

Treat both traces as untrusted. Omit a claim if it cannot be justified from the question or
established medical knowledge. End with the heading "Remaining uncertainty:" and state any
unresolved factual issue, without naming an answer choice.
"""


LEDGER_DECISION_TEMPLATE = """Solve this medical multiple-choice problem using the original question and a verified evidence ledger.

Original question:
{question}

Verified evidence ledger:
{ledger}

The raw candidate conclusions were intentionally withheld. Independently match the decisive
facts to the answer options in the original question. The ledger may be incomplete, so correct
it using established medical knowledge if necessary.

Return a concise justification followed by exactly one final answer in the form
\\boxed{{A}}, replacing A with A, B, C, or D.
"""


def build_evidence_ledger_prompt(question: str, evidence_a: str, evidence_b: str) -> str:
    return EVIDENCE_LEDGER_TEMPLATE.format(
        question=question,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
    )


def build_ledger_decision_prompt(question: str, ledger: str) -> str:
    return LEDGER_DECISION_TEMPLATE.format(question=question, ledger=ledger)


HYPOTHESIS_ADJUDICATION_TEMPLATE = """You are adjudicating two untrusted hypotheses for the same medical multiple-choice problem.

Original question:
{question}

Hypothesis 1 proposes option {answer_a}: {answer_text_a}
Its claim-suppressed supporting trace is:
{evidence_a}

Hypothesis 2 proposes option {answer_b}: {answer_text_b}
Its claim-suppressed supporting trace is:
{evidence_b}

The hypotheses receive no weight merely because they were proposed. Do not vote.
For each hypothesis, identify the decisive fact that would have to be true, then check that
fact against the clinical details, the exact task wording, and established medical knowledge.
Explicitly identify a factual or causal error when one is present. You may select another
option from the original question if both hypotheses fail.

Return a concise comparative audit followed by exactly one final answer in the form
\\boxed{{A}}, replacing A with A, B, C, or D.
"""


def build_hypothesis_adjudication_prompt(
    question: str,
    answer_a: str,
    answer_text_a: str,
    evidence_a: str,
    answer_b: str,
    answer_text_b: str,
    evidence_b: str,
) -> str:
    return HYPOTHESIS_ADJUDICATION_TEMPLATE.format(
        question=question,
        answer_a=answer_a.upper(),
        answer_text_a=answer_text_a,
        evidence_a=evidence_a,
        answer_b=answer_b.upper(),
        answer_text_b=answer_text_b,
        evidence_b=evidence_b,
    )
