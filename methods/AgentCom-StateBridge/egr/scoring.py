"""Teacher-forced candidate scoring for Evidence-Grounded Revision."""

from __future__ import annotations

import time
from typing import Any

import torch

from icr.protocol import SYSTEM_PROMPT, sha256_text

from .prompts import build_dual_scoring_prompt, build_scoring_prompt


def egr_gate(evidence_margin: float, evidence_gain: float, tau: float = 0.0) -> bool:
    return evidence_margin > 0.0 and evidence_gain > tau


class EGRScorer:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.model = runtime.model.model
        self.tokenizer = runtime.model.tokenizer
        self.device = runtime.device

    def render_context(
        self, question: str, receiver_reasoning: str, evidence_packet: str | None
    ) -> str:
        user = build_scoring_prompt(question, receiver_reasoning, evidence_packet)
        return self.runtime.model.render_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def render_dual_context(self, question: str, evidence_packet: str | None) -> str:
        user = build_dual_scoring_prompt(question, evidence_packet)
        return self.runtime.model.render_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.inference_mode()
    def score_candidate(self, context: str, candidate: str) -> dict[str, Any]:
        context_ids = self.tokenizer(
            context, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        candidate_ids = self.tokenizer(
            candidate, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        if candidate_ids.shape[1] == 0:
            raise ValueError("Candidate answer tokenized to an empty sequence")
        input_ids = torch.cat((context_ids, candidate_ids), dim=1).to(self.device)
        labels = torch.full_like(input_ids, -100)
        labels[:, context_ids.shape[1] :] = candidate_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        score = -float(outputs.loss.detach().float().cpu())
        return {
            "average_log_likelihood": score,
            "candidate": candidate,
            "candidate_tokens": int(candidate_ids.shape[1]),
            "context_tokens": int(context_ids.shape[1]),
            "context_sha256": sha256_text(context),
        }

    def score_pair(
        self,
        *,
        question: str,
        receiver_reasoning: str,
        evidence_packet: str,
        sender_candidate: str,
        receiver_candidate: str,
    ) -> dict[str, Any]:
        prior_context = self.render_context(question, receiver_reasoning, None)
        evidence_context = self.render_context(
            question, receiver_reasoning, evidence_packet
        )
        torch.cuda.synchronize(self.device)
        started = time.time()
        prior_sender = self.score_candidate(prior_context, sender_candidate)
        prior_receiver = self.score_candidate(prior_context, receiver_candidate)
        evidence_sender = self.score_candidate(evidence_context, sender_candidate)
        evidence_receiver = self.score_candidate(evidence_context, receiver_candidate)
        torch.cuda.synchronize(self.device)
        scoring_seconds = time.time() - started

        prior_margin = (
            prior_sender["average_log_likelihood"]
            - prior_receiver["average_log_likelihood"]
        )
        evidence_margin = (
            evidence_sender["average_log_likelihood"]
            - evidence_receiver["average_log_likelihood"]
        )
        evidence_gain = evidence_margin - prior_margin
        return {
            "prior_sender": prior_sender,
            "prior_receiver": prior_receiver,
            "evidence_sender": evidence_sender,
            "evidence_receiver": evidence_receiver,
            "M0": prior_margin,
            "ME": evidence_margin,
            "G": evidence_gain,
            "gate_zero_open": egr_gate(evidence_margin, evidence_gain),
            "scoring_seconds": scoring_seconds,
        }

    def score_evidence_duel(
        self,
        *,
        question: str,
        evidence_a: str,
        evidence_b: str,
        candidate_a: str,
        candidate_b: str,
    ) -> dict[str, Any]:
        question_context = self.render_dual_context(question, None)
        context_a = self.render_dual_context(question, evidence_a)
        context_b = self.render_dual_context(question, evidence_b)
        torch.cuda.synchronize(self.device)
        started = time.time()
        q_a = self.score_candidate(question_context, candidate_a)
        q_b = self.score_candidate(question_context, candidate_b)
        a_a = self.score_candidate(context_a, candidate_a)
        a_b = self.score_candidate(context_a, candidate_b)
        b_a = self.score_candidate(context_b, candidate_a)
        b_b = self.score_candidate(context_b, candidate_b)
        torch.cuda.synchronize(self.device)

        margin_q = q_a["average_log_likelihood"] - q_b["average_log_likelihood"]
        margin_a = a_a["average_log_likelihood"] - a_b["average_log_likelihood"]
        margin_b = b_b["average_log_likelihood"] - b_a["average_log_likelihood"]
        gain_a = margin_a - margin_q
        gain_b = margin_b + margin_q
        support_a = min(margin_a, gain_a)
        support_b = min(margin_b, gain_b)
        winner = None
        if support_a > 0.0 and support_a > support_b:
            winner = "A"
        elif support_b > 0.0 and support_b > support_a:
            winner = "B"
        return {
            "question_a": q_a,
            "question_b": q_b,
            "evidence_a_candidate_a": a_a,
            "evidence_a_candidate_b": a_b,
            "evidence_b_candidate_a": b_a,
            "evidence_b_candidate_b": b_b,
            "M_question_A_vs_B": margin_q,
            "M_A": margin_a,
            "M_B": margin_b,
            "G_A": gain_a,
            "G_B": gain_b,
            "S_A": support_a,
            "S_B": support_b,
            "winner": winner,
            "scoring_seconds": time.time() - started,
        }
