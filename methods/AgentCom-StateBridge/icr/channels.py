"""Extensible communication-channel interface for ICR."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from typing import Any, Mapping, Optional

from safetensors.torch import load_file


@dataclass
class CommunicationMessage:
    condition: str
    source_item_id: Optional[int]
    source_agent_id: Optional[str]
    text: Optional[str] = None
    prefix: Optional[torch.Tensor] = None
    trajectory: Optional[Mapping[str, Any]] = None
    diagnostics: Optional[dict[str, Any]] = None


class CommunicationChannel(ABC):
    """A channel constructs a payload without changing revision semantics."""

    condition: str

    @abstractmethod
    def build_message(
        self,
        sender_record: Mapping[str, Any],
        receiver_record: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> CommunicationMessage:
        raise NotImplementedError

    def apply_to_receiver(
        self, receiver_inputs: Mapping[str, Any], message: CommunicationMessage
    ) -> dict[str, Any]:
        return {**receiver_inputs, "communication_message": message}


class NoCommunicationChannel(CommunicationChannel):
    condition = "none"

    def build_message(self, sender_record, receiver_record, context):
        del sender_record, receiver_record, context
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=None,
            source_agent_id=None,
            diagnostics={"modality": "none", "payload_bytes": 0},
        )


class TextCommunicationChannel(CommunicationChannel):
    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown text source: {self.source}")
        text = str(source["reasoning_text"])
        tokenizer = context["tokenizer"]
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            text=text,
            diagnostics={
                "modality": "text",
                "tokens": token_count,
                "characters": len(text),
                "payload_bytes": len(text.encode("utf-8")),
            },
        )


class EvidenceCommunicationChannel(CommunicationChannel):
    """Transmit a deterministic claim-suppressed sender reasoning trace."""

    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        from egr.evidence import build_evidence_packet

        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown evidence source: {self.source}")

        tokenizer = context["tokenizer"]

        def token_count(value: str) -> int:
            return len(tokenizer(value, add_special_tokens=False)["input_ids"])

        metadata = context["benchmark_metadata_by_item"][int(source["item_id"])]
        packet = build_evidence_packet(source, metadata, token_counter=token_count)
        if packet["explicit_answer_cue_present_after_filter"]:
            raise RuntimeError(
                "Evidence channel retained an explicit answer cue; refusing to run"
            )
        text = str(packet["evidence_packet"])
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            text=text,
            diagnostics={
                "modality": "claim_suppressed_evidence_v1",
                "tokens": int(packet["evidence_token_count"]),
                "original_tokens": int(packet["original_token_count"]),
                "characters": len(text),
                "payload_bytes": len(text.encode("utf-8")),
                "payload_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "removed_span_count": len(packet["removed_spans"]),
                "removed_answer_cue_count": len(packet["removed_answer_cues"]),
                "sender_answer_label_present_after_filter": bool(
                    packet["sender_answer_label_present_after_filter"]
                ),
                "sender_answer_text_present_after_filter": bool(
                    packet["sender_answer_text_present_after_filter"]
                ),
                "explicit_answer_cue_present_after_filter": False,
            },
        )


class EvidenceV2CommunicationChannel(CommunicationChannel):
    """Sentence-level claim suppression (EGR V2).

    V1 removed the formal answer span, which on a benchmark whose options are
    noun phrases left the claim standing in prose. V2 removes whole sentences
    that name any option, symmetrically: removing only the chosen option's
    sentences made the answer readable by elimination on 52% of packets.

    A packet emptied by the filter falls back to the V1 packet and stays in the
    population rather than being dropped.
    """

    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        from egr.evidence import build_evidence_packet
        from egr.evidence_v2 import build_evidence_packet_v2

        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown evidence source: {self.source}")

        tokenizer = context["tokenizer"]

        def token_count(value: str) -> int:
            return len(tokenizer(value, add_special_tokens=False)["input_ids"])

        metadata = context["benchmark_metadata_by_item"][int(source["item_id"])]
        packet = build_evidence_packet_v2(
            source, metadata, token_counter=token_count, policy="all_options"
        )
        fell_back = False
        if not str(packet["evidence_packet"]).strip():
            packet = build_evidence_packet(source, metadata, token_counter=token_count)
            fell_back = True
        text = str(packet["evidence_packet"])
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            text=text,
            diagnostics={
                "modality": "sentence_claim_suppressed_evidence_v2",
                "policy": "all_options",
                "fell_back_to_v1": fell_back,
                "tokens": int(packet["evidence_token_count"]),
                "original_tokens": int(packet["original_token_count"]),
                "characters": len(text),
                "payload_bytes": len(text.encode("utf-8")),
                "payload_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "sentences_total": packet.get("sentences_total"),
                "sentences_removed": packet.get("sentences_removed"),
                "removal_reasons": packet.get("removal_reasons"),
                "options_still_mentioned": packet.get("options_still_mentioned"),
                "sender_answer_text_present_after_filter": bool(
                    packet["sender_answer_text_present_after_filter"]
                ),
            },
        )


class StateBridgeCommunicationChannel(CommunicationChannel):
    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown StateBridge source: {self.source}")
        prefix_path = Path(context["artifact_root"]) / source["statebridge_prefix_file"]
        prefix = load_file(str(prefix_path), device="cpu")["statebridge_prefix"]
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            prefix=prefix,
            diagnostics={
                "modality": "statebridge",
                "states": int(prefix.shape[1]),
                "hidden_dimension": int(prefix.shape[2]),
                "dtype": str(prefix.dtype).replace("torch.", ""),
                "payload_bytes": int(prefix.numel() * prefix.element_size()),
            },
        )


class CRDNCCommunicationChannel(CommunicationChannel):
    """StateBridge payload with the estimated decision direction projected out.

    Loads a prefix built exactly like the StateBridge one, from a payload that
    had ``H - alpha (H d) d^T`` applied before the same unmodified alignment. The
    alignment, the receiver prompt, the decoding settings and the revision seed
    are untouched, so a paired comparison against ``true_statebridge`` differs in
    the payload and nothing else.

    Conditions are named ``cr_dnc_a<alpha>``, with the alpha written without a
    decimal point: ``cr_dnc_a050`` is alpha = 0.50.
    """

    def __init__(self, condition: str, alpha_tag: str) -> None:
        self.condition = condition
        self.alpha_tag = alpha_tag

    def build_message(self, sender_record, receiver_record, context):
        del receiver_record
        source = sender_record
        root = Path(context["cr_dnc_prefix_root"]) / self.alpha_tag
        prefix_path = root / f"item_{int(source['item_id']):04d}_{source['agent_id']}.safetensors"
        if not prefix_path.exists():
            raise FileNotFoundError(
                f"no CR-DNC prefix for item {source['item_id']} agent "
                f"{source['agent_id']} at alpha tag {self.alpha_tag}: {prefix_path}"
            )
        loaded = load_file(str(prefix_path), device="cpu")
        prefix = loaded["statebridge_prefix"]
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            prefix=prefix,
            diagnostics={
                "modality": "cr_dnc",
                "alpha_tag": self.alpha_tag,
                "states": int(prefix.shape[1]),
                "hidden_dimension": int(prefix.shape[2]),
                "dtype": str(prefix.dtype).replace("torch.", ""),
                "payload_bytes": int(prefix.numel() * prefix.element_size()),
                "fell_back_to_raw": bool(loaded.get("fell_back_to_raw", torch.zeros(1)).item())
                if "fell_back_to_raw" in loaded else False,
            },
        )


class LatentMASCommunicationChannel(CommunicationChannel):
    """Reference an exact cached Phase-1 trajectory for KV reconstruction."""

    def __init__(self, condition: str, source: str) -> None:
        self.condition = condition
        self.source = source

    def build_message(self, sender_record, receiver_record, context):
        if self.source == "true":
            source = sender_record
        elif self.source == "self":
            source = receiver_record
        elif self.source == "other":
            source = context["other_sender_record"]
        else:
            raise ValueError(f"Unknown LatentMAS source: {self.source}")
        return CommunicationMessage(
            condition=self.condition,
            source_item_id=int(source["item_id"]),
            source_agent_id=str(source["agent_id"]),
            trajectory=source,
            diagnostics={
                "modality": "latentmas",
                "source_prompt_sha256": source["prompt_sha256"],
                "source_generation_seed": int(source["generation_seed"]),
                "source_generated_tokens": len(source["generated_token_ids"]),
                "cache_storage": "reconstructible_reference",
            },
        )


def make_channel(condition: str) -> CommunicationChannel:
    if condition == "none":
        return NoCommunicationChannel()
    if condition.endswith("_text"):
        return TextCommunicationChannel(condition, condition.removesuffix("_text"))
    if condition.endswith("_evidence_v2"):
        return EvidenceV2CommunicationChannel(
            condition, condition.removesuffix("_evidence_v2")
        )
    if condition.endswith("_evidence"):
        return EvidenceCommunicationChannel(
            condition, condition.removesuffix("_evidence")
        )
    if condition.endswith("_statebridge"):
        return StateBridgeCommunicationChannel(
            condition, condition.removesuffix("_statebridge")
        )
    if condition.endswith("_latentmas"):
        return LatentMASCommunicationChannel(
            condition, condition.removesuffix("_latentmas")
        )
    if condition.startswith("cr_dnc_"):
        return CRDNCCommunicationChannel(condition, condition.removeprefix("cr_dnc_"))
    raise ValueError(f"Unknown condition: {condition}")
