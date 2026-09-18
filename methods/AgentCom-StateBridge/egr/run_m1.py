"""Run EGR checkpoint M1 from immutable cached ICR prebeliefs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoTokenizer

from icr.channels import CommunicationMessage
from icr.benchmarks import benchmark_metadata, benchmark_spec, load_benchmark
from icr.protocol import (
    atomic_write_json,
    atomic_write_jsonl,
    answer_is_correct,
    classify_pair,
    revision_seed,
    sha256_json,
    sha256_text,
)
from icr.runtime import ICRRuntime

from .evidence import build_evidence_packet, summarize_leakage
from .prompts import CLAIM_ONLY_MESSAGE
from .scoring import EGRScorer


NEW_CONDITIONS = ("claim_only", "evidence_only", "egr_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGR checkpoint M1")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--conditions", nargs="+", choices=NEW_CONDITIONS, default=list(NEW_CONDITIONS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--item-ids", nargs="+", type=int)
    parser.add_argument("--leakage-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def medqa_metadata(path: Path) -> list[dict[str, Any]]:
    """Legacy helper retained for compatibility with earlier MedQA artifacts."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for item in rows:
        options = {
            chr(ord("a") + index): str(value)
            for index, value in enumerate(item["options"])
        }
        result.append({"benchmark": "medqa300", "answer_options": options})
    return result


def canonical_option_text(metadata: Mapping[str, Any], answer: str) -> str:
    import re

    value = str(metadata["answer_options"][answer.lower()])
    return re.sub(r"^\s*[A-Da-d]\s*[.):]\s*", "", value).strip()


def ensure_config(
    *,
    output_root: Path,
    source_root: Path,
    source_config: Mapping[str, Any],
    selected_ids: list[int],
    model: str,
    conditions: Iterable[str],
) -> dict[str, Any]:
    stable = {
        "protocol": "EGR-M1-v1",
        "status": "diagnostic",
        "source_protocol": source_config.get("protocol", "ICR-MEDQA300-V2"),
        "source_root": str(source_root.resolve()),
        "source_config_fingerprint": source_config.get("fingerprint"),
        "replication_id": source_config.get("replication_id", "seed_pair_00"),
        "model": model,
        "task": source_config.get("dataset", "medqa"),
        "benchmark": source_config.get("benchmark", "medqa300"),
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "selected_item_ids": selected_ids,
        "conditions": list(conditions),
        "evidence_builder": "claim_suppressed_evidence_v1",
        "scoring_prompt": "egr_continuation_v1_thinking_disabled",
        "gate": {"name": "EGR-zero", "ME_gt": 0.0, "G_gt": 0.0},
        "threshold_tuned_on_benchmark": False,
    }
    candidate = {**stable, "fingerprint": sha256_json(stable)}
    path = output_root / "config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != candidate["fingerprint"]:
            raise RuntimeError("EGR output root contains a different configuration")
        return existing
    atomic_write_json(path, candidate)
    return candidate


def prepare_evidence_packets(
    *,
    prebeliefs: Mapping[tuple[int, str], Mapping[str, Any]],
    metadata: list[Mapping[str, Any]],
    selected_ids: list[int],
    output_root: Path,
    tokenizer: Any,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    records_dir = output_root / "evidence_packets" / "records"
    packets: dict[tuple[int, str], dict[str, Any]] = {}

    def token_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    for item_id in selected_ids:
        for agent_id in ("A", "B"):
            key = (item_id, agent_id)
            path = records_dir / f"item_{item_id:04d}_{agent_id}.json"
            if path.exists():
                packet = json.loads(path.read_text(encoding="utf-8"))
            else:
                packet = build_evidence_packet(
                    prebeliefs[key], metadata[item_id], token_counter=token_count
                )
                atomic_write_json(path, packet)
            packets[key] = packet
    ordered = [packets[key] for key in sorted(packets)]
    atomic_write_jsonl(output_root / "evidence_packets" / "merged.jsonl", ordered)
    leakage = summarize_leakage(ordered)
    leakage.update(
        {
            "interpretation": (
                "claim-suppressed, not answer-blind; answer-text presence may be legitimate evidence"
            ),
            "hard_stop_explicit_cue_leakage": bool(leakage["explicit_answer_cue_present"]),
        }
    )
    atomic_write_json(output_root / "evidence_packets" / "leakage_summary.json", leakage)
    return packets, leakage


def base_record(
    *,
    sender: Mapping[str, Any],
    receiver: Mapping[str, Any],
    condition: str,
    direction: str,
    revision_seed_value: int,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "benchmark": sender.get("benchmark", "medqa300"),
        "task": sender.get("task", "medqa"),
        "replication_id": str(sender.get("replication_id", "seed_pair_00")),
        "item_id": int(sender["item_id"]),
        "direction": direction,
        "condition": condition,
        "sender_agent_id": str(sender["agent_id"]),
        "receiver_agent_id": str(receiver["agent_id"]),
        "sender_pre_answer": sender.get("parsed_answer"),
        "sender_correct": bool(sender["correct"]),
        "receiver_pre_answer": receiver.get("parsed_answer"),
        "receiver_pre_correct": bool(receiver["correct"]),
        "receiver_prior_sha256": sha256_text(str(receiver["reasoning_text"])),
        "gold": receiver["gold"],
        "pair_classification": classify_pair(
            bool(sender["correct"]), bool(receiver["correct"])
        ),
        "revision_seed": revision_seed_value,
    }


def finalize_record(record: dict[str, Any], post_answer: Any) -> dict[str, Any]:
    correct = answer_is_correct(str(record.get("task", "medqa")), post_answer, record["gold"])
    return {
        **record,
        "receiver_post_answer": post_answer,
        "receiver_post_correct": correct,
        "answer_changed": post_answer != record["receiver_pre_answer"],
        "followed_sender": post_answer == record["sender_pre_answer"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def run_generation_condition(
    *,
    runtime: ICRRuntime,
    condition: str,
    pairs: list[tuple[int, str]],
    prebeliefs: Mapping[tuple[int, str], Mapping[str, Any]],
    packets: Mapping[tuple[int, str], Mapping[str, Any]],
    output_root: Path,
    global_seed: int,
    replication_id: str,
    seed_replication_id: str | None,
    stop_requested: callable,
) -> None:
    records_dir = output_root / "revisions" / condition / "records"
    for item_id, direction in pairs:
        sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
        sender = prebeliefs[(item_id, sender_id)]
        receiver = prebeliefs[(item_id, receiver_id)]
        record_path = records_dir / f"item_{item_id:04d}_{direction}.json"
        if record_path.exists():
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                continue
        seed = revision_seed(global_seed, item_id, direction, seed_replication_id)
        if condition == "claim_only":
            text = CLAIM_ONLY_MESSAGE.format(
                sender_answer=(str(sender.get("parsed_answer") or "UNPARSEABLE").upper())
            )
        else:
            text = str(packets[(item_id, sender_id)]["evidence_packet"])
        token_count = len(runtime.model.tokenizer(text, add_special_tokens=False)["input_ids"])
        message = CommunicationMessage(
            condition=condition,
            source_item_id=item_id,
            source_agent_id=sender_id,
            text=text,
            diagnostics={
                "modality": condition,
                "tokens": token_count,
                "characters": len(text),
                "payload_bytes": len(text.encode("utf-8")),
            },
        )
        generated = runtime.generate_revision(
            question=str(receiver["question"]),
            receiver_reasoning=str(receiver["reasoning_text"]),
            receiver_answer=receiver.get("parsed_answer"),
            message=message,
            seed=seed,
        )
        record = base_record(
            sender=sender,
            receiver=receiver,
            condition=condition,
            direction=direction,
            revision_seed_value=seed,
        )
        record.update(
            {
                "message_source_item_id": item_id,
                "message_source_agent_id": sender_id,
                "communication_payload": message.diagnostics,
                "evidence_packet_file": (
                    f"evidence_packets/records/item_{item_id:04d}_{sender_id}.json"
                    if condition == "evidence_only"
                    else None
                ),
                **generated,
            }
        )
        atomic_write_json(
            record_path, finalize_record(record, generated["parsed_answer"])
        )
        print(
            f"[EGR {condition}] item={item_id} {direction} "
            f"pre={receiver.get('parsed_answer')} sender={sender.get('parsed_answer')} "
            f"post={generated['parsed_answer']}",
            flush=True,
        )
        if stop_requested():
            break


def run_egr_zero(
    *,
    scorer: EGRScorer,
    pairs: list[tuple[int, str]],
    prebeliefs: Mapping[tuple[int, str], Mapping[str, Any]],
    packets: Mapping[tuple[int, str], Mapping[str, Any]],
    metadata: list[Mapping[str, Any]],
    output_root: Path,
    global_seed: int,
    replication_id: str,
    seed_replication_id: str | None,
    stop_requested: callable,
) -> None:
    records_dir = output_root / "revisions" / "egr_zero" / "records"
    score_dir = output_root / "candidate_scores" / "records"
    for item_id, direction in pairs:
        sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
        sender = prebeliefs[(item_id, sender_id)]
        receiver = prebeliefs[(item_id, receiver_id)]
        record_path = records_dir / f"item_{item_id:04d}_{direction}.json"
        score_path = score_dir / f"item_{item_id:04d}_{direction}.json"
        if record_path.exists() and score_path.exists():
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                continue

        sender_answer = sender.get("parsed_answer")
        receiver_answer = receiver.get("parsed_answer")
        score_result: dict[str, Any]
        gate_open = False
        post_answer = receiver_answer
        reason = "answers_agree"
        if sender_answer is None or receiver_answer is None:
            reason = "unscorable_answer"
            score_result = {}
        elif sender_answer == receiver_answer:
            score_result = {}
        else:
            reason = "scored_disagreement"
            score_result = scorer.score_pair(
                question=str(receiver["question"]),
                receiver_reasoning=str(receiver["reasoning_text"]),
                evidence_packet=str(packets[(item_id, sender_id)]["evidence_packet"]),
                sender_candidate=canonical_option_text(metadata[item_id], sender_answer),
                receiver_candidate=canonical_option_text(metadata[item_id], receiver_answer),
            )
            gate_open = bool(score_result["gate_zero_open"])
            post_answer = sender_answer if gate_open else receiver_answer

        score_artifact = {
            "status": "complete",
            "benchmark": receiver.get("benchmark", "medqa300"),
            "replication_id": replication_id,
            "item_id": item_id,
            "direction": direction,
            "sender_answer": sender_answer,
            "receiver_answer": receiver_answer,
            "sender_correct": bool(sender["correct"]),
            "receiver_correct": bool(receiver["correct"]),
            "prior_sender_score": score_result.get("prior_sender"),
            "prior_receiver_score": score_result.get("prior_receiver"),
            "evidence_sender_score": score_result.get("evidence_sender"),
            "evidence_receiver_score": score_result.get("evidence_receiver"),
            "M0": score_result.get("M0"),
            "ME": score_result.get("ME"),
            "G": score_result.get("G"),
            "gate_zero_open": gate_open,
            "gate_tau_open": None,
            "egr_zero_answer": post_answer,
            "egr_tau_answer": None,
            "decision_reason": reason,
            "scoring_seconds": float(score_result.get("scoring_seconds", 0.0)),
        }
        atomic_write_json(score_path, score_artifact)
        seed = revision_seed(global_seed, item_id, direction, seed_replication_id)
        score_tokens = sum(
            int(value.get("context_tokens", 0)) + int(value.get("candidate_tokens", 0))
            for value in (
                score_result.get("prior_sender") or {},
                score_result.get("prior_receiver") or {},
                score_result.get("evidence_sender") or {},
                score_result.get("evidence_receiver") or {},
            )
        )
        record = base_record(
            sender=sender,
            receiver=receiver,
            condition="egr_zero",
            direction=direction,
            revision_seed_value=seed,
        )
        packet_text = str(packets[(item_id, sender_id)]["evidence_packet"])
        record.update(
            {
                "message_source_item_id": item_id,
                "message_source_agent_id": sender_id,
                "communication_payload": {
                    "modality": "egr_zero_claim_suppressed_evidence",
                    "payload_bytes": len(packet_text.encode("utf-8")),
                    "tokens": int(packets[(item_id, sender_id)]["evidence_token_count"]),
                },
                "response": "",
                "raw_response": "",
                "parsed_answer": post_answer,
                "generated_token_ids": [],
                "generation_length": 0,
                "hit_eos": False,
                "prompt_tokens": score_tokens,
                "prompt_sha256": sha256_json(
                    {
                        "prior": (score_result.get("prior_sender") or {}).get("context_sha256"),
                        "evidence": (score_result.get("evidence_sender") or {}).get("context_sha256"),
                    }
                ),
                "generation_seconds": float(score_result.get("scoring_seconds", 0.0)),
                "candidate_score_file": str(score_path.relative_to(output_root)),
                "gate_zero_open": gate_open,
            }
        )
        atomic_write_json(record_path, finalize_record(record, post_answer))
        print(
            f"[EGR egr_zero] item={item_id} {direction} sender={sender_answer} "
            f"receiver={receiver_answer} gate={gate_open} post={post_answer}",
            flush=True,
        )
        if stop_requested():
            break


def merge_outputs(output_root: Path, conditions: Iterable[str]) -> None:
    for condition in conditions:
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output_root / "revisions" / condition / "records").glob("*.json"))
        ]
        atomic_write_jsonl(output_root / "revisions" / f"{condition}.jsonl", rows)
    scores = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "candidate_scores" / "records").glob("*.json"))
    ]
    if scores:
        atomic_write_jsonl(output_root / "candidate_scores" / "merged.jsonl", scores)


def main() -> None:
    cli = parse_args()
    if cli.world_size < 1 or not 0 <= cli.rank < cli.world_size:
        raise ValueError("Invalid rank/world-size")
    source_config = json.loads((cli.source_root / "config.json").read_text(encoding="utf-8"))
    seed_replication_id = source_config.get("replication_id")
    replication_id = str(seed_replication_id or "seed_pair_00")
    global_seed = int(source_config.get("global_seed", 42))
    generation = source_config.get("generation") or source_config.get("generation_config") or {}
    max_new_tokens = cli.max_new_tokens or int(generation.get("max_new_tokens", 8192))
    prebelief_rows = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    prebeliefs = {
        (int(row["item_id"]), str(row["agent_id"])): row for row in prebelief_rows
    }
    available_ids = sorted({item_id for item_id, _ in prebeliefs})
    selected_ids = cli.item_ids or available_ids[: cli.limit]
    if not selected_ids or any(item_id not in available_ids for item_id in selected_ids):
        raise ValueError("Selected item IDs are unavailable in cached prebeliefs")
    task = str(source_config.get("dataset", "medqa"))
    benchmark_spec(task)
    metadata = benchmark_metadata(task, load_benchmark(task))
    config = ensure_config(
        output_root=cli.output_root,
        source_root=cli.source_root,
        source_config=source_config,
        selected_ids=selected_ids,
        model=cli.model,
        conditions=cli.conditions,
    )
    tokenizer = AutoTokenizer.from_pretrained(cli.model)
    packets, leakage = prepare_evidence_packets(
        prebeliefs=prebeliefs,
        metadata=metadata,
        selected_ids=selected_ids,
        output_root=cli.output_root,
        tokenizer=tokenizer,
    )
    print(f"[EGR leakage] {json.dumps(leakage, sort_keys=True)}", flush=True)
    if leakage["hard_stop_explicit_cue_leakage"]:
        raise RuntimeError("Explicit answer cues remain after filtering; stopping before inference")
    if cli.leakage_only:
        return

    assigned_pairs = [
        (item_id, direction)
        for item_id in selected_ids
        for direction in ("A_to_B", "B_to_A")
    ][cli.rank :: cli.world_size]
    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[EGR] stop requested signal={signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.cuda.set_device(cli.gpu)
    runtime = ICRRuntime(
        model_name=cli.model,
        device=torch.device(f"cuda:{cli.gpu}"),
        max_new_tokens=max_new_tokens,
        temperature=float(generation.get("temperature", 0.6)),
        top_p=float(generation.get("top_p", 0.95)),
        task=task,
    )
    started = time.time()
    for condition in cli.conditions:
        if condition in {"claim_only", "evidence_only"}:
            run_generation_condition(
                runtime=runtime,
                condition=condition,
                pairs=assigned_pairs,
                prebeliefs=prebeliefs,
                packets=packets,
                output_root=cli.output_root,
                global_seed=global_seed,
                replication_id=replication_id,
                seed_replication_id=seed_replication_id,
                stop_requested=lambda: stop,
            )
        elif condition == "egr_zero":
            run_egr_zero(
                scorer=EGRScorer(runtime),
                pairs=assigned_pairs,
                prebeliefs=prebeliefs,
                packets=packets,
                metadata=metadata,
                output_root=cli.output_root,
                global_seed=global_seed,
                replication_id=replication_id,
                seed_replication_id=seed_replication_id,
                stop_requested=lambda: stop,
            )
        if stop:
            break
    merge_outputs(cli.output_root, cli.conditions)
    atomic_write_json(
        cli.output_root / "run_status.json",
        {
            "status": "stopped" if stop else "complete",
            "config_fingerprint": config["fingerprint"],
            "rank": cli.rank,
            "world_size": cli.world_size,
            "assigned_directional_cases": len(assigned_pairs),
            "wall_clock_seconds": time.time() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
