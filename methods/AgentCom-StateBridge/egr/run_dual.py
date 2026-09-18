"""Run training-free symmetric evidence-duel EGR on frozen MedQA300 beliefs."""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from icr.protocol import atomic_write_json, atomic_write_jsonl, revision_seed, sha256_json
from icr.runtime import ICRRuntime

from .run_m1 import base_record, canonical_option_text, finalize_record, medqa_metadata
from .scoring import EGRScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symmetric EGR evidence duel")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    cli = parse_args()
    source_config = json.loads((cli.source_root / "config.json").read_text(encoding="utf-8"))
    seed_replication_id = source_config.get("replication_id")
    replication_id = str(seed_replication_id or "seed_pair_00")
    global_seed = int(source_config.get("global_seed", 42))
    generation = source_config.get("generation") or source_config.get("generation_config") or {}
    beliefs = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    prebeliefs = {(int(row["item_id"]), str(row["agent_id"])): row for row in beliefs}
    packets = {
        (int(row["item_id"]), str(row["agent_id"])): row
        for row in read_jsonl(cli.m1_root / "evidence_packets" / "merged.jsonl")
    }
    metadata = medqa_metadata(Path(__file__).resolve().parents[1] / "data" / "medqa.json")
    stable = {
        "protocol": "EGR-DUAL-v1",
        "status": "MedQA300 development diagnostic",
        "source_fingerprint": source_config.get("fingerprint"),
        "m1_root": str(cli.m1_root.resolve()),
        "model": cli.model,
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "decision": "winner=argmax min(evidence_margin,evidence_gain), positive only; else receiver",
        "threshold_tuned": False,
    }
    config = {**stable, "fingerprint": sha256_json(stable)}
    config_path = cli.output_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise RuntimeError("Dual output root has a different configuration")
    else:
        atomic_write_json(config_path, config)

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[EGR dual] stop requested signal={signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.cuda.set_device(cli.gpu)
    runtime = ICRRuntime(
        model_name=cli.model,
        device=torch.device(f"cuda:{cli.gpu}"),
        max_new_tokens=int(generation.get("max_new_tokens", 8192)),
        temperature=float(generation.get("temperature", 0.6)),
        top_p=float(generation.get("top_p", 0.95)),
    )
    scorer = EGRScorer(runtime)
    started = time.time()
    score_dir = cli.output_root / "dual_scores" / "records"
    revision_dir = cli.output_root / "revisions" / "egr_dual" / "records"
    for item_id in range(300):
        agent_a = prebeliefs[(item_id, "A")]
        agent_b = prebeliefs[(item_id, "B")]
        answer_a = agent_a.get("parsed_answer")
        answer_b = agent_b.get("parsed_answer")
        score_path = score_dir / f"item_{item_id:04d}.json"
        result: dict[str, Any] = {}
        reason = "answers_agree"
        winner = None
        if answer_a is None or answer_b is None:
            reason = "unscorable_answer"
        elif answer_a != answer_b:
            reason = "scored_disagreement"
            if score_path.exists():
                cached = json.loads(score_path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    result = cached["scores"]
                    winner = result["winner"]
            if not result:
                result = scorer.score_evidence_duel(
                    question=str(agent_a["question"]),
                    evidence_a=str(packets[(item_id, "A")]["evidence_packet"]),
                    evidence_b=str(packets[(item_id, "B")]["evidence_packet"]),
                    candidate_a=canonical_option_text(metadata[item_id], answer_a),
                    candidate_b=canonical_option_text(metadata[item_id], answer_b),
                )
                winner = result["winner"]
        score_artifact = {
            "status": "complete",
            "benchmark": "medqa300",
            "replication_id": replication_id,
            "item_id": item_id,
            "answer_A": answer_a,
            "answer_B": answer_b,
            "correct_A": bool(agent_a["correct"]),
            "correct_B": bool(agent_b["correct"]),
            "decision_reason": reason,
            "scores": result,
            "winner": winner,
        }
        atomic_write_json(score_path, score_artifact)

        for direction in ("A_to_B", "B_to_A"):
            sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
            sender = prebeliefs[(item_id, sender_id)]
            receiver = prebeliefs[(item_id, receiver_id)]
            record_path = revision_dir / f"item_{item_id:04d}_{direction}.json"
            winner_answer = answer_a if winner == "A" else answer_b if winner == "B" else None
            post = winner_answer if winner_answer is not None else receiver.get("parsed_answer")
            seed = revision_seed(global_seed, item_id, direction, seed_replication_id)
            record = base_record(
                sender=sender,
                receiver=receiver,
                condition="egr_dual",
                direction=direction,
                revision_seed_value=seed,
            )
            record.update(
                {
                    "message_source_item_id": item_id,
                    "message_source_agent_id": sender_id,
                    "communication_payload": {
                        "modality": "symmetric_claim_suppressed_evidence_duel",
                        "payload_bytes": len(str(packets[(item_id, sender_id)]["evidence_packet"]).encode("utf-8")),
                        "tokens": int(packets[(item_id, sender_id)]["evidence_token_count"]),
                    },
                    "response": "",
                    "raw_response": "",
                    "parsed_answer": post,
                    "generated_token_ids": [],
                    "generation_length": 0,
                    "hit_eos": False,
                    "prompt_tokens": sum(
                        int(value.get("context_tokens", 0)) + int(value.get("candidate_tokens", 0))
                        for value in result.values() if isinstance(value, dict)
                    ),
                    "prompt_sha256": sha256_json({"score_file": str(score_path)}),
                    "generation_seconds": float(result.get("scoring_seconds", 0.0)) / 2.0,
                    "dual_score_file": str(score_path.relative_to(cli.output_root)),
                    "dual_winner": winner,
                }
            )
            atomic_write_json(record_path, finalize_record(record, post))
        print(
            f"[EGR dual] item={item_id} A={answer_a} B={answer_b} winner={winner} "
            f"S_A={result.get('S_A')} S_B={result.get('S_B')}",
            flush=True,
        )
        if stop:
            break

    score_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(score_dir.glob("*.json"))]
    revision_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(revision_dir.glob("*.json"))]
    atomic_write_jsonl(cli.output_root / "dual_scores" / "merged.jsonl", score_rows)
    atomic_write_jsonl(cli.output_root / "revisions" / "egr_dual.jsonl", revision_rows)
    atomic_write_json(
        cli.output_root / "run_status.json",
        {
            "status": "stopped" if stop else "complete",
            "items": len(score_rows),
            "directional_cases": len(revision_rows),
            "wall_clock_seconds": time.time() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
