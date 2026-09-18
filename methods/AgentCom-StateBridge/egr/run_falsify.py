"""Falsification pass over cached EGR contrast adjudications."""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from icr.protocol import atomic_write_json, atomic_write_jsonl, sha256_json, stable_seed
from icr.runtime import ICRRuntime

from .prompts import build_falsification_prompt
from .run_contrast import generate_verdict, read_jsonl
from .run_m1 import base_record, finalize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGR falsification verification")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--contrast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source_config = json.loads((cli.source_root / "config.json").read_text(encoding="utf-8"))
    global_seed = int(source_config.get("global_seed", 42))
    generation = source_config.get("generation") or source_config.get("generation_config") or {}
    beliefs = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    prebeliefs = {(int(row["item_id"]), str(row["agent_id"])): row for row in beliefs}
    packets = {
        (int(row["item_id"]), str(row["agent_id"])): row
        for row in read_jsonl(cli.m1_root / "evidence_packets" / "merged.jsonl")
    }
    contrast = {
        int(row["item_id"]): row
        for row in read_jsonl(cli.contrast_root / "verdicts" / "merged.jsonl")
    }
    replication_ids = {str(row["replication_id"]) for row in contrast.values()}
    if len(replication_ids) != 1 or len(contrast) != 300:
        raise RuntimeError("Expected one complete 300-item contrast replication")
    replication_id = replication_ids.pop()

    stable = {
        "protocol": "EGR-FALSIFY-v1",
        "status": "MedQA300 development diagnostic",
        "source_fingerprint": source_config.get("fingerprint"),
        "contrast_fingerprint": json.loads(
            (cli.contrast_root / "config.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "model": cli.model,
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "verification": "one falsification generation per parsed disagreement item",
        "threshold_tuned": False,
    }
    config = {**stable, "fingerprint": sha256_json(stable)}
    config_path = cli.output_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise RuntimeError("Falsification output root has a different configuration")
    else:
        atomic_write_json(config_path, config)

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[EGR falsify] stop requested signal={signum}", flush=True)

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
    started = time.time()
    verdict_dir = cli.output_root / "verdicts" / "records"
    revision_dir = cli.output_root / "revisions" / "egr_falsify" / "records"
    calls = 0
    for item_id in range(300):
        agent_a = prebeliefs[(item_id, "A")]
        agent_b = prebeliefs[(item_id, "B")]
        answer_a, answer_b = agent_a.get("parsed_answer"), agent_b.get("parsed_answer")
        preliminary = contrast[item_id]
        verdict_path = verdict_dir / f"item_{item_id:04d}.json"
        result: dict[str, Any] = {}
        reason = "answers_agree"
        verdict = answer_a if answer_a == answer_b else None
        if answer_a is None or answer_b is None:
            reason = "unscorable_answer"
            verdict = preliminary.get("verdict")
        elif answer_a != answer_b:
            reason = "falsification_verification"
            calls += 1
            if verdict_path.exists():
                cached = json.loads(verdict_path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    result = cached["generation"]
            if not result:
                preliminary_text = str(
                    (preliminary.get("generation") or {}).get("response") or ""
                )
                prompt = build_falsification_prompt(
                    str(agent_a["question"]),
                    str(packets[(item_id, "A")]["evidence_packet"]),
                    str(packets[(item_id, "B")]["evidence_packet"]),
                    preliminary_text,
                )
                result = generate_verdict(
                    runtime, prompt, stable_seed(global_seed, item_id, "egr_falsify_v1")
                )
            verdict = result.get("parsed_answer")
        if verdict is None:
            verdict = preliminary.get("verdict")

        atomic_write_json(
            verdict_path,
            {
                "status": "complete",
                "benchmark": "medqa300",
                "replication_id": replication_id,
                "item_id": item_id,
                "answer_A": answer_a,
                "answer_B": answer_b,
                "correct_A": bool(agent_a["correct"]),
                "correct_B": bool(agent_b["correct"]),
                "decision_reason": reason,
                "preliminary_verdict": preliminary.get("verdict"),
                "verdict": verdict,
                "generation": result,
            },
        )
        payload_bytes = sum(
            len(str(packets[(item_id, agent)]["evidence_packet"]).encode("utf-8"))
            for agent in ("A", "B")
        ) + len(str((preliminary.get("generation") or {}).get("response") or "").encode("utf-8"))
        for direction in ("A_to_B", "B_to_A"):
            sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
            sender, receiver = prebeliefs[(item_id, sender_id)], prebeliefs[(item_id, receiver_id)]
            post = verdict if verdict is not None else receiver.get("parsed_answer")
            record = base_record(
                sender=sender,
                receiver=receiver,
                condition="egr_falsify",
                direction=direction,
                revision_seed_value=stable_seed(global_seed, item_id, "egr_falsify_v1"),
            )
            record.update(
                {
                    "message_source_item_id": item_id,
                    "message_source_agent_id": sender_id,
                    "communication_payload": {
                        "modality": "symmetric_evidence_plus_falsification",
                        "payload_bytes": payload_bytes,
                        "tokens": sum(
                            int(packets[(item_id, agent)]["evidence_token_count"])
                            for agent in ("A", "B")
                        ),
                    },
                    "response": result.get("response", ""),
                    "raw_response": result.get("raw_response", ""),
                    "parsed_answer": post,
                    "generated_token_ids": result.get("generated_token_ids", []),
                    "generation_length": result.get("generation_length", 0),
                    "hit_eos": result.get("hit_eos", False),
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "prompt_sha256": result.get("prompt_sha256", ""),
                    "generation_seconds": float(result.get("generation_seconds", 0.0)) / 2.0,
                    "preliminary_verdict": preliminary.get("verdict"),
                    "falsification_verdict_file": str(verdict_path.relative_to(cli.output_root)),
                }
            )
            atomic_write_json(
                revision_dir / f"item_{item_id:04d}_{direction}.json",
                finalize_record(record, post),
            )
        print(
            f"[EGR falsify] item={item_id} A={answer_a} B={answer_b} "
            f"initial={preliminary.get('verdict')} final={verdict} reason={reason}",
            flush=True,
        )
        if stop:
            break

    verdict_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(verdict_dir.glob("*.json"))
    ]
    revision_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(revision_dir.glob("*.json"))
    ]
    atomic_write_jsonl(cli.output_root / "verdicts" / "merged.jsonl", verdict_rows)
    atomic_write_jsonl(cli.output_root / "revisions" / "egr_falsify.jsonl", revision_rows)
    atomic_write_json(
        cli.output_root / "run_status.json",
        {
            "status": "stopped" if stop else "complete",
            "items": len(verdict_rows),
            "directional_cases": len(revision_rows),
            "generative_calls": calls,
            "wall_clock_seconds": time.time() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
