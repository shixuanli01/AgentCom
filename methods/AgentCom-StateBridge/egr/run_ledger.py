"""Two-stage evidence-ledger EGR for disagreeing MedQA300 pairs."""

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

from .prompts import build_evidence_ledger_prompt, build_ledger_decision_prompt
from .run_contrast import generate_verdict, read_jsonl
from .run_m1 import base_record, finalize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-ledger EGR")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
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
    replication_id = str(next(iter(prebeliefs.values())).get("replication_id") or "seed_pair_00")
    stable = {
        "protocol": "EGR-LEDGER-v1",
        "status": "MedQA300 development diagnostic",
        "source_fingerprint": source_config.get("fingerprint"),
        "model": cli.model,
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "method": "claim-free evidence ledger followed by independent decision",
        "calls_per_disagreement": 2,
        "threshold_tuned": False,
    }
    config = {**stable, "fingerprint": sha256_json(stable)}
    config_path = cli.output_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise RuntimeError("Ledger output root has a different configuration")
    else:
        atomic_write_json(config_path, config)

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[EGR ledger] stop requested signal={signum}", flush=True)

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
    ledger_dir = cli.output_root / "ledgers" / "records"
    verdict_dir = cli.output_root / "verdicts" / "records"
    revision_dir = cli.output_root / "revisions" / "egr_ledger" / "records"
    calls = 0
    for item_id in range(300):
        agent_a, agent_b = prebeliefs[(item_id, "A")], prebeliefs[(item_id, "B")]
        answer_a, answer_b = agent_a.get("parsed_answer"), agent_b.get("parsed_answer")
        ledger_result: dict[str, Any] = {}
        decision_result: dict[str, Any] = {}
        reason = "answers_agree"
        verdict = answer_a if answer_a == answer_b else None
        ledger_path = ledger_dir / f"item_{item_id:04d}.json"
        verdict_path = verdict_dir / f"item_{item_id:04d}.json"
        if answer_a is None or answer_b is None:
            reason = "unscorable_answer"
            verdict = answer_a if answer_a is not None else answer_b
        elif answer_a != answer_b:
            reason = "evidence_ledger"
            calls += 2
            if ledger_path.exists():
                cached = json.loads(ledger_path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    ledger_result = cached["generation"]
            if not ledger_result:
                prompt = build_evidence_ledger_prompt(
                    str(agent_a["question"]),
                    str(packets[(item_id, "A")]["evidence_packet"]),
                    str(packets[(item_id, "B")]["evidence_packet"]),
                )
                ledger_result = generate_verdict(
                    runtime, prompt, stable_seed(global_seed, item_id, "egr_ledger_facts_v1")
                )
                atomic_write_json(
                    ledger_path,
                    {
                        "status": "complete",
                        "benchmark": "medqa300",
                        "replication_id": replication_id,
                        "item_id": item_id,
                        "generation": ledger_result,
                    },
                )
            if verdict_path.exists():
                cached = json.loads(verdict_path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    decision_result = cached["generation"]
            if not decision_result:
                prompt = build_ledger_decision_prompt(
                    str(agent_a["question"]), str(ledger_result.get("response") or "")
                )
                decision_result = generate_verdict(
                    runtime, prompt, stable_seed(global_seed, item_id, "egr_ledger_decision_v1")
                )
            verdict = decision_result.get("parsed_answer")
        if verdict is None:
            verdict = answer_a

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
                "verdict": verdict,
                "ledger_file": str(ledger_path.relative_to(cli.output_root)) if ledger_result else None,
                "generation": decision_result,
            },
        )
        payload_bytes = sum(
            len(str(packets[(item_id, agent)]["evidence_packet"]).encode("utf-8"))
            for agent in ("A", "B")
        )
        for direction in ("A_to_B", "B_to_A"):
            sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
            sender, receiver = prebeliefs[(item_id, sender_id)], prebeliefs[(item_id, receiver_id)]
            post = verdict if verdict is not None else receiver.get("parsed_answer")
            record = base_record(
                sender=sender,
                receiver=receiver,
                condition="egr_ledger",
                direction=direction,
                revision_seed_value=stable_seed(global_seed, item_id, "egr_ledger_decision_v1"),
            )
            record.update(
                {
                    "message_source_item_id": item_id,
                    "message_source_agent_id": sender_id,
                    "communication_payload": {
                        "modality": "symmetric_claim_free_evidence_ledger",
                        "payload_bytes": payload_bytes,
                        "tokens": sum(
                            int(packets[(item_id, agent)]["evidence_token_count"])
                            for agent in ("A", "B")
                        ),
                    },
                    "response": decision_result.get("response", ""),
                    "raw_response": decision_result.get("raw_response", ""),
                    "parsed_answer": post,
                    "generated_token_ids": decision_result.get("generated_token_ids", []),
                    "generation_length": decision_result.get("generation_length", 0),
                    "hit_eos": decision_result.get("hit_eos", False),
                    "prompt_tokens": decision_result.get("prompt_tokens", 0),
                    "prompt_sha256": decision_result.get("prompt_sha256", ""),
                    "generation_seconds": (
                        float(ledger_result.get("generation_seconds", 0.0))
                        + float(decision_result.get("generation_seconds", 0.0))
                    ) / 2.0,
                    "ledger_file": str(ledger_path.relative_to(cli.output_root)) if ledger_result else None,
                    "ledger_sha256": ledger_result.get("prompt_sha256", ""),
                    "verdict_file": str(verdict_path.relative_to(cli.output_root)),
                }
            )
            atomic_write_json(
                revision_dir / f"item_{item_id:04d}_{direction}.json",
                finalize_record(record, post),
            )
        print(
            f"[EGR ledger] item={item_id} A={answer_a} B={answer_b} final={verdict} reason={reason}",
            flush=True,
        )
        if stop:
            break

    ledger_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(ledger_dir.glob("*.json"))
    ]
    verdict_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(verdict_dir.glob("*.json"))
    ]
    revision_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(revision_dir.glob("*.json"))
    ]
    atomic_write_jsonl(cli.output_root / "ledgers" / "merged.jsonl", ledger_rows)
    atomic_write_jsonl(cli.output_root / "verdicts" / "merged.jsonl", verdict_rows)
    atomic_write_jsonl(cli.output_root / "revisions" / "egr_ledger.jsonl", revision_rows)
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
