"""Order-invariance gate over cached EGR contrast adjudications."""

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

from .prompts import build_contrast_adjudication_prompt
from .run_contrast import generate_verdict, read_jsonl
from .run_m1 import base_record, finalize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGR permutation-invariance gate")
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
    task = str(source_config.get("dataset", "medqa"))
    benchmark = str(source_config.get("benchmark", "medqa300"))
    selected_ids = [int(value) for value in source_config["selected_item_ids"]]
    generation = source_config.get("generation") or source_config.get("generation_config") or {}
    beliefs = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    prebeliefs = {(int(row["item_id"]), str(row["agent_id"])): row for row in beliefs}
    packets = {
        (int(row["item_id"]), str(row["agent_id"])): row
        for row in read_jsonl(cli.m1_root / "evidence_packets" / "merged.jsonl")
    }
    original = {
        int(row["item_id"]): row
        for row in read_jsonl(cli.contrast_root / "verdicts" / "merged.jsonl")
    }
    replication_ids = {str(row["replication_id"]) for row in original.values()}
    if set(original) != set(selected_ids) or len(replication_ids) != 1:
        raise RuntimeError("Expected one complete contrast replication for selected IDs")
    replication_id = replication_ids.pop()
    stable = {
        "protocol": "EGR-PERMUTATION-GATE-v1",
        "status": f"{benchmark} cross-benchmark evaluation",
        "source_fingerprint": source_config.get("fingerprint"),
        "contrast_fingerprint": json.loads(
            (cli.contrast_root / "config.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "model": cli.model,
        "task": task,
        "benchmark": benchmark,
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "gate": "adopt only when AB and BA evidence orders yield the same verdict",
        "fallback": "directional receiver prior",
        "threshold_tuned": False,
    }
    config = {**stable, "fingerprint": sha256_json(stable)}
    config_path = cli.output_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise RuntimeError("Permutation output root has a different configuration")
    else:
        atomic_write_json(config_path, config)

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[EGR permutation] stop requested signal={signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.cuda.set_device(cli.gpu)
    runtime = ICRRuntime(
        model_name=cli.model,
        device=torch.device(f"cuda:{cli.gpu}"),
        max_new_tokens=int(generation.get("max_new_tokens", 8192)),
        temperature=float(generation.get("temperature", 0.6)),
        top_p=float(generation.get("top_p", 0.95)),
        task=task,
    )
    started = time.time()
    verdict_dir = cli.output_root / "verdicts" / "records"
    revision_dir = cli.output_root / "revisions" / "egr_permutation_gate" / "records"
    calls = stable_items = unstable_items = 0
    for item_id in selected_ids:
        agent_a, agent_b = prebeliefs[(item_id, "A")], prebeliefs[(item_id, "B")]
        answer_a, answer_b = agent_a.get("parsed_answer"), agent_b.get("parsed_answer")
        packet_a, packet_b = packets[(item_id, "A")], packets[(item_id, "B")]
        original_verdict = original[item_id].get("verdict")
        verdict_path = verdict_dir / f"item_{item_id:04d}.json"
        swapped_result: dict[str, Any] = {}
        swapped_verdict = original_verdict
        stable_order = True
        reason = "answers_agree"
        if answer_a is None or answer_b is None:
            reason = "unscorable_answer"
        elif answer_a != answer_b:
            reason = "permutation_check"
            calls += 1
            if verdict_path.exists():
                cached = json.loads(verdict_path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    swapped_result = cached["swapped_generation"]
            if not swapped_result:
                prompt = build_contrast_adjudication_prompt(
                    str(agent_a["question"]),
                    str(packet_b["evidence_packet"]),
                    str(packet_a["evidence_packet"]),
                    task=task,
                )
                # Reuse the original seed so only evidence order changes.
                swapped_result = generate_verdict(
                    runtime, prompt, stable_seed(global_seed, item_id, "egr_contrast_v1")
                )
            swapped_verdict = swapped_result.get("parsed_answer")
            stable_order = original_verdict is not None and original_verdict == swapped_verdict
            stable_items += int(stable_order)
            unstable_items += int(not stable_order)

        atomic_write_json(
            verdict_path,
            {
                "status": "complete",
                "benchmark": benchmark,
                "replication_id": replication_id,
                "item_id": item_id,
                "answer_A": answer_a,
                "answer_B": answer_b,
                "correct_A": bool(agent_a["correct"]),
                "correct_B": bool(agent_b["correct"]),
                "decision_reason": reason,
                "original_verdict": original_verdict,
                "swapped_verdict": swapped_verdict,
                "order_stable": stable_order,
                "swapped_generation": swapped_result,
            },
        )
        for direction in ("A_to_B", "B_to_A"):
            sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
            sender, receiver = prebeliefs[(item_id, sender_id)], prebeliefs[(item_id, receiver_id)]
            post = original_verdict if stable_order else receiver.get("parsed_answer")
            if post is None:
                post = receiver.get("parsed_answer")
            record = base_record(
                sender=sender,
                receiver=receiver,
                condition="egr_permutation_gate",
                direction=direction,
                revision_seed_value=stable_seed(global_seed, item_id, "egr_contrast_v1"),
            )
            record.update(
                {
                    "message_source_item_id": item_id,
                    "message_source_agent_id": sender_id,
                    "communication_payload": {
                        "modality": "symmetric_evidence_permutation_check",
                        "payload_bytes": sum(
                            len(str(packet["evidence_packet"]).encode("utf-8"))
                            for packet in (packet_a, packet_b)
                        ),
                        "tokens": int(packet_a["evidence_token_count"])
                        + int(packet_b["evidence_token_count"]),
                    },
                    "response": swapped_result.get("response", ""),
                    "raw_response": swapped_result.get("raw_response", ""),
                    "parsed_answer": post,
                    "generated_token_ids": swapped_result.get("generated_token_ids", []),
                    "generation_length": swapped_result.get("generation_length", 0),
                    "hit_eos": swapped_result.get("hit_eos", False),
                    "prompt_tokens": swapped_result.get("prompt_tokens", 0),
                    "prompt_sha256": swapped_result.get("prompt_sha256", ""),
                    "generation_seconds": float(swapped_result.get("generation_seconds", 0.0)) / 2.0,
                    "original_verdict": original_verdict,
                    "swapped_verdict": swapped_verdict,
                    "order_stable": stable_order,
                    "permutation_verdict_file": str(verdict_path.relative_to(cli.output_root)),
                }
            )
            atomic_write_json(
                revision_dir / f"item_{item_id:04d}_{direction}.json",
                finalize_record(record, post),
            )
        print(
            f"[EGR permutation] item={item_id} original={original_verdict} "
            f"swapped={swapped_verdict} stable={stable_order} reason={reason}",
            flush=True,
        )
        if stop:
            break

    verdict_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(verdict_dir.glob("*.json"))
    ]
    revision_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(revision_dir.glob("*.json"))
    ]
    atomic_write_jsonl(cli.output_root / "verdicts" / "merged.jsonl", verdict_rows)
    atomic_write_jsonl(cli.output_root / "revisions" / "egr_permutation_gate.jsonl", revision_rows)
    atomic_write_json(
        cli.output_root / "run_status.json",
        {
            "status": "stopped" if stop else "complete",
            "items": len(verdict_rows),
            "directional_cases": len(revision_rows),
            "generative_calls": calls,
            "order_stable_disagreements": stable_items,
            "order_unstable_disagreements": unstable_items,
            "wall_clock_seconds": time.time() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
