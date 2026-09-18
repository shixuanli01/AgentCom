"""One-call symmetric evidence adjudication for disagreeing MCQ pairs."""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from icr.protocol import (
    atomic_write_json,
    atomic_write_jsonl,
    parse_task_answer,
    reset_rng,
    sha256_json,
    sha256_text,
    stable_seed,
)
from icr.runtime import ICRRuntime
from methods.state_bridge import strip_thinking

from .prompts import build_contrast_adjudication_prompt
from .run_m1 import base_record, finalize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGR contrast adjudication")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@torch.no_grad()
def generate_verdict(runtime: ICRRuntime, user_prompt: str, seed: int) -> dict[str, Any]:
    reset_rng(seed)
    prompt = runtime._render(user_prompt)
    input_ids, attention_mask, _, clean_prompt = runtime._encode_prompt(prompt, None)
    embeddings = runtime.embedding_layer(input_ids)
    started = time.time()
    generated = runtime.model.model.generate(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        max_new_tokens=runtime.max_new_tokens,
        temperature=runtime.temperature,
        top_p=runtime.top_p,
        do_sample=True,
        pad_token_id=runtime.model.tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    torch.cuda.synchronize(runtime.device)
    generated_ids = generated.sequences
    raw = runtime.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    response = strip_thinking(raw)
    last = int(generated_ids[0, -1]) if generated_ids.shape[1] else None
    result = {
        "response": response,
        "raw_response": raw,
        "parsed_answer": parse_task_answer(runtime.task, response),
        "generation_seed": seed,
        "generated_token_ids": generated_ids[0].detach().cpu().tolist(),
        "generation_length": int(generated_ids.shape[1]),
        "hit_eos": last == runtime.model.tokenizer.eos_token_id,
        "prompt_tokens": int(attention_mask.sum()),
        "prompt_sha256": sha256_text(clean_prompt),
        "generation_seconds": time.time() - started,
    }
    del input_ids, attention_mask, embeddings, generated
    torch.cuda.empty_cache()
    return result


def main() -> None:
    cli = parse_args()
    source_config = json.loads((cli.source_root / "config.json").read_text(encoding="utf-8"))
    replication_id = str(source_config.get("replication_id") or "seed_pair_00")
    task = str(source_config.get("dataset", "medqa"))
    benchmark = str(source_config.get("benchmark", "medqa300"))
    selected_ids = [int(value) for value in source_config["selected_item_ids"]]
    global_seed = int(source_config.get("global_seed", 42))
    generation = source_config.get("generation") or source_config.get("generation_config") or {}
    beliefs = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    prebeliefs = {(int(row["item_id"]), str(row["agent_id"])): row for row in beliefs}
    packets = {
        (int(row["item_id"]), str(row["agent_id"])): row
        for row in read_jsonl(cli.m1_root / "evidence_packets" / "merged.jsonl")
    }
    stable = {
        "protocol": "EGR-CONTRAST-v1",
        "status": f"{benchmark} cross-benchmark evaluation",
        "source_fingerprint": source_config.get("fingerprint"),
        "m1_root": str(cli.m1_root.resolve()),
        "model": cli.model,
        "task": task,
        "benchmark": benchmark,
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "adjudication": "one anonymous contrastive generation per parsed disagreement item",
        "threshold_tuned": False,
    }
    config = {**stable, "fingerprint": sha256_json(stable)}
    config_path = cli.output_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise RuntimeError("Contrast output root has a different configuration")
    else:
        atomic_write_json(config_path, config)

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"[EGR contrast] stop requested signal={signum}", flush=True)

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
    revision_dir = cli.output_root / "revisions" / "egr_contrast" / "records"
    calls = 0
    for item_id in selected_ids:
        agent_a = prebeliefs[(item_id, "A")]
        agent_b = prebeliefs[(item_id, "B")]
        answer_a = agent_a.get("parsed_answer")
        answer_b = agent_b.get("parsed_answer")
        verdict_path = verdict_dir / f"item_{item_id:04d}.json"
        result: dict[str, Any] = {}
        reason = "answers_agree"
        verdict = answer_a if answer_a == answer_b else None
        if answer_a is None or answer_b is None:
            reason = "unscorable_answer"
            verdict = answer_a if answer_a is not None else answer_b
        elif answer_a != answer_b:
            reason = "contrast_adjudication"
            calls += 1
            if verdict_path.exists():
                cached = json.loads(verdict_path.read_text(encoding="utf-8"))
                if cached.get("status") == "complete":
                    result = cached["generation"]
            if not result:
                prompt = build_contrast_adjudication_prompt(
                    str(agent_a["question"]),
                    str(packets[(item_id, "A")]["evidence_packet"]),
                    str(packets[(item_id, "B")]["evidence_packet"]),
                    task=task,
                )
                result = generate_verdict(
                    runtime, prompt, stable_seed(global_seed, item_id, "egr_contrast_v1")
                )
            verdict = result.get("parsed_answer")
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
                "verdict": verdict,
                "generation": result,
            },
        )
        for direction in ("A_to_B", "B_to_A"):
            sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
            sender = prebeliefs[(item_id, sender_id)]
            receiver = prebeliefs[(item_id, receiver_id)]
            post = verdict if verdict is not None else receiver.get("parsed_answer")
            record = base_record(
                sender=sender,
                receiver=receiver,
                condition="egr_contrast",
                direction=direction,
                revision_seed_value=stable_seed(global_seed, item_id, "egr_contrast_v1"),
            )
            record.update(
                {
                    "message_source_item_id": item_id,
                    "message_source_agent_id": sender_id,
                    "communication_payload": {
                        "modality": "symmetric_contrastive_evidence",
                        "payload_bytes": sum(
                            len(str(packets[(item_id, agent)]["evidence_packet"]).encode("utf-8"))
                            for agent in ("A", "B")
                        ),
                        "tokens": sum(int(packets[(item_id, agent)]["evidence_token_count"]) for agent in ("A", "B")),
                    },
                    **({
                        "response": result.get("response", ""),
                        "raw_response": result.get("raw_response", ""),
                        "parsed_answer": post,
                        "generated_token_ids": result.get("generated_token_ids", []),
                        "generation_length": result.get("generation_length", 0),
                        "hit_eos": result.get("hit_eos", False),
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "prompt_sha256": result.get("prompt_sha256", ""),
                        "generation_seconds": float(result.get("generation_seconds", 0.0)) / 2.0,
                    }),
                    "contrast_verdict_file": str(verdict_path.relative_to(cli.output_root)),
                }
            )
            atomic_write_json(
                revision_dir / f"item_{item_id:04d}_{direction}.json",
                finalize_record(record, post),
            )
        print(
            f"[EGR contrast] item={item_id} A={answer_a} B={answer_b} verdict={verdict} reason={reason}",
            flush=True,
        )
        if stop:
            break

    verdict_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(verdict_dir.glob("*.json"))]
    revision_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(revision_dir.glob("*.json"))]
    atomic_write_jsonl(cli.output_root / "verdicts" / "merged.jsonl", verdict_rows)
    atomic_write_jsonl(cli.output_root / "revisions" / "egr_contrast.jsonl", revision_rows)
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
