"""Sharded Phase-1 independent-belief generation and StateBridge caching."""

from __future__ import annotations

import argparse
import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from . import CONDITIONS, PROTOCOL_V3
from .benchmarks import (
    SPECS,
    benchmark_spec,
    load_benchmark,
    select_item_ids,
    structural_exclusions,
)
from .prompts_v3 import PROMPT_VERSION
from .protocol import (
    atomic_write_json,
    atomic_write_jsonl,
    answer_is_correct,
    canonical_gold,
    prebelief_seed,
    sha256_json,
)
from .runtime import ICRRuntime, atomic_save_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{PROTOCOL_V3} phase 1")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--task", choices=sorted(SPECS), default="medqa")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--replication-id")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--item-ids", nargs="+", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    return parser.parse_args()


def rank_and_world(cli: argparse.Namespace) -> tuple[int, int]:
    rank = cli.rank if cli.rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world = cli.world_size if cli.world_size is not None else int(os.getenv("WORLD_SIZE", "1"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid rank/world-size topology")
    return rank, world


def build_config(
    cli: argparse.Namespace,
    selected_ids: list[int],
    data: list[dict],
    excluded_ids: list[int],
) -> dict[str, Any]:
    stable = {
        "protocol": PROTOCOL_V3,
        "dataset": cli.task,
        "benchmark": benchmark_spec(cli.task).label,
        "dataset_rows": len(data),
        "selected_item_ids": selected_ids,
        "excluded_item_ids": excluded_ids,
        "exclusion_rule": (
            "arc_challenge: trailing option count != 4 (label-free, pre-registered)"
            if excluded_ids
            else None
        ),
        "dataset_sha256": sha256_json(data),
        "selection": {
            "mode": (
                "explicit_ids" if cli.item_ids else
                "seeded_sample" if cli.sample_size is not None else
                "prefix" if cli.limit is not None else "full"
            ),
            "sample_size": cli.sample_size,
            "selection_seed": cli.selection_seed if cli.sample_size is not None else None,
        },
        "model": cli.model,
        "global_seed": cli.global_seed,
        "generation": {
            "do_sample": True,
            "temperature": cli.temperature,
            "top_p": cli.top_p,
            "top_k": None,
            "max_new_tokens": cli.max_new_tokens,
        },
        "statebridge": {
            "selection_method": "last_k",
            "max_prefix_tokens": 64,
            "adaptive_reg": 0.001,
            "snap_ratio": 0.3,
            "prefix_strategy": "scale",
            "prefix_scale": 1.0,
            "use_hook": True,
            "algorithm_source": "methods/state_bridge.py (unmodified)",
            "receiver_injection_position": "message_slot_after_receiver_prior",
        },
        "revision_conditions": list(CONDITIONS),
        "revision_prompt_version": PROMPT_VERSION,
        "answer_parser_version": "icr.parsing_v3@ICR-V3",
        "other_mapping_offset": 137,
    }
    if cli.replication_id is not None:
        stable["replication_id"] = cli.replication_id
        stable["statebridge"]["message_directory"] = "messages/statebridge"
    return {**stable, "fingerprint": sha256_json(stable)}


def ensure_config(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != candidate["fingerprint"]:
            raise RuntimeError("Artifact root contains a different ICR configuration")
        return existing
    atomic_write_json(path, candidate)
    return candidate


def main() -> None:
    cli = parse_args()
    if cli.max_new_tokens is None:
        cli.max_new_tokens = benchmark_spec(cli.task).default_max_new_tokens
    rank, world = rank_and_world(cli)
    data = load_benchmark(cli.task)
    selected_ids = select_item_ids(
        len(data),
        item_ids=cli.item_ids,
        limit=cli.limit,
        sample_size=cli.sample_size,
        selection_seed=cli.selection_seed,
    )
    excluded_ids = structural_exclusions(cli.task, data)
    if excluded_ids:
        removed = sorted(set(selected_ids) & set(excluded_ids))
        selected_ids = [value for value in selected_ids if value not in set(excluded_ids)]
        print(
            f"[ICR prebeliefs] structural exclusion removed {len(removed)} items: {removed}",
            flush=True,
        )
    config = ensure_config(
        cli.artifact_root / "config.json",
        build_config(cli, selected_ids, data, excluded_ids),
    )
    assigned_ids = selected_ids[rank::world]
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[ICR prebeliefs rank={rank}] stop requested signal={signum}", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Shard rank and local CUDA index are deliberately decoupled so a formal
    # run can place multiple independent workers on each physical GPU.
    device_index = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_index)
    runtime = ICRRuntime(
        model_name=cli.model,
        device=torch.device(f"cuda:{device_index}"),
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        top_p=cli.top_p,
        task=cli.task,
    )
    records_dir = cli.artifact_root / "prebeliefs" / f"rank{rank}" / "records"
    message_directory = str(
        config["statebridge"].get("message_directory", "statebridge_messages")
    )
    prefix_dir = cli.artifact_root / message_directory / f"rank{rank}"
    print(
        f"[ICR prebeliefs rank={rank}] items={len(assigned_ids)} model={cli.model}",
        flush=True,
    )
    for item_id in assigned_ids:
        item = data[item_id]
        for agent_id in ("A", "B"):
            record_path = records_dir / f"item_{item_id:04d}_{agent_id}.json"
            if record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                prefix = cli.artifact_root / cached.get("statebridge_prefix_file", "")
                if (
                    cached.get("status") == "complete"
                    and cached.get("config_fingerprint") == config["fingerprint"]
                    and prefix.is_file()
                ):
                    print(f"[ICR prebeliefs rank={rank}] resume item={item_id} agent={agent_id}", flush=True)
                    continue
            seed = prebelief_seed(
                cli.global_seed, item_id, agent_id, cli.replication_id
            )
            generated = runtime.generate_prebelief(str(item["question"]), seed=seed)
            relative_prefix = Path(message_directory) / f"rank{rank}" / f"item_{item_id:04d}_{agent_id}.safetensors"
            prefix_path = cli.artifact_root / relative_prefix
            atomic_save_prefix(prefix_path, generated.pop("prefix"))
            gold = canonical_gold(cli.task, item.get("gold"))
            record = {
                "status": "complete",
                "config_fingerprint": config["fingerprint"],
                "item_id": item_id,
                "agent_id": agent_id,
                "replication_id": cli.replication_id or "seed_pair_00",
                "benchmark": config["benchmark"],
                "task": cli.task,
                "question": str(item["question"]),
                "gold": gold,
                **generated["record"],
                "correct": answer_is_correct(
                    cli.task, generated["record"]["parsed_answer"], gold
                ),
                "statebridge_prefix_file": str(relative_prefix),
                "statebridge_prefix_shape": list(generated["record"]["statebridge"]["K"] and [1, generated["record"]["statebridge"]["K"], runtime.bridge.hidden_size]),
                "statebridge_prefix_dtype": str(runtime.bridge.dtype).replace("torch.", ""),
                "completed_at": datetime.now().isoformat(),
            }
            atomic_write_json(record_path, record)
            print(
                f"[ICR prebeliefs rank={rank}] item={item_id} agent={agent_id} "
                f"pred={record['parsed_answer']} gold={gold} correct={record['correct']} "
                f"tokens={record['generation_length']}",
                flush=True,
            )
            if stop_requested:
                break
        if stop_requested:
            break
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(records_dir.glob("item_*.json"))
    ]
    atomic_write_jsonl(cli.artifact_root / "prebeliefs" / f"rank{rank}.jsonl", rows)


if __name__ == "__main__":
    main()
