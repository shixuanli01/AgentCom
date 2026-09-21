"""Cache the pre-alignment StateBridge payload for every unique sender.

CR-DNC modifies the sender hidden states before StateBridge aligns them. That
tensor is not on disk: `_prepare_handoff` returns only the aligned prefix.

It cannot be recovered by teacher-forcing the cached token ids. That reproduces
the right tokens -- selected_indices match exactly -- but the payload lands
10-13% away in relative Frobenius error, because incremental decoding over a KV
cache and a single full-sequence forward do not agree in bfloat16 once the
difference has accumulated through 36 layers and thousands of positions. Mixing
the two would mean CR-DNC was measured against a StateBridge baseline it does
not actually share.

Re-running the recorded seed does agree: identical token ids, zero relative
difference against the cached prefix. So each sender is replayed once under its
own seed, and the replay is only accepted when the token ids match the cached
trajectory byte for byte and the realigned prefix matches the cached prefix.
This regenerates nothing new -- the trajectory is bit-identical -- it recovers a
discarded intermediate.

The payload depends on the sender alone, so it is cached per
(dataset, item_id, sender_agent_id) and reused for every receiver.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from icr.protocol import prebelief_seed
from icr.runtime import ICRRuntime

AGENTS = ("A", "B", "C")


def _belief_index(root: Path) -> dict[tuple[int, str], dict]:
    index: dict[tuple[int, str], dict] = {}
    for path in root.glob("prebeliefs/rank*/records/item_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        index[(int(record["item_id"]), str(record["agent_id"]))] = record
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("artifacts/icr_v3/medqa_full_seed42"))
    parser.add_argument("--split", type=Path,
                        default=Path("runs/medqa/cr_dnc_v1/split.json"))
    parser.add_argument("--subset", choices=("dev", "eval", "all"), default="dev")
    parser.add_argument("--out", type=Path,
                        default=Path("runs/medqa/cr_dnc_v1/sender_cache"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    cli = parser.parse_args()

    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    split = json.loads(cli.split.read_text(encoding="utf-8"))
    if cli.subset == "all":
        item_ids = sorted({*split["dev_item_ids"], *split["eval_item_ids"]})
    else:
        item_ids = list(split[f"{cli.subset}_item_ids"])

    beliefs = _belief_index(cli.artifact_root)
    senders = [(i, a) for i in item_ids for a in AGENTS if (i, a) in beliefs]
    assigned = senders[cli.rank :: cli.world_size]
    cli.out.mkdir(parents=True, exist_ok=True)

    runtime = ICRRuntime(
        task=config["dataset"],
        model_name=config["model"],
        device=torch.device("cuda:0"),
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )
    print(f"[rank {cli.rank}] {len(assigned)} senders of {len(senders)}", flush=True)

    done = skipped = 0
    for item_id, agent_id in assigned:
        target = cli.out / f"item_{item_id:04d}_{agent_id}.safetensors"
        meta_path = target.with_suffix(".json")
        if target.exists() and meta_path.exists():
            skipped += 1
            continue
        record = beliefs[(item_id, agent_id)]
        seed = prebelief_seed(
            config["global_seed"], item_id, agent_id, config["replication_id"]
        )
        if seed != record["generation_seed"]:
            raise RuntimeError(
                f"seed mismatch for item {item_id} agent {agent_id}: "
                f"derived {seed}, recorded {record['generation_seed']}"
            )
        started = time.time()
        out = runtime.generate_prebelief(
            record["question"], seed=seed, return_sender_states=True
        )
        elapsed = time.time() - started

        # Accept the replay only if it is the same trajectory and the same payload.
        if out["record"]["generated_token_ids"] != record["generated_token_ids"]:
            raise RuntimeError(f"token ids diverged for item {item_id} agent {agent_id}")
        cached_prefix = load_file(
            str(cli.artifact_root / record["statebridge_prefix_file"])
        )["statebridge_prefix"].float()
        replay_prefix = out["prefix"].float()
        drift = float((replay_prefix - cached_prefix).abs().max())
        if drift != 0.0:
            raise RuntimeError(
                f"prefix drift {drift:.3e} for item {item_id} agent {agent_id}; "
                "the payload would not correspond to the published baseline"
            )

        states = out["sender_states"]
        save_file(
            {
                "selected_hidden": states["selected_hidden"].contiguous(),
                "selected_token_ids": states["selected_token_ids"].contiguous(),
            },
            str(target),
        )
        meta_path.write_text(json.dumps({
            "dataset": config["dataset"],
            "benchmark": config["benchmark"],
            "item_id": item_id,
            "sender_agent_id": agent_id,
            "sender_answer": record["parsed_answer"],
            "model": config["model"],
            "config_fingerprint": config["fingerprint"],
            "generation_seed": seed,
            "selected_indices": states["selected_indices"],
            "post_think_length": states["post_think_length"],
            "generated_length": record["generation_length"],
            "hidden_shape": list(states["selected_hidden"].shape),
            "prefix_drift_vs_cache": drift,
            "replay_seconds": elapsed,
            "payload_version": "raw_statebridge_selected_hidden_v1",
        }, indent=2), encoding="utf-8")
        done += 1
        if done % 5 == 0:
            print(f"[rank {cli.rank}] {done} built, {skipped} cached", flush=True)

    print(f"[rank {cli.rank}] done: {done} built, {skipped} reused", flush=True)


if __name__ == "__main__":
    main()
