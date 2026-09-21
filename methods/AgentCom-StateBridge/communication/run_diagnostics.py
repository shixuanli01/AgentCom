"""Per-sender CR-DNC diagnostics over a split (plan section 14).

Writes one JSON line per sender so the projection profile, the intervention
size, the fallback rate and the construction cost can all be read back without
re-running the model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from icr.protocol import independent_solver_prompt
from icr.runtime import ICRRuntime
from communication.cr_dnc_runtime import build_payload

AGENTS = ("A", "B", "C")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("artifacts/icr_v3/medqa_full_seed42"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/medqa/cr_dnc_v1"))
    parser.add_argument("--subset", choices=("dev", "eval", "all"), default="dev")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    cli = parser.parse_args()

    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    split = json.loads((cli.run_root / "split.json").read_text(encoding="utf-8"))
    if cli.subset == "all":
        item_ids = sorted({*split["dev_item_ids"], *split["eval_item_ids"]})
    else:
        item_ids = list(split[f"{cli.subset}_item_ids"])
    wanted = set(item_ids)

    beliefs = {}
    for path in cli.artifact_root.glob("prebeliefs/rank*/records/item_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["item_id"]) in wanted:
            beliefs[(int(record["item_id"]), str(record["agent_id"]))] = record

    cache = cli.run_root / "sender_cache"
    out_path = cli.out or (cli.run_root / f"sender_diagnostics_{cli.subset}_alpha{cli.alpha}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = ICRRuntime(
        task=config["dataset"], model_name=config["model"],
        device=torch.device("cuda:0"),
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )

    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for item_id in item_ids:
            for agent_id in AGENTS:
                meta_path = cache / f"item_{item_id:04d}_{agent_id}.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                record = beliefs[(item_id, agent_id)]
                payload = load_file(str(meta_path.with_suffix(".safetensors")))["selected_hidden"]

                prompt = runtime._render(
                    independent_solver_prompt(config["dataset"], record["question"])
                )
                prompt_ids, _, _, _ = runtime._encode_prompt(prompt, None)

                started = time.time()
                suppressed, diagnostics = build_payload(
                    runtime,
                    raw_payload=payload.to(runtime.device),
                    token_ids=record["generated_token_ids"],
                    prompt_ids=prompt_ids,
                    sender_answer=meta["sender_answer"],
                    alpha=cli.alpha,
                )
                diagnostics.update(
                    dataset=config["dataset"], item_id=item_id, sender_agent_id=agent_id,
                    construction_seconds=time.time() - started,
                    generated_length=record["generation_length"],
                    payload_shape=list(payload.shape),
                )
                handle.write(json.dumps(diagnostics) + "\n")
                written += 1
    print(f"wrote {written} diagnostics to {out_path}")


if __name__ == "__main__":
    main()
