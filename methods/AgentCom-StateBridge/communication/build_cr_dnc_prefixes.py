"""Pre-build CR-DNC prefixes so the revision run differs only in the payload.

The prefix is produced exactly as StateBridge produces its own -- same
alignment, same post-processing -- from a payload that had the estimated
decision direction projected out first. Writing them ahead of time means the
revision pipeline loads a file and changes nothing else: same receiver prompt,
same decoding settings, same revision seed, same directed pairs.

A sender whose counterfactual cannot be built falls back to its raw payload and
stays in the population, as the plan requires; the fallback is recorded in the
per-sender metadata and in the prefix file itself.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from icr.protocol import independent_solver_prompt
from icr.runtime import ICRRuntime
from communication.cr_dnc_runtime import build_payload

AGENTS = ("A", "B", "C")


def alpha_tag(alpha: float) -> str:
    """0.5 -> 'a050', 1.0 -> 'a100'."""
    return f"a{int(round(alpha * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("artifacts/icr_v3/medqa_full_seed42"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/medqa/cr_dnc_v1"))
    parser.add_argument("--subset", choices=("dev", "eval", "all"), default="dev")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--control", choices=("none", "random_same_alpha", "random_matched_norm"),
        default="none",
        help=(
            "Control arm. random_same_alpha projects out a random unit direction "
            "at the same alpha, which tests whether removing any one direction "
            "hurts. random_matched_norm rescales alpha per sender so the relative "
            "intervention norm equals CR-DNC's, which tests whether a perturbation "
            "of that size hurts regardless of what it points at. A random "
            "direction in 2560 dimensions barely touches the payload -- measured "
            "1.1% of CR-DNC's projection ratio -- so the first control alone would "
            "compare two very different intervention sizes."
        ),
    )
    parser.add_argument("--control-seed", type=int, default=20260921)
    parser.add_argument("--tag-suffix", type=str, default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    cli = parser.parse_args()

    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    split = json.loads((cli.run_root / "split.json").read_text(encoding="utf-8"))
    item_ids = (sorted({*split["dev_item_ids"], *split["eval_item_ids"]})
                if cli.subset == "all" else list(split[f"{cli.subset}_item_ids"]))
    wanted = set(item_ids)

    beliefs = {}
    for path in cli.artifact_root.glob("prebeliefs/rank*/records/item_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["item_id"]) in wanted:
            beliefs[(int(record["item_id"]), str(record["agent_id"]))] = record

    cache = cli.run_root / "sender_cache"
    senders = [(i, a) for i in item_ids for a in AGENTS
               if (cache / f"item_{i:04d}_{a}.json").exists()]
    assigned = senders[cli.rank :: cli.world_size]

    runtime = ICRRuntime(
        task=config["dataset"], model_name=config["model"],
        device=torch.device("cuda:0"),
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )
    print(f"[rank {cli.rank}] {len(assigned)} senders x {len(cli.alphas)} alphas",
          flush=True)

    built = fallbacks = 0
    for item_id, agent_id in assigned:
        meta_path = cache / f"item_{item_id:04d}_{agent_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        record = beliefs[(item_id, agent_id)]
        cached = load_file(str(meta_path.with_suffix(".safetensors")))
        raw = cached["selected_hidden"]
        selected_token_ids = cached["selected_token_ids"].to(runtime.device)

        prompt = runtime._render(
            independent_solver_prompt(config["dataset"], record["question"])
        )
        prompt_ids, _, _, _ = runtime._encode_prompt(prompt, None)

        for alpha in cli.alphas:
            tag = alpha_tag(alpha)
            if cli.tag_suffix:
                tag = f"{cli.tag_suffix}_{tag}"
            out_dir = cli.run_root / "prefixes" / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"item_{item_id:04d}_{agent_id}.safetensors"
            if target.exists():
                continue
            started = time.time()
            override = None
            effective_alpha = alpha
            if cli.control != "none":
                # The reference run fixes what is being controlled for: the
                # direction CR-DNC would have used, and the intervention size it
                # would have produced.
                _, reference = build_payload(
                    runtime, raw_payload=raw.to(runtime.device),
                    token_ids=record["generated_token_ids"], prompt_ids=prompt_ids,
                    sender_answer=meta["sender_answer"], alpha=alpha,
                )
                generator = torch.Generator().manual_seed(
                    cli.control_seed + item_id * 31 + ord(agent_id)
                )
                override = torch.randn(raw.shape[-1], generator=generator)
                override = override / override.norm()
                if cli.control == "random_matched_norm":
                    flat = raw.reshape(-1, raw.shape[-1]).float()
                    random_pull = float((flat @ override).norm())
                    wanted_norm = (
                        reference["relative_intervention_norm"] * float(flat.norm())
                    )
                    effective_alpha = wanted_norm / max(random_pull, 1e-8)
            suppressed, diagnostics = build_payload(
                runtime, raw_payload=raw.to(runtime.device),
                token_ids=record["generated_token_ids"], prompt_ids=prompt_ids,
                sender_answer=meta["sender_answer"], alpha=effective_alpha,
                direction_override=override,
            )
            diagnostics["control"] = cli.control
            diagnostics["nominal_alpha"] = alpha
            diagnostics["effective_alpha"] = effective_alpha
            with torch.no_grad():
                # The alignment and its post-processing are the unmodified
                # StateBridge ones; only the payload handed to them changed.
                aligned = runtime.bridge._align_hidden_sequence(
                    suppressed.to(runtime.device), selected_token_ids
                )
                prefix = runtime.bridge._process_prefix(aligned).detach().cpu()
            fell_back = diagnostics["counterfactual_status"] == "fallback"
            fallbacks += int(fell_back)
            save_file(
                {
                    "statebridge_prefix": prefix.contiguous(),
                    "fell_back_to_raw": torch.tensor([float(fell_back)]),
                },
                str(target),
            )
            target.with_suffix(".json").write_text(
                json.dumps(
                    {
                        k: v for k, v in diagnostics.items()
                        if k not in ("projection_magnitude_by_position",
                                     "projection_ratio_by_position")
                    }
                    | {
                        "item_id": item_id, "sender_agent_id": agent_id,
                        "alpha": alpha, "alpha_tag": tag,
                        "prefix_seconds": time.time() - started,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            built += 1
    print(f"[rank {cli.rank}] built {built} prefixes, {fallbacks} fallbacks", flush=True)


if __name__ == "__main__":
    main()
