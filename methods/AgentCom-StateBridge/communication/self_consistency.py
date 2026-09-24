"""Does self-consistency predict whether a belief is correct?

Every single-sample quantity measured so far is uninformative about
correctness: fifteen sender-side features top out at AUC 0.547, and the
receiver's own answer margin reaches 0.490. Yet the margin strongly predicts
whether the receiver will change its answer -- AUC 0.26 to 0.43 across seven
conditions, every one at z < -3 on 718 records. The model gates how open it is
on a variable that carries no information about being wrong, which is a
sufficient explanation for every channel sitting on one correction/preservation
trade-off.

Consistency across resamples is the one candidate gate signal that is not a
function of a single sample. This measures whether it separates correct beliefs
from incorrect ones before anything is built on it.

Sampling matches phase 1 exactly -- same prompt, same temperature and top-p --
so the resamples are draws from the distribution the frozen belief came from.
Selection is case-control, equal numbers of correct and incorrect beliefs, which
leaves AUC unbiased while tightening it at a given cost.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from icr.protocol import (
    atomic_write_json, independent_solver_prompt, parse_task_answer,
    replicated_stable_seed, reset_rng,
)
from icr.runtime import ICRRuntime

AGENTS = ("A", "B", "C")


def select_case_control(beliefs, per_class: int, seed_tag: str):
    """Equal correct and incorrect beliefs, chosen by a deterministic hash."""
    import hashlib

    def rank(key):
        payload = f"{seed_tag}|{key[0]}|{key[1]}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    correct = sorted((k for k, r in beliefs.items() if r["correct"]), key=rank)
    wrong = sorted((k for k, r in beliefs.items() if not r["correct"]), key=rank)
    return correct[:per_class] + wrong[:per_class]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("artifacts/icr_v3/medqa_full_seed42"))
    parser.add_argument("--out", type=Path,
                        default=Path("runs/medqa/self_consistency"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    cli = parser.parse_args()

    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    task = str(config.get("dataset", "medqa"))

    by_item: dict[int, dict[str, dict]] = {}
    for path in cli.artifact_root.glob("prebeliefs/rank*/records/item_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        by_item.setdefault(int(record["item_id"]), {})[str(record["agent_id"])] = record
    beliefs = {
        (item, agent): record
        for item, agents in by_item.items()
        if len(agents) == len(AGENTS) and not all(a["correct"] for a in agents.values())
        for agent, record in agents.items()
    }
    chosen = select_case_control(beliefs, cli.per_class, "self_consistency_v1")
    assigned = chosen[cli.rank :: cli.world_size]

    runtime = ICRRuntime(
        task=task, model_name=config["model"], device=torch.device("cuda:0"),
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )
    records_dir = cli.out / f"rank{cli.rank}" / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rank {cli.rank}] {len(assigned)} of {len(chosen)} beliefs "
          f"x {cli.samples} resamples", flush=True)

    for item, agent in assigned:
        target = records_dir / f"item_{item:04d}_{agent}.json"
        if target.exists():
            continue
        record = beliefs[(item, agent)]
        prompt = runtime._render(independent_solver_prompt(task, str(record["question"])))
        input_ids, mask, _, _ = runtime._encode_prompt(prompt, None)
        embeds = runtime.embedding_layer(input_ids)
        answers, seconds = [], []
        for index in range(cli.samples):
            seed = replicated_stable_seed(
                int(config["global_seed"]), str(config["replication_id"]),
                item, f"consistency_{agent}", str(index),
            )
            reset_rng(seed)
            started = time.time()
            texts, _, _, _ = runtime.bridge._generate_with_prefix(
                embeds, mask, prefix_embeds=None, insert_position=None,
                need_hidden_states=False,
            )
            seconds.append(time.time() - started)
            answers.append(parse_task_answer(task, texts[0]))

        original = record["parsed_answer"]
        parsed = [a for a in answers if a is not None]
        counts = Counter(parsed)
        modal, modal_n = (counts.most_common(1)[0] if counts else (None, 0))
        atomic_write_json(target, {
            "item_id": item, "agent_id": agent,
            "original_answer": original, "correct": bool(record["correct"]),
            "gold": record["gold"], "resamples": cli.samples,
            "resample_answers": answers, "parsed_resamples": len(parsed),
            "agreement_with_original": (
                sum(1 for a in parsed if a == original) / len(parsed) if parsed else None
            ),
            "modal_answer": modal,
            "modal_agreement": (modal_n / len(parsed)) if parsed else None,
            "distinct_answers": len(counts),
            "modal_matches_original": (modal == original) if modal else None,
            "seconds_total": sum(seconds), "seconds_mean": sum(seconds) / len(seconds),
        })
    print(f"[rank {cli.rank}] done", flush=True)


if __name__ == "__main__":
    main()
