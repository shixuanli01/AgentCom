"""Is the sender's own answer margin predictive of whether it is right?

A channel that transmits more when the sender is confident and less when it is
not could raise correction and preservation together, which no channel tested so
far does. That only works if confidence carries signal about correctness.

The margin is read from the frozen trajectory, teacher-forced: at the position
that produced the final answer token, take the logits restricted to the four
option tokens and compute

    margin = logit(chosen) - logsumexp(logits of the other three)

It uses the sender's own answer and nothing else. Correctness labels are read
here only to score the diagnostic offline; no channel would see them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from communication.cr_dnc_runtime import locate_answer_token
from icr.protocol import independent_solver_prompt
from icr.runtime import ICRRuntime

AGENTS = ("A", "B", "C")
CHOICES = ("A", "B", "C", "D")


def auc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based AUC; ties get half credit."""
    pairs = [(s, l) for s, l in zip(scores, labels)]
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
    )
    return wins / (len(pos) * len(neg))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("artifacts/icr_v3/medqa_full_seed42"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/medqa/cr_dnc_v1"))
    parser.add_argument("--subset", choices=("dev", "eval", "all"), default="all")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    cli = parser.parse_args()

    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    split = json.loads((cli.run_root / "split.json").read_text(encoding="utf-8"))
    item_ids = (sorted({*split["dev_item_ids"], *split["eval_item_ids"]})
                if cli.subset == "all" else list(split[f"{cli.subset}_item_ids"]))
    wanted = set(item_ids)

    records = []
    for path in cli.artifact_root.glob("prebeliefs/rank*/records/item_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["item_id"]) in wanted:
            records.append(record)
    records.sort(key=lambda r: (int(r["item_id"]), str(r["agent_id"])))
    assigned = records[cli.rank :: cli.world_size]

    runtime = ICRRuntime(
        task=config["dataset"], model_name=config["model"],
        device=torch.device("cuda:0"),
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )
    tokenizer = runtime.model.tokenizer
    option_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES]

    out_path = cli.out or (cli.run_root / f"sender_margins_{cli.subset}_rank{cli.rank}.jsonl")
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in assigned:
            ids = record["generated_token_ids"]
            position = locate_answer_token(ids, tokenizer)
            if position is None:
                handle.write(json.dumps({
                    "item_id": record["item_id"], "agent_id": record["agent_id"],
                    "status": "no_answer_token", "correct": bool(record["correct"]),
                }) + "\n")
                continue
            prompt = runtime._render(
                independent_solver_prompt(config["dataset"], record["question"])
            )
            prompt_ids, _, _, _ = runtime._encode_prompt(prompt, None)
            full = torch.cat(
                [prompt_ids, torch.tensor([ids], dtype=torch.long, device=runtime.device)],
                dim=1,
            )
            # The state one position left of the answer token produced it.
            logit_index = int(prompt_ids.shape[1]) + position - 1
            out = runtime.model.model(
                input_ids=full, attention_mask=torch.ones_like(full), use_cache=False
            )
            logits = out.logits[0, logit_index, :].float()
            option_logits = logits[option_ids]
            chosen = CHOICES.index(tokenizer.convert_ids_to_tokens([ids[position]])[0].upper())
            others = torch.cat([option_logits[:chosen], option_logits[chosen + 1 :]])
            margin = float(option_logits[chosen] - torch.logsumexp(others, dim=0))
            probabilities = torch.softmax(option_logits, dim=0)
            handle.write(json.dumps({
                "item_id": record["item_id"], "agent_id": record["agent_id"],
                "status": "ok", "sender_answer": CHOICES[chosen],
                "margin": margin,
                "chosen_probability": float(probabilities[chosen]),
                "entropy": float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()),
                "option_logits": [float(x) for x in option_logits],
                "correct": bool(record["correct"]),
            }) + "\n")
            written += 1
    print(f"[rank {cli.rank}] wrote {written} margins to {out_path}", flush=True)


if __name__ == "__main__":
    main()
