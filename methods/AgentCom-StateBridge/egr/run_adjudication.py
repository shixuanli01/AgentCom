"""Symmetric evidence adjudication over the three-agent ICR artifacts.

The ICR channels all hand a receiver its own prior answer and ask it to revise.
That framing is what makes correction and preservation separate quantities, and
what makes them trade off: a receiver told "you said B, here is a message" is
deciding whether to defect from B, not deciding the question.

This protocol removes the prior answer. Two evidence packets are presented, the
question is solved fresh, and neither source is marked as the reader's own. The
pair is therefore unordered: {X, Y} produces one answer, where ICR produced two.

That makes the comparison exact rather than approximate. An unordered mixed pair
-- one agent right, one wrong -- contributes exactly one correction case and one
destruction case to ICR, so ICR's SI is the average over the two orderings of
the same pair. The adjudicator's resolution rate on those pairs measures the
same thing on the same denominator.

Agents are ordered alphabetically in the prompt, so evidence set 1 is always the
earlier agent. Position effects are not averaged out; they are a declared
limitation rather than a hidden one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from egr.evidence import build_evidence_packet
from egr.evidence_v2 import build_evidence_packet_v2
from egr.prompts import build_contrast_adjudication_prompt
from icr.benchmarks import benchmark_metadata, load_benchmark
from icr.protocol import atomic_write_json, parse_task_answer, replicated_stable_seed, reset_rng
from icr.runtime import ICRRuntime

AGENTS = ("A", "B", "C")


def classify(cx: bool, cy: bool) -> str:
    if cx and cy:
        return "both_correct"
    if cx or cy:
        return "mixed"
    return "both_wrong"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", choices=("v1", "v2", "raw"), default="v1")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    cli = parser.parse_args()

    config = json.loads((cli.artifact_root / "config.json").read_text(encoding="utf-8"))
    task = str(config.get("dataset", "medqa"))
    metadata = benchmark_metadata(task, load_benchmark(task))

    beliefs: dict[int, dict[str, dict]] = {}
    for path in cli.artifact_root.glob("prebeliefs/rank*/records/item_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        beliefs.setdefault(int(record["item_id"]), {})[str(record["agent_id"])] = record
    retained = sorted(
        item for item, agents in beliefs.items()
        if len(agents) == len(AGENTS)
        and not all(bool(a["correct"]) for a in agents.values())
    )
    pairs = [(item, x, y) for item in retained for x, y in itertools.combinations(AGENTS, 2)]
    assigned = pairs[cli.rank :: cli.world_size]

    runtime = ICRRuntime(
        task=task, model_name=config["model"], device=torch.device("cuda:0"),
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
    )
    records_dir = cli.out / f"rank{cli.rank}" / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    def packet(record):
        if cli.evidence == "raw":
            return str(record["reasoning_text"])
        meta = metadata[int(record["item_id"])]
        counter = lambda t: len(runtime.model.tokenizer(t, add_special_tokens=False)["input_ids"])
        if cli.evidence == "v2":
            built = build_evidence_packet_v2(
                record, meta, token_counter=counter, policy="all_options"
            )
            if not str(built["evidence_packet"]).strip():
                built = build_evidence_packet(record, meta, token_counter=counter)
        else:
            built = build_evidence_packet(record, meta, token_counter=counter)
        return str(built["evidence_packet"])

    print(f"[rank {cli.rank}] {len(assigned)} of {len(pairs)} unordered pairs", flush=True)
    written = 0
    for item, x, y in assigned:
        target = records_dir / f"item_{item:04d}_{x}{y}.json"
        if target.exists():
            continue
        rx, ry = beliefs[item][x], beliefs[item][y]
        seed = replicated_stable_seed(
            int(config["global_seed"]), str(config["replication_id"]), item,
            f"adjudicate_{x}{y}", cli.evidence,
        )
        prompt_body = build_contrast_adjudication_prompt(
            str(rx["question"]), packet(rx), packet(ry), task=task
        )
        reset_rng(seed)
        rendered = runtime._render(prompt_body)
        input_ids, mask, _, clean = runtime._encode_prompt(rendered, None)
        started = time.time()
        texts, _, token_ids, info = runtime.bridge._generate_with_prefix(
            runtime.embedding_layer(input_ids), mask,
            prefix_embeds=None, insert_position=None, need_hidden_states=False,
        )
        seconds = time.time() - started
        answer = parse_task_answer(task, texts[0])
        gold = rx["gold"]
        atomic_write_json(target, {
            "item_id": item, "pair": f"{x}{y}", "agent_x": x, "agent_y": y,
            "protocol": "symmetric_evidence_adjudication",
            "evidence_version": cli.evidence,
            "x_correct": bool(rx["correct"]), "y_correct": bool(ry["correct"]),
            "x_answer": rx["parsed_answer"], "y_answer": ry["parsed_answer"],
            "pair_classification": classify(bool(rx["correct"]), bool(ry["correct"])),
            "gold": gold, "parsed_answer": answer,
            "correct": bool(answer is not None and str(answer).lower() == str(gold).lower()),
            "adjudication_seed": seed,
            "prompt_tokens": int(mask.sum()), "prompt_sha256": __import__("hashlib").sha256(clean.encode()).hexdigest(),
            "generation_length": int(token_ids.shape[1]),
            "generation_seconds": seconds,
            "hit_eos": bool(info["hit_eos"]),
            "response": texts[0], "status": "complete",
        })
        written += 1
        if written % 20 == 0:
            print(f"[rank {cli.rank}] {written} written", flush=True)
    print(f"[rank {cli.rank}] done, {written} written", flush=True)


if __name__ == "__main__":
    main()
