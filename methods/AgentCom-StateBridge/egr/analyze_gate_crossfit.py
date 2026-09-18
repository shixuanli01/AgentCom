"""Development-only cross-fitted audit of cached EGR margins on MedQA300."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from icr.analysis_v2 import ci95, condition_bootstrap, condition_metrics
from icr.protocol import atomic_write_json, atomic_write_jsonl, classify_pair


METRICS = ("accuracy", "cr", "pr", "si", "sra", "fcs", "fws", "follow_selectivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fitted EGR gate audit")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--egr-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260916)
    parser.add_argument("--num-bootstrap", type=int, default=10_000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def threshold_grid(scores: Iterable[Mapping[str, Any]]) -> list[float]:
    positive = np.array(
        [float(row["G"]) for row in scores if row.get("G") is not None and float(row["G"]) > 0.0]
    )
    values = [0.0]
    if positive.size:
        values.extend(float(np.quantile(positive, q)) for q in (0.25, 0.50, 0.75, 0.90))
    return sorted(set(values))


def materialize(
    scores: Iterable[Mapping[str, Any]],
    prebeliefs: Mapping[tuple[int, str], Mapping[str, Any]],
    tau: float,
    condition: str,
) -> list[dict[str, Any]]:
    rows = []
    for score in scores:
        item_id = int(score["item_id"])
        direction = str(score["direction"])
        sender_id, receiver_id = ("A", "B") if direction == "A_to_B" else ("B", "A")
        sender = prebeliefs[(item_id, sender_id)]
        receiver = prebeliefs[(item_id, receiver_id)]
        gate = bool(
            score.get("decision_reason") == "scored_disagreement"
            and score.get("ME") is not None
            and score.get("G") is not None
            and float(score["ME"]) > 0.0
            and float(score["G"]) > tau
        )
        post = sender.get("parsed_answer") if gate else receiver.get("parsed_answer")
        rows.append(
            {
                "status": "complete",
                "benchmark": "medqa300",
                "replication_id": score["replication_id"],
                "item_id": item_id,
                "direction": direction,
                "condition": condition,
                "sender_agent_id": sender_id,
                "receiver_agent_id": receiver_id,
                "sender_pre_answer": sender.get("parsed_answer"),
                "sender_correct": bool(sender["correct"]),
                "receiver_pre_answer": receiver.get("parsed_answer"),
                "receiver_pre_correct": bool(receiver["correct"]),
                "receiver_post_answer": post,
                "receiver_post_correct": post == receiver["gold"],
                "gold": receiver["gold"],
                "pair_classification": classify_pair(bool(sender["correct"]), bool(receiver["correct"])),
                "answer_changed": post != receiver.get("parsed_answer"),
                "followed_sender": post == sender.get("parsed_answer"),
                "generation_seconds": float(score.get("scoring_seconds", 0.0)),
                "generation_length": 0,
                "prompt_tokens": 0,
                "communication_payload": {},
                "gate_open": gate,
                "tau": tau,
            }
        )
    return rows


def objective(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["si"]),
        float(row["accuracy"]),
        float(row["pr"]),
        -float(row["tau"]),
    )


def main() -> None:
    cli = parse_args()
    if cli.folds < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    scores = read_jsonl(cli.egr_root / "candidate_scores" / "merged.jsonl")
    beliefs = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    prebeliefs = {(int(row["item_id"]), str(row["agent_id"])): row for row in beliefs}
    item_ids = sorted({key[0] for key in prebeliefs})
    if len(scores) != 2 * len(item_ids):
        raise RuntimeError(f"Incomplete candidate scores: {len(scores)}/{2 * len(item_ids)}")

    shuffled = np.array(item_ids, dtype=np.int64)
    np.random.default_rng(cli.fold_seed).shuffle(shuffled)
    fold_ids = [set(map(int, values)) for values in np.array_split(shuffled, cli.folds)]
    selected_folds = []
    crossfit_rows = []
    for fold_index, heldout in enumerate(fold_ids):
        training_scores = [row for row in scores if int(row["item_id"]) not in heldout]
        candidates = []
        for tau in threshold_grid(training_scores):
            rows = materialize(training_scores, prebeliefs, tau, "egr_tau_train")
            metrics = condition_metrics(rows)
            candidates.append({"tau": tau, **{key: metrics[key] for key in METRICS}})
        selected = max(candidates, key=objective)
        heldout_scores = [row for row in scores if int(row["item_id"]) in heldout]
        heldout_rows = materialize(
            heldout_scores, prebeliefs, float(selected["tau"]), "egr_crossfit"
        )
        for row in heldout_rows:
            row["crossfit_fold"] = fold_index
        crossfit_rows.extend(heldout_rows)
        selected_folds.append(
            {
                "fold": fold_index,
                "heldout_items": sorted(heldout),
                "heldout_items_count": len(heldout),
                "train_items_count": len(item_ids) - len(heldout),
                "grid": candidates,
                "selected": selected,
            }
        )

    zero_rows = materialize(scores, prebeliefs, 0.0, "egr_zero_recomputed")
    baseline = read_jsonl(cli.source_root / "analysis" / "revisions" / "merged.jsonl")
    text_rows = [row for row in baseline if row.get("condition") == "true_text"]
    none_rows = [row for row in baseline if row.get("condition") == "none"]
    conditions = {
        "none": none_rows,
        "true_text": text_rows,
        "egr_zero": zero_rows,
        "egr_crossfit": crossfit_rows,
    }
    if any(len(rows) != 2 * len(item_ids) for rows in conditions.values()):
        raise RuntimeError({name: len(rows) for name, rows in conditions.items()})
    estimates = {name: condition_metrics(rows) for name, rows in conditions.items()}

    rng = np.random.default_rng(cli.fold_seed + 1)
    samples = rng.integers(0, len(item_ids), size=(cli.num_bootstrap, len(item_ids)))
    boot = {
        name: condition_bootstrap(rows, item_ids, samples)
        for name, rows in conditions.items()
    }
    pairwise = {}
    for name in ("egr_zero", "egr_crossfit"):
        pairwise[name] = {}
        for metric in METRICS:
            pairwise[name][metric] = {
                "delta": estimates[name][metric] - estimates["true_text"][metric],
                "ci95": ci95(boot[name][metric] - boot["true_text"][metric]),
            }

    output = cli.egr_root / "gate_crossfit"
    atomic_write_jsonl(output / "egr_crossfit.jsonl", crossfit_rows)
    atomic_write_json(output / "folds.json", selected_folds)
    atomic_write_json(
        output / "summary.json",
        {
            "status": "development_only",
            "warning": "MedQA300 has been heavily inspected; cross-fitting does not make it a new held-out benchmark.",
            "folds": cli.folds,
            "fold_seed": cli.fold_seed,
            "items": len(item_ids),
            "directional_cases": 2 * len(item_ids),
            "selected_thresholds": [fold["selected"]["tau"] for fold in selected_folds],
            "conditions": {
                name: {key: values[key] for key in (*METRICS, "rescues", "destructions", "net_correction")}
                for name, values in estimates.items()
            },
            "pairwise_vs_full_text": pairwise,
            "bootstrap": {
                "iterations": cli.num_bootstrap,
                "cluster": "item_id",
                "seed": cli.fold_seed + 1,
            },
        },
    )
    print(f"Wrote development-only cross-fit audit to {output}", flush=True)


if __name__ == "__main__":
    main()
