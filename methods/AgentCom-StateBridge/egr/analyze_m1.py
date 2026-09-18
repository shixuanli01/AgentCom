"""Checkpoint-M1 analysis against immutable frozen ICR controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from icr.analysis_v2 import ci95, condition_bootstrap, condition_metrics
from icr.protocol import atomic_write_json


DISPLAY = {
    "none": "None",
    "true_text": "Full Text",
    "claim_only": "Claim Only",
    "evidence_only": "Evidence Only",
    "egr_zero": "EGR-zero",
    "true_statebridge": "StateBridge",
    "true_latentmas": "LatentMAS",
}
ORDER = tuple(DISPLAY)
METRICS = ("accuracy", "cr", "pr", "si", "sra", "fcs", "fws", "follow_selectivity")
PAIRWISE = ("claim_only", "evidence_only", "egr_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EGR checkpoint M1")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--egr-root", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_baselines(source_root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(source_root / "analysis" / "revisions" / "merged.jsonl")
    for path in sorted((source_root / "revisions").glob("rank*_latentmas.jsonl")):
        rows.extend(read_jsonl(path))
    wanted = {"none", "true_text", "true_statebridge", "true_latentmas"}
    return [row for row in rows if row.get("condition") in wanted]


def load_new(egr_root: Path) -> list[dict[str, Any]]:
    rows = []
    for condition in ("claim_only", "evidence_only", "egr_zero"):
        rows.extend(read_jsonl(egr_root / "revisions" / f"{condition}.jsonl"))
    return rows


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    cli = parse_args()
    baseline_rows = load_baselines(cli.source_root)
    new_rows = load_new(cli.egr_root)
    new_conditions = {
        condition: [row for row in new_rows if row.get("condition") == condition]
        for condition in PAIRWISE
    }
    present_new = [values for values in new_conditions.values() if values]
    if present_new:
        common_keys = set.intersection(
            *[
                {(int(row["item_id"]), str(row["direction"])) for row in values}
                for values in present_new
            ]
        )
        baseline_rows = [
            row
            for row in baseline_rows
            if (int(row["item_id"]), str(row["direction"])) in common_keys
        ]
        new_rows = [
            row
            for row in new_rows
            if (int(row["item_id"]), str(row["direction"])) in common_keys
        ]
    rows = baseline_rows + new_rows
    by_condition = {
        condition: [row for row in rows if row.get("condition") == condition]
        for condition in ORDER
    }
    by_condition = {key: value for key, value in by_condition.items() if value}
    expected = 2 * len({int(row["item_id"]) for row in new_rows}) if cli.allow_partial else 600
    incomplete = {key: len(value) for key, value in by_condition.items() if len(value) != expected}
    if incomplete and not cli.allow_partial:
        raise RuntimeError(f"Incomplete M1 conditions: {incomplete}")

    item_ids = sorted({int(row["item_id"]) for values in by_condition.values() for row in values})
    rng = np.random.default_rng(cli.bootstrap_seed)
    samples = rng.integers(0, len(item_ids), size=(cli.num_bootstrap, len(item_ids)))
    metrics = {condition: condition_metrics(values) for condition, values in by_condition.items()}
    boot = {
        condition: condition_bootstrap(values, item_ids, samples)
        for condition, values in by_condition.items()
    }

    metric_rows: list[dict[str, Any]] = []
    bootstrap_summary: dict[str, Any] = {"conditions": {}, "pairwise_vs_full_text": {}}
    for condition in ORDER:
        if condition not in metrics:
            continue
        metric = metrics[condition]
        output = {"condition": condition, "display": DISPLAY[condition], "records": len(by_condition[condition])}
        bootstrap_summary["conditions"][condition] = {}
        for name in METRICS:
            output[name] = metric[name]
            interval = ci95(boot[condition][name])
            output[f"{name}_ci95_low"] = interval[0] if interval else None
            output[f"{name}_ci95_high"] = interval[1] if interval else None
            bootstrap_summary["conditions"][condition][name] = {
                "estimate": metric[name], "ci95": interval
            }
        output.update(
            {
                "answer_change_rate": metric["answer_change_rate"],
                "rescues": metric["rescues"],
                "destructions": metric["destructions"],
                "net_correction": metric["net_correction"],
                "invalid_outputs": metric["invalid_outputs"],
                "generation_seconds": metric["total_generation_seconds"],
            }
        )
        metric_rows.append(output)

    if "true_text" in boot:
        for condition in PAIRWISE:
            if condition not in boot:
                continue
            bootstrap_summary["pairwise_vs_full_text"][condition] = {}
            for name in METRICS:
                values = boot[condition][name] - boot["true_text"][name]
                bootstrap_summary["pairwise_vs_full_text"][condition][name] = {
                    "delta": metrics[condition][name] - metrics["true_text"][name],
                    "ci95": ci95(values),
                }

    bootstrap_summary.update(
        {
            "iterations": cli.num_bootstrap,
            "seed": cli.bootstrap_seed,
            "cluster": "item_id",
            "both_directions_retained": True,
        }
    )
    write_csv(cli.egr_root / "metrics.csv", metric_rows)
    atomic_write_json(cli.egr_root / "bootstrap.json", bootstrap_summary)
    write_csv(
        cli.egr_root / "cr_pr_points.csv",
        [
            {"condition": row["condition"], "display": row["display"], "cr": row["cr"], "pr": row["pr"]}
            for row in metric_rows
        ],
    )
    write_csv(
        cli.egr_root / "fcs_fws_points.csv",
        [
            {
                "condition": row["condition"],
                "display": row["display"],
                "fcs": row["fcs"],
                "fws": row["fws"],
                "follow_selectivity": row["follow_selectivity"],
                "influence": row["answer_change_rate"],
                "utility_vs_none": (
                    row["accuracy"] - metrics["none"]["accuracy"] if "none" in metrics else None
                ),
            }
            for row in metric_rows
        ],
    )
    frame_columns = [
        "benchmark", "replication_id", "item_id", "direction", "condition", "gold",
        "sender_correct", "receiver_pre_correct", "receiver_post_correct",
        "sender_pre_answer", "receiver_pre_answer", "receiver_post_answer",
        "pair_classification", "answer_changed", "followed_sender", "generation_seconds",
    ]
    frame = pd.DataFrame(rows)
    for column in frame_columns:
        if column not in frame:
            frame[column] = None
    frame[frame_columns].to_parquet(cli.egr_root / "paired_cases.parquet", index=False)

    leakage_path = cli.egr_root / "evidence_packets" / "leakage_summary.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8")) if leakage_path.exists() else None
    summary = {
        "checkpoint": "M1",
        "status": "partial" if incomplete else "complete",
        "label": "MedQA300 diagnostic; no threshold tuning",
        "items": len(item_ids),
        "conditions": {row["condition"]: row for row in metric_rows},
        "pairwise_vs_full_text": bootstrap_summary["pairwise_vs_full_text"],
        "evidence_leakage": leakage,
        "incomplete_conditions": incomplete,
    }
    atomic_write_json(cli.egr_root / "summary.json", summary)

    lines = [
        "# EGR checkpoint M1 — MedQA300 diagnostic",
        "",
        "No EGR threshold was tuned on MedQA300.",
        "",
        "| Condition | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity | Rescue/Destroy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['display']} | {fmt(row['accuracy'])} | {fmt(row['cr'])} | "
            f"{fmt(row['pr'])} | {fmt(row['si'])} | {fmt(row['sra'])} | "
            f"{fmt(row['fcs'])} | {fmt(row['fws'])} | {fmt(row['follow_selectivity'])} | "
            f"{row['rescues']}/{row['destructions']} |"
        )
    if leakage:
        lines.extend(
            [
                "",
                "## Evidence leakage",
                "",
                f"- Explicit answer cues after filtering: {leakage['explicit_answer_cue_present']}/{leakage['packets']}",
                f"- Sender answer labels present: {leakage['answer_label_present']}/{leakage['packets']}",
                f"- Sender answer text present: {leakage['answer_text_present']}/{leakage['packets']}",
                f"- Token retention: {100 * leakage['token_retention_rate']:.2f}%",
            ]
        )
    lines.extend(["", "## Pairwise differences vs Full Text", ""])
    for condition, values in bootstrap_summary["pairwise_vs_full_text"].items():
        lines.append(f"### {DISPLAY[condition]}")
        for name in METRICS:
            result = values[name]
            interval = result["ci95"]
            lines.append(
                f"- {name}: {100 * result['delta']:.2f} pp, 95% CI "
                f"[{100 * interval[0]:.2f}, {100 * interval[1]:.2f}]"
            )
        lines.append("")
    (cli.egr_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote M1 analysis to {cli.egr_root}")


if __name__ == "__main__":
    main()
