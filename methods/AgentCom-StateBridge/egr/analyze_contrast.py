"""Pooled MedQA300 analysis for symmetric EGR contrast adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from icr.analysis_v2 import ci95, condition_bootstrap, condition_metrics
from icr.protocol import atomic_write_json


CONDITIONS = ("none", "true_text", "true_statebridge", "egr_contrast")
DISPLAY = {
    "none": "None",
    "true_text": "Full Text",
    "true_statebridge": "StateBridge",
    "egr_contrast": "EGR-Contrast",
}
METRICS = ("accuracy", "cr", "pr", "si", "sra", "fcs", "fws", "follow_selectivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EGR-Contrast on MedQA300")
    parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--contrast-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def baseline_path(root: Path) -> Path:
    candidates = (
        root / "revisions" / "merged.jsonl",
        root / "analysis" / "revisions" / "merged.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No frozen revision file below {root}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verdict_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "items": len(rows),
        "agreements": 0,
        "disagreements": 0,
        "one_correct_disagreements": 0,
        "both_wrong_disagreements": 0,
        "one_correct_selected_correct": 0,
        "one_correct_selected_wrong": 0,
        "one_correct_invalid_or_third": 0,
        "correct_source_A_cases": 0,
        "correct_source_A_selected": 0,
        "correct_source_B_cases": 0,
        "correct_source_B_selected": 0,
        "selected_A": 0,
        "selected_B": 0,
        "selected_third": 0,
        "selected_invalid": 0,
        "both_wrong_selected_third": 0,
    }
    for row in rows:
        if row["decision_reason"] != "contrast_adjudication":
            result["agreements"] += 1
            continue
        result["disagreements"] += 1
        answer_a, answer_b, verdict = row["answer_A"], row["answer_B"], row["verdict"]
        if verdict is None:
            result["selected_invalid"] += 1
        elif verdict == answer_a:
            result["selected_A"] += 1
        elif verdict == answer_b:
            result["selected_B"] += 1
        else:
            result["selected_third"] += 1

        correct_a, correct_b = bool(row["correct_A"]), bool(row["correct_B"])
        if correct_a != correct_b:
            result["one_correct_disagreements"] += 1
            gold_answer = answer_a if correct_a else answer_b
            if correct_a:
                result["correct_source_A_cases"] += 1
                result["correct_source_A_selected"] += int(verdict == answer_a)
            else:
                result["correct_source_B_cases"] += 1
                result["correct_source_B_selected"] += int(verdict == answer_b)
            if verdict == gold_answer:
                result["one_correct_selected_correct"] += 1
            elif verdict in {answer_a, answer_b}:
                result["one_correct_selected_wrong"] += 1
            else:
                result["one_correct_invalid_or_third"] += 1
        else:
            result["both_wrong_disagreements"] += 1
            # The gold label is absent from both candidates exactly when both are wrong.
            if verdict not in {None, answer_a, answer_b}:
                result["both_wrong_selected_third"] += 1
    denominator = result["one_correct_disagreements"]
    result["one_correct_selection_accuracy"] = (
        result["one_correct_selected_correct"] / denominator if denominator else None
    )
    return result


def main() -> None:
    cli = parse_args()
    if len(cli.source_roots) != len(cli.contrast_roots):
        raise ValueError("source-roots and contrast-roots must have the same length")

    all_rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    per_seed: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    statuses: dict[str, Any] = {}

    for source_root, contrast_root in zip(cli.source_roots, cli.contrast_roots):
        contrast = read_jsonl(contrast_root / "revisions" / "egr_contrast.jsonl")
        verdicts = read_jsonl(contrast_root / "verdicts" / "merged.jsonl")
        if len(contrast) != 600 or len(verdicts) != 300:
            raise RuntimeError(f"Incomplete contrast artifact: {contrast_root}")
        replication_ids = {str(row["replication_id"]) for row in contrast}
        if len(replication_ids) != 1:
            raise RuntimeError(f"Ambiguous replication ids: {replication_ids}")
        replication_id = replication_ids.pop()

        frozen = read_jsonl(baseline_path(source_root))
        seed_rows: dict[str, list[dict[str, Any]]] = {
            condition: [row for row in frozen if row.get("condition") == condition]
            for condition in CONDITIONS[:-1]
        }
        seed_rows["egr_contrast"] = contrast
        for condition, rows in seed_rows.items():
            if len(rows) != 600:
                raise RuntimeError(
                    f"{replication_id}/{condition} has {len(rows)} records, expected 600"
                )
            keys = {(int(row["item_id"]), str(row["direction"])) for row in rows}
            if len(keys) != 600:
                raise RuntimeError(f"Duplicate/missing keys in {replication_id}/{condition}")
            all_rows[condition].extend(rows)

        per_seed[replication_id] = {
            condition: condition_metrics(rows) for condition, rows in seed_rows.items()
        }
        audit = verdict_audit(verdicts)
        audit["replication_id"] = replication_id
        audit_rows.append(audit)
        statuses[replication_id] = json.loads(
            (contrast_root / "run_status.json").read_text(encoding="utf-8")
        )

    expected = 600 * len(cli.contrast_roots)
    for condition, rows in all_rows.items():
        if len(rows) != expected:
            raise RuntimeError(f"{condition}: {len(rows)} pooled rows, expected {expected}")

    item_ids = list(range(300))
    rng = np.random.default_rng(cli.bootstrap_seed)
    samples = rng.integers(0, len(item_ids), size=(cli.num_bootstrap, len(item_ids)))
    pooled = {condition: condition_metrics(rows) for condition, rows in all_rows.items()}
    boot = {
        condition: condition_bootstrap(rows, item_ids, samples)
        for condition, rows in all_rows.items()
    }

    comparisons: dict[str, Any] = {}
    for baseline in ("true_text", "true_statebridge", "none"):
        key = f"egr_contrast_vs_{baseline}"
        comparisons[key] = {}
        for metric in METRICS:
            delta = pooled["egr_contrast"][metric] - pooled[baseline][metric]
            interval = ci95(boot["egr_contrast"][metric] - boot[baseline][metric])
            comparisons[key][metric] = {"delta": delta, "ci95": interval}

    pooled_audit = verdict_audit(
        [
            row
            for contrast_root in cli.contrast_roots
            for row in read_jsonl(contrast_root / "verdicts" / "merged.jsonl")
        ]
    )
    summary = {
        "protocol": "EGR-CONTRAST-v1",
        "label": "MedQA300 development diagnostic; not held-out",
        "items": 300,
        "replications": len(cli.contrast_roots),
        "directional_records_per_condition": expected,
        "pooled_conditions": pooled,
        "per_seed": per_seed,
        "pairwise": comparisons,
        "verdict_audit": {"pooled": pooled_audit, "per_seed": audit_rows},
        "runtime": statuses,
        "bootstrap": {
            "iterations": cli.num_bootstrap,
            "seed": cli.bootstrap_seed,
            "cluster": "item_id",
            "retains_all_replications_and_both_directions": True,
        },
    }
    cli.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cli.output_root / "summary.json", summary)

    metric_rows = []
    for condition in CONDITIONS:
        row = {"condition": condition, "display": DISPLAY[condition], **pooled[condition]}
        for metric in METRICS:
            row[f"{metric}_ci95"] = ci95(boot[condition][metric])
        metric_rows.append(row)
    write_csv(cli.output_root / "metrics.csv", metric_rows)
    write_csv(cli.output_root / "verdict_audit.csv", audit_rows)

    frame = pd.DataFrame(
        [
            {
                "replication_id": row.get("replication_id", "seed_pair_00"),
                "item_id": row["item_id"],
                "direction": row["direction"],
                "condition": condition,
                "gold": row["gold"],
                "sender_correct": row["sender_correct"],
                "receiver_pre_correct": row["receiver_pre_correct"],
                "receiver_post_correct": row["receiver_post_correct"],
                "receiver_pre_answer": row["receiver_pre_answer"],
                "receiver_post_answer": row["receiver_post_answer"],
                "pair_classification": row["pair_classification"],
            }
            for condition, rows in all_rows.items()
            for row in rows
        ]
    )
    frame.to_parquet(cli.output_root / "paired_cases.parquet", index=False)

    def pct(value: float | None) -> str:
        return "--" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# EGR-Contrast v1: MedQA300 three-seed diagnostic",
        "",
        "This is a development diagnostic on the fixed MedQA300 set, not a held-out result.",
        "The item-cluster bootstrap retains all replications and both directions.",
        "",
        "| Condition | Acc | CR | PR | SI | SRA | FCS | FWS | FS | Rescue/Destroy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = pooled[condition]
        lines.append(
            f"| {DISPLAY[condition]} | {pct(values['accuracy'])} | {pct(values['cr'])} | "
            f"{pct(values['pr'])} | {pct(values['si'])} | {pct(values['sra'])} | "
            f"{pct(values['fcs'])} | {pct(values['fws'])} | "
            f"{pct(values['follow_selectivity'])} | {values['rescues']}/{values['destructions']} |"
        )
    lines.extend(["", "## EGR-Contrast vs Full Text", ""])
    for metric, result in comparisons["egr_contrast_vs_true_text"].items():
        low, high = result["ci95"]
        lines.append(
            f"- {metric}: {100 * result['delta']:+.2f} pp "
            f"(95% CI [{100 * low:+.2f}, {100 * high:+.2f}])"
        )
    lines.extend(
        [
            "",
            "## Verdict audit",
            "",
            f"- Disagreements adjudicated: {pooled_audit['disagreements']}",
            f"- One-correct disagreements: {pooled_audit['one_correct_disagreements']}",
            f"- Correct source selected: {pooled_audit['one_correct_selected_correct']}/"
            f"{pooled_audit['one_correct_disagreements']} "
            f"({pct(pooled_audit['one_correct_selection_accuracy'])})",
            f"- Source A/B selected: {pooled_audit['selected_A']}/{pooled_audit['selected_B']}",
            f"- Third/invalid verdict: {pooled_audit['selected_third']}/"
            f"{pooled_audit['selected_invalid']}",
            f"- Both-wrong disagreements selecting a third answer: "
            f"{pooled_audit['both_wrong_selected_third']}/"
            f"{pooled_audit['both_wrong_disagreements']}",
        ]
    )
    (cli.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
