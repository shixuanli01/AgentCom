"""Pooled MedQA300 analysis for the EGR permutation-invariance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from icr.analysis_v2 import ci95, condition_bootstrap, condition_metrics
from icr.protocol import atomic_write_json

from .analyze_contrast import METRICS, baseline_path, read_jsonl, write_csv


CONDITIONS = ("none", "true_text", "true_statebridge", "egr_contrast", "egr_permutation_gate")
DISPLAY = {
    "none": "None",
    "true_text": "Full Text",
    "true_statebridge": "StateBridge",
    "egr_contrast": "EGR-Contrast",
    "egr_permutation_gate": "EGR-Permutation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EGR permutation gate")
    parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--contrast-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--permutation-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    return parser.parse_args()


def permutation_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        "items": len(rows),
        "checked_disagreements": 0,
        "order_stable": 0,
        "order_unstable": 0,
        "one_correct_stable": 0,
        "one_correct_stable_correct": 0,
        "one_correct_unstable": 0,
        "unstable_original_correct": 0,
        "unstable_original_wrong": 0,
        "both_wrong_unstable": 0,
    }
    for row in rows:
        if row["decision_reason"] != "permutation_check":
            continue
        result["checked_disagreements"] += 1
        stable = bool(row["order_stable"])
        result["order_stable" if stable else "order_unstable"] += 1
        one_correct = bool(row["correct_A"]) != bool(row["correct_B"])
        if not one_correct:
            result["both_wrong_unstable"] += int(not stable)
            continue
        original_correct = (
            row["original_verdict"] == row["answer_A"] and bool(row["correct_A"])
        ) or (
            row["original_verdict"] == row["answer_B"] and bool(row["correct_B"])
        )
        if stable:
            result["one_correct_stable"] += 1
            result["one_correct_stable_correct"] += int(original_correct)
        else:
            result["one_correct_unstable"] += 1
            result["unstable_original_correct" if original_correct else "unstable_original_wrong"] += 1
    result["stable_one_correct_accuracy"] = (
        result["one_correct_stable_correct"] / result["one_correct_stable"]
        if result["one_correct_stable"] else None
    )
    return result


def main() -> None:
    cli = parse_args()
    count = len(cli.source_roots)
    if len(cli.contrast_roots) != count or len(cli.permutation_roots) != count:
        raise ValueError("All root lists must have the same length")
    all_rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    per_seed: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    runtimes: dict[str, Any] = {}

    for source_root, contrast_root, permutation_root in zip(
        cli.source_roots, cli.contrast_roots, cli.permutation_roots
    ):
        frozen = read_jsonl(baseline_path(source_root))
        contrast = read_jsonl(contrast_root / "revisions" / "egr_contrast.jsonl")
        permutation = read_jsonl(
            permutation_root / "revisions" / "egr_permutation_gate.jsonl"
        )
        seed_rows = {
            condition: [row for row in frozen if row.get("condition") == condition]
            for condition in CONDITIONS[:3]
        }
        seed_rows["egr_contrast"] = contrast
        seed_rows["egr_permutation_gate"] = permutation
        ids = {str(row["replication_id"]) for row in permutation}
        if len(ids) != 1:
            raise RuntimeError(f"Ambiguous replication in {permutation_root}")
        replication_id = ids.pop()
        for condition, rows in seed_rows.items():
            if len(rows) != 600:
                raise RuntimeError(f"{replication_id}/{condition}: expected 600, got {len(rows)}")
            all_rows[condition].extend(rows)
        per_seed[replication_id] = {
            condition: condition_metrics(rows) for condition, rows in seed_rows.items()
        }
        audit = permutation_audit(read_jsonl(permutation_root / "verdicts" / "merged.jsonl"))
        audit["replication_id"] = replication_id
        audits.append(audit)
        runtimes[replication_id] = json.loads(
            (permutation_root / "run_status.json").read_text(encoding="utf-8")
        )

    item_ids = list(range(300))
    rng = np.random.default_rng(cli.bootstrap_seed)
    samples = rng.integers(0, 300, size=(cli.num_bootstrap, 300))
    pooled = {condition: condition_metrics(rows) for condition, rows in all_rows.items()}
    boot = {
        condition: condition_bootstrap(rows, item_ids, samples)
        for condition, rows in all_rows.items()
    }
    pairwise: dict[str, Any] = {}
    for baseline in ("true_text", "egr_contrast", "true_statebridge", "none"):
        key = f"egr_permutation_gate_vs_{baseline}"
        pairwise[key] = {}
        for metric in METRICS:
            pairwise[key][metric] = {
                "delta": pooled["egr_permutation_gate"][metric] - pooled[baseline][metric],
                "ci95": ci95(
                    boot["egr_permutation_gate"][metric] - boot[baseline][metric]
                ),
            }

    pooled_audit = permutation_audit(
        [
            row
            for root in cli.permutation_roots
            for row in read_jsonl(root / "verdicts" / "merged.jsonl")
        ]
    )
    summary = {
        "protocol": "EGR-PERMUTATION-GATE-v1",
        "label": "MedQA300 development diagnostic; not held-out",
        "items": 300,
        "replications": count,
        "directional_records_per_condition": 600 * count,
        "pooled_conditions": pooled,
        "per_seed": per_seed,
        "pairwise": pairwise,
        "permutation_audit": {"pooled": pooled_audit, "per_seed": audits},
        "runtime": runtimes,
        "bootstrap": {
            "iterations": cli.num_bootstrap,
            "seed": cli.bootstrap_seed,
            "cluster": "item_id",
            "retains_all_replications_and_both_directions": True,
        },
    }
    cli.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cli.output_root / "summary.json", summary)
    write_csv(
        cli.output_root / "metrics.csv",
        [
            {"condition": condition, "display": DISPLAY[condition], **pooled[condition]}
            for condition in CONDITIONS
        ],
    )
    write_csv(cli.output_root / "permutation_audit.csv", audits)

    def pct(value: float | None) -> str:
        return "--" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# EGR permutation gate: MedQA300 three-seed diagnostic",
        "",
        "Development diagnostic only; not held-out.",
        "",
        "| Condition | Acc | CR | PR | SI | FCS | FWS | FS | Rescue/Destroy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        m = pooled[condition]
        lines.append(
            f"| {DISPLAY[condition]} | {pct(m['accuracy'])} | {pct(m['cr'])} | "
            f"{pct(m['pr'])} | {pct(m['si'])} | {pct(m['fcs'])} | {pct(m['fws'])} | "
            f"{pct(m['follow_selectivity'])} | {m['rescues']}/{m['destructions']} |"
        )
    for baseline in ("true_text", "egr_contrast"):
        lines.extend(["", f"## EGR-Permutation vs {DISPLAY[baseline]}", ""])
        for metric, result in pairwise[f"egr_permutation_gate_vs_{baseline}"].items():
            low, high = result["ci95"]
            lines.append(
                f"- {metric}: {100 * result['delta']:+.2f} pp "
                f"(95% CI [{100 * low:+.2f}, {100 * high:+.2f}])"
            )
    lines.extend(
        [
            "",
            "## Permutation audit",
            "",
            f"- Checked disagreements: {pooled_audit['checked_disagreements']}",
            f"- Stable/unstable: {pooled_audit['order_stable']}/{pooled_audit['order_unstable']}",
            f"- One-correct unstable: {pooled_audit['one_correct_unstable']}",
            f"- Unstable original correct/wrong: {pooled_audit['unstable_original_correct']}/"
            f"{pooled_audit['unstable_original_wrong']}",
            f"- Stable one-correct accuracy: {pct(pooled_audit['stable_one_correct_accuracy'])}",
        ]
    )
    (cli.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
