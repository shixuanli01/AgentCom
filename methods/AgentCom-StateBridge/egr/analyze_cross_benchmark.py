"""Produce one exact-count report for cross-benchmark ICR and EGR runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from icr.analysis import exact_mcnemar_p
from icr.analysis import condition_metrics
from icr.protocol import atomic_write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def paired_comparison(
    reference: list[Mapping[str, Any]], candidate: list[Mapping[str, Any]]
) -> dict[str, Any]:
    def keyed(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, str], bool]:
        return {
            (int(row["item_id"]), str(row["direction"])): bool(row["receiver_post_correct"])
            for row in rows
        }

    left, right = keyed(reference), keyed(candidate)
    if set(left) != set(right):
        raise RuntimeError("Paired conditions do not contain identical directional cases")
    gained = sum(not left[key] and right[key] for key in left)
    lost = sum(left[key] and not right[key] for key in left)
    return {
        "candidate_minus_reference_correct": gained - lost,
        "gained": gained,
        "lost": lost,
        "ties": len(left) - gained - lost,
        "delta_accuracy": (gained - lost) / len(left),
        "mcnemar_exact_p": exact_mcnemar_p(gained, lost),
    }


def collect_egr_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted((root / "revisions").glob("*.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--egr-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--output-root", type=Path)
    cli = parser.parse_args()

    config = json.loads((cli.source_root / "config.json").read_text(encoding="utf-8"))
    selected_ids = [int(value) for value in config["selected_item_ids"]]
    prebeliefs = read_jsonl(cli.source_root / "prebeliefs" / "merged.jsonl")
    baseline_rows = read_jsonl(cli.source_root / "revisions" / "merged.jsonl")
    all_rows = baseline_rows + collect_egr_rows(cli.egr_roots)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["condition"])].append(row)

    expected = len(selected_ids) * 2
    for condition, rows in grouped.items():
        keys = {(int(row["item_id"]), str(row["direction"])) for row in rows}
        if len(rows) != expected or len(keys) != expected:
            raise RuntimeError(
                f"Condition {condition} is incomplete: rows={len(rows)}, expected={expected}"
            )

    metrics = {condition: condition_metrics(rows) for condition, rows in sorted(grouped.items())}
    comparisons: dict[str, Any] = {}
    for reference_name in ("none", "true_text"):
        if reference_name not in grouped:
            continue
        for condition, rows in grouped.items():
            if condition == reference_name:
                continue
            comparisons[f"{condition}_vs_{reference_name}"] = paired_comparison(
                grouped[reference_name], rows
            )

    pre_correct = sum(bool(row["correct"]) for row in prebeliefs)
    by_item: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prebeliefs:
        by_item[int(row["item_id"])].append(row)
    prebelief_summary = {
        "records": len(prebeliefs),
        "correct": pre_correct,
        "accuracy": pre_correct / len(prebeliefs),
        "agreement_items": sum(
            len({row.get("parsed_answer") for row in rows}) == 1
            for rows in by_item.values()
        ),
        "oracle2_correct_items": sum(any(bool(row["correct"]) for row in rows) for rows in by_item.values()),
        "items": len(by_item),
    }
    report = {
        "benchmark": config.get("benchmark", config.get("dataset")),
        "task": config.get("dataset"),
        "replication_id": config.get("replication_id", "seed_pair_00"),
        "selected_item_ids": selected_ids,
        "selection": config.get("selection"),
        "prebeliefs": prebelief_summary,
        "conditions": metrics,
        "paired_comparisons": comparisons,
    }
    output = cli.output_root or cli.source_root / "cross_benchmark_analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "report.json", report)

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    lines = [
        f"# {report['benchmark']} single-seed cross-benchmark report",
        "",
        f"Items: {len(selected_ids)}; prebelief accuracy: {pct(prebelief_summary['accuracy'])}; "
        f"oracle-2: {prebelief_summary['oracle2_correct_items']}/{prebelief_summary['items']}.",
        "",
        "| Condition | Correct | Accuracy | CR | PR | Rescue | Destroy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in metrics.items():
        lines.append(
            f"| {condition} | {values['post_correct']}/{values['directional_examples']} | "
            f"{pct(values['post_accuracy'])} | {pct(values['cr'])} | {pct(values['pr'])} | "
            f"{values['rescues']} | {values['destructions']} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, values in comparisons.items():
        lines.append(
            f"- {name}: delta={pct(values['delta_accuracy'])}, gained={values['gained']}, "
            f"lost={values['lost']}, exact p={values['mcnemar_exact_p']:.6g}."
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
