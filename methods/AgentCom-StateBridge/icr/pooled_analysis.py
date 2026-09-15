"""Pooled five-seed MedQA300 analysis with item-clustered uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis_v2 import (
    PAIRWISE_METRICS,
    PRIMARY_METRICS,
    atomic_write_csv,
    atomic_write_parquet,
    ci95,
    condition_bootstrap,
    condition_metrics,
    fmt,
    fmt_ci,
    safe_delta,
)
from .protocol import atomic_write_json


CONDITIONS = (
    "none",
    "true_text",
    "self_text",
    "other_text",
    "true_statebridge",
    "self_statebridge",
    "other_statebridge",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument(
        "--replications",
        nargs="+",
        default=[f"seed_pair_{index:02d}" for index in range(5)],
    )
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--expected-items", type=int, default=300)
    cli = parser.parse_args()

    base_root = cli.base_root.resolve()
    output = base_root / "pooled_5seed"
    all_rows: list[dict[str, Any]] = []
    pooled_cases = []
    per_seed_metrics = []
    summaries: dict[str, dict[str, Any]] = {}

    for replication_id in cli.replications:
        root = base_root / replication_id
        summary_path = root / "analysis_v2/summary.json"
        revision_path = root / "revisions/merged.jsonl"
        case_path = root / "analysis_v2/per_direction_cases.parquet"
        if not summary_path.is_file() or not revision_path.is_file() or not case_path.is_file():
            raise RuntimeError(f"Incomplete replication artifact: {root}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[replication_id] = summary
        for condition in CONDITIONS:
            if condition not in summary["conditions"]:
                raise RuntimeError(f"{replication_id} is missing {condition}")
            values = summary["conditions"][condition]
            per_seed_metrics.append(
                {
                    "seed_pair": replication_id,
                    "condition": condition,
                    **{metric: values[metric] for metric in PRIMARY_METRICS},
                }
            )
        for line in revision_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                row["replication_id"] = replication_id
                all_rows.append(row)
        pooled_cases.append(pd.read_parquet(case_path))

    expected = len(cli.replications) * cli.expected_items * 2 * len(CONDITIONS)
    keys = {
        (row["replication_id"], row["item_id"], row["direction"], row["condition"])
        for row in all_rows
    }
    if len(all_rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"Pooled records are incomplete/non-unique: rows={len(all_rows)} unique={len(keys)} expected={expected}"
        )

    by_condition = {
        condition: [row for row in all_rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    item_ids = sorted({int(row["item_id"]) for row in all_rows})
    if len(item_ids) != cli.expected_items:
        raise RuntimeError(
            f"Expected {cli.expected_items} pooled item IDs, found {len(item_ids)}"
        )
    rng = np.random.default_rng(cli.bootstrap_seed)
    samples = rng.integers(0, len(item_ids), (cli.num_bootstrap, len(item_ids)))
    metrics = {
        condition: condition_metrics(rows) for condition, rows in by_condition.items()
    }
    boot = {
        condition: condition_bootstrap(rows, item_ids, samples)
        for condition, rows in by_condition.items()
    }
    pooled_metric_rows = []
    for condition in CONDITIONS:
        record: dict[str, Any] = {"condition": condition, **metrics[condition]}
        for metric in PRIMARY_METRICS:
            interval = ci95(boot[condition][metric])
            record[f"{metric}_ci95_low"] = interval[0] if interval else None
            record[f"{metric}_ci95_high"] = interval[1] if interval else None
        pooled_metric_rows.append(record)

    pairwise_rows = []
    for replication_id in cli.replications:
        values = summaries[replication_id]["conditions"]
        for metric in PAIRWISE_METRICS:
            pairwise_rows.append(
                {
                    "scope": replication_id,
                    "metric": metric,
                    "text": values["true_text"][metric],
                    "statebridge": values["true_statebridge"][metric],
                    "text_minus_statebridge": safe_delta(
                        values["true_text"][metric],
                        values["true_statebridge"][metric],
                    ),
                    "ci95_low": None,
                    "ci95_high": None,
                }
            )
    pooled_pairwise: dict[str, Any] = {}
    for metric in PAIRWISE_METRICS:
        values = boot["true_text"][metric] - boot["true_statebridge"][metric]
        interval = ci95(values)
        estimate = safe_delta(
            metrics["true_text"][metric], metrics["true_statebridge"][metric]
        )
        pooled_pairwise[metric] = {"point_estimate": estimate, "ci95": interval}
        pairwise_rows.append(
            {
                "scope": "pooled_item_cluster",
                "metric": metric,
                "text": metrics["true_text"][metric],
                "statebridge": metrics["true_statebridge"][metric],
                "text_minus_statebridge": estimate,
                "ci95_low": interval[0] if interval else None,
                "ci95_high": interval[1] if interval else None,
            }
        )

    across_seed_rows = []
    for condition in CONDITIONS:
        rows = [row for row in per_seed_metrics if row["condition"] == condition]
        for metric in PRIMARY_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in rows if row[metric] is not None],
                dtype=np.float64,
            )
            across_seed_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else None,
                    "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                    "seed_pairs": len(values),
                }
            )

    hierarchical = {
        "metadata": {
            "method": "item-cluster bootstrap",
            "cluster": "item_id",
            "included_with_each_sampled_item": "all directions and all replication IDs",
            "iterations": cli.num_bootstrap,
            "seed": cli.bootstrap_seed,
            "replications": list(cli.replications),
        },
        "conditions": {
            condition: {
                metric: {
                    "point_estimate": metrics[condition][metric],
                    "ci95": ci95(boot[condition][metric]),
                }
                for metric in PRIMARY_METRICS
            }
            for condition in CONDITIONS
        },
        "text_minus_statebridge": pooled_pairwise,
    }

    cases = pd.concat(pooled_cases, ignore_index=True)
    valid_disagreement = (
        cases["sender_answer"].notna()
        & cases["receiver_pre_answer"].notna()
        & (cases["sender_answer"] != cases["receiver_pre_answer"])
    )
    disagreement_cases = cases[valid_disagreement].copy()
    sender_following_cases = cases[
        valid_disagreement
        & cases["pre_category"].isin(["correction_opportunity", "destruction_risk"])
    ].copy()

    paper_rows = []
    channel_conditions = {
        "None": "none",
        "Text": "true_text",
        "StateBridge": "true_statebridge",
    }
    for scope in (*cli.replications, "pooled_item_cluster"):
        for channel, condition in channel_conditions.items():
            if scope == "pooled_item_cluster":
                values = metrics[condition]
            else:
                values = summaries[scope]["conditions"][condition]
            paper_rows.append(
                {
                    "scope": scope,
                    "channel": channel,
                    **{metric: values[metric] for metric in PRIMARY_METRICS},
                }
            )

    pattern_rows = [
        {
            "seed_pair": replication_id,
            "statebridge_cr": summaries[replication_id]["conditions"][
                "true_statebridge"
            ]["cr"],
            "statebridge_pr": summaries[replication_id]["conditions"][
                "true_statebridge"
            ]["pr"],
            "text_cr": summaries[replication_id]["conditions"]["true_text"]["cr"],
            "text_pr": summaries[replication_id]["conditions"]["true_text"]["pr"],
        }
        for replication_id in cli.replications
    ]
    robustness = {
        "statebridge_cr_exceeds_pr_all_seeds": all(
            row["statebridge_cr"] is not None
            and row["statebridge_pr"] is not None
            and row["statebridge_cr"] > row["statebridge_pr"]
            for row in pattern_rows
        ),
        "text_pr_exceeds_statebridge_pr_all_seeds": all(
            row["text_pr"] is not None
            and row["statebridge_pr"] is not None
            and row["text_pr"] > row["statebridge_pr"]
            for row in pattern_rows
        ),
        "per_seed": pattern_rows,
    }
    summary = {
        "status": "complete",
        "step": 2,
        "replications": list(cli.replications),
        "directional_condition_records": len(all_rows),
        "pooled_conditions": metrics,
        "text_minus_statebridge": pooled_pairwise,
        "robustness": robustness,
        "bootstrap": hierarchical["metadata"],
    }

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "hierarchical_bootstrap.json", hierarchical)
    atomic_write_csv(output / "per_seed_metrics.csv", per_seed_metrics)
    atomic_write_csv(output / "pooled_metrics.csv", pooled_metric_rows)
    atomic_write_csv(output / "pairwise_text_vs_statebridge.csv", pairwise_rows)
    atomic_write_csv(output / "mean_std_across_seeds.csv", across_seed_rows)
    atomic_write_csv(output / "paper_table_medqa300.csv", paper_rows)
    atomic_write_parquet(output / "pooled_cases.parquet", cases.to_dict("records"))
    atomic_write_parquet(
        output / "disagreement_cases.parquet", disagreement_cases.to_dict("records")
    )
    atomic_write_parquet(
        output / "sender_following_cases.parquet",
        sender_following_cases.to_dict("records"),
    )

    lines = [
        "# ICR MedQA300 — pooled five-seed analysis",
        "",
        f"Pooled {len(cli.replications)} seed pairs and {len(all_rows):,} directional-condition records. Confidence intervals resample item IDs and retain all replications and both directions.",
        "",
        "| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, condition in channel_conditions.items():
        value = metrics[condition]
        lines.append(
            f"| {label} | {fmt(value['accuracy'])} | {fmt(value['cr'])} | {fmt(value['pr'])} | "
            f"{fmt(value['si'])} | {fmt(value['sra'])} | {fmt(value['fcs'])} | "
            f"{fmt(value['fws'])} | {fmt(value['follow_selectivity'])} |"
        )
    lines.extend(["", "## Text − StateBridge", ""])
    for metric in PAIRWISE_METRICS:
        value = pooled_pairwise[metric]
        lines.append(
            f"- {metric}: {fmt(value['point_estimate'])}, 95% CI {fmt_ci(value['ci95'])}"
        )
    lines.extend(
        [
            "",
            "## Robustness checks",
            "",
            f"- StateBridge CR > PR in every seed pair: {robustness['statebridge_cr_exceeds_pr_all_seeds']}",
            f"- Text PR > StateBridge PR in every seed pair: {robustness['text_pr_exceeds_statebridge_pr_all_seeds']}",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
