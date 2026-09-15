"""Pooled Text/StateBridge/LatentMAS analysis for completed ICR replications."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .analysis_v2 import (
    PAIRWISE_METRICS,
    PRIMARY_METRICS,
    atomic_write_csv,
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
    "true_latentmas",
    "self_latentmas",
    "other_latentmas",
)
CHANNELS = {
    "None": "none",
    "Text": "true_text",
    "StateBridge": "true_statebridge",
    "LatentMAS": "true_latentmas",
}
CONTROLS = {
    "text": ("true_text", "self_text", "other_text"),
    "statebridge": (
        "true_statebridge",
        "self_statebridge",
        "other_statebridge",
    ),
    "latentmas": ("true_latentmas", "self_latentmas", "other_latentmas"),
}


def _record_paths(root: Path) -> Iterable[Path]:
    yield from root.glob("revisions/rank*/records/item_*.json")
    yield from root.glob("analysis/revisions/rank*/records/item_*.json")


def load_replication(root: Path, replication_id: str) -> list[dict[str, Any]]:
    rows: dict[tuple[int, str, str], dict[str, Any]] = {}
    for path in sorted(_record_paths(root)):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") != "complete" or row.get("condition") not in CONDITIONS:
            continue
        key = (int(row["item_id"]), str(row["direction"]), str(row["condition"]))
        if key in rows:
            comparable = {
                name: row.get(name)
                for name in (
                    "receiver_post_answer",
                    "receiver_post_correct",
                    "revision_seed",
                    "config_fingerprint",
                )
            }
            prior = {name: rows[key].get(name) for name in comparable}
            if comparable != prior:
                raise RuntimeError(f"Conflicting records for {replication_id} {key}")
            continue
        row["replication_id"] = replication_id
        rows[key] = row
    expected = 300 * 2 * len(CONDITIONS)
    if len(rows) != expected:
        counts: dict[str, int] = defaultdict(int)
        for _, _, condition in rows:
            counts[condition] += 1
        raise RuntimeError(
            f"Incomplete {replication_id}: have {len(rows)}/{expected}; "
            f"conditions={dict(sorted(counts.items()))}"
        )
    return [rows[key] for key in sorted(rows)]


def mean_present(rows: list[Mapping[str, Any]], accessor) -> float | None:
    values = [float(value) for row in rows if (value := accessor(row)) is not None]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--replications", nargs="+", required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    cli = parser.parse_args()

    base = cli.base_root.resolve()
    output = base / (
        cli.output_name or f"pooled_{len(cli.replications)}seed_with_latentmas"
    )
    by_seed: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for replication_id in cli.replications:
        rows = load_replication(base / replication_id, replication_id)
        by_seed[replication_id] = rows
        all_rows.extend(rows)

    grouped = {
        condition: [row for row in all_rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    item_ids = list(range(300))
    rng = np.random.default_rng(cli.bootstrap_seed)
    samples = rng.integers(0, len(item_ids), (cli.num_bootstrap, len(item_ids)))
    metrics = {
        condition: condition_metrics(rows) for condition, rows in grouped.items()
    }
    boot = {
        condition: condition_bootstrap(rows, item_ids, samples)
        for condition, rows in grouped.items()
    }

    per_seed_rows = []
    for replication_id, seed_rows in by_seed.items():
        for condition in CONDITIONS:
            values = condition_metrics(
                [row for row in seed_rows if row["condition"] == condition]
            )
            per_seed_rows.append(
                {
                    "seed_pair": replication_id,
                    "condition": condition,
                    **{metric: values[metric] for metric in PRIMARY_METRICS},
                    "answer_change_rate": values["answer_change_rate"],
                    "invalid_outputs": values["invalid_outputs"],
                }
            )

    bootstrap_conditions = {
        condition: {
            metric: {
                "point_estimate": metrics[condition][metric],
                "ci95": ci95(boot[condition][metric]),
            }
            for metric in PRIMARY_METRICS
        }
        for condition in CONDITIONS
    }
    pairwise_rows = []
    pairwise_json: dict[str, Any] = {}
    pairs = (
        ("text_minus_statebridge", "true_text", "true_statebridge"),
        ("text_minus_latentmas", "true_text", "true_latentmas"),
        ("statebridge_minus_latentmas", "true_statebridge", "true_latentmas"),
    )
    for name, left, right in pairs:
        pairwise_json[name] = {}
        for metric in PAIRWISE_METRICS:
            interval = ci95(boot[left][metric] - boot[right][metric])
            estimate = safe_delta(metrics[left][metric], metrics[right][metric])
            record = {
                "comparison": name,
                "left": left,
                "right": right,
                "metric": metric,
                "delta": estimate,
                "ci95_low": interval[0] if interval else None,
                "ci95_high": interval[1] if interval else None,
            }
            pairwise_rows.append(record)
            pairwise_json[name][metric] = {
                "point_estimate": estimate,
                "ci95": interval,
            }

    control_rows = []
    control_json: dict[str, Any] = {}
    for modality, (true_condition, self_condition, other_condition) in CONTROLS.items():
        control_json[modality] = {}
        for label, control in (
            ("true_vs_none", "none"),
            ("true_vs_self", self_condition),
            ("true_vs_other", other_condition),
        ):
            control_json[modality][label] = {}
            for metric in PAIRWISE_METRICS:
                interval = ci95(boot[true_condition][metric] - boot[control][metric])
                estimate = safe_delta(
                    metrics[true_condition][metric], metrics[control][metric]
                )
                control_rows.append(
                    {
                        "modality": modality,
                        "comparison": label,
                        "true_condition": true_condition,
                        "control_condition": control,
                        "metric": metric,
                        "delta": estimate,
                        "ci95_low": interval[0] if interval else None,
                        "ci95_high": interval[1] if interval else None,
                    }
                )
                control_json[modality][label][metric] = {
                    "point_estimate": estimate,
                    "ci95": interval,
                }

    paper_rows = []
    for label, condition in CHANNELS.items():
        paper_rows.append(
            {
                "channel": label,
                "condition": condition,
                **{metric: metrics[condition][metric] for metric in PAIRWISE_METRICS},
            }
        )

    compute_rows = []
    for condition, rows in grouped.items():
        compute_rows.append(
            {
                "condition": condition,
                "records": len(rows),
                "mean_revision_generation_seconds": mean_present(
                    rows, lambda row: row.get("generation_seconds")
                ),
                "mean_revision_generated_tokens": mean_present(
                    rows, lambda row: row.get("generation_length")
                ),
                "mean_payload_bytes": mean_present(
                    rows,
                    lambda row: (row.get("communication_payload") or {}).get(
                        "kv_payload_bytes",
                        (row.get("communication_payload") or {}).get("payload_bytes"),
                    ),
                ),
                "mean_communication_seconds": mean_present(
                    rows,
                    lambda row: (row.get("communication_payload") or {}).get(
                        "communication_seconds"
                    ),
                ),
                "mean_latent_steps": mean_present(
                    rows,
                    lambda row: (row.get("communication_payload") or {}).get(
                        "generated_latent_steps"
                    ),
                ),
            }
        )

    statebridge_cr_minus_pr = safe_delta(
        metrics["true_statebridge"]["cr"], metrics["true_statebridge"]["pr"]
    )
    latentmas_cr_minus_pr = safe_delta(
        metrics["true_latentmas"]["cr"], metrics["true_latentmas"]["pr"]
    )
    if statebridge_cr_minus_pr is not None and statebridge_cr_minus_pr > 0:
        if latentmas_cr_minus_pr is not None and latentmas_cr_minus_pr > 0:
            supported_pattern = "A_directional_signature"
        else:
            supported_pattern = "B_directional_signature"
    else:
        supported_pattern = "neither_A_nor_B_directional_signature"
    pattern_evidence = {
        "classification": supported_pattern,
        "criterion": (
            "A when CR>PR for both latent channels; B when CR>PR for "
            "StateBridge but not LatentMAS. This directional criterion does "
            "not by itself assert that either rate is substantively high or low."
        ),
        "statebridge_cr_minus_pr": statebridge_cr_minus_pr,
        "latentmas_cr_minus_pr": latentmas_cr_minus_pr,
    }

    summary = {
        "status": "complete",
        "step": 3,
        "replications": list(cli.replications),
        "directional_condition_records": len(all_rows),
        "conditions": metrics,
        "channel_pairwise": pairwise_json,
        "causal_controls": control_json,
        "pattern_evidence": pattern_evidence,
        "bootstrap": {
            "method": "item-cluster bootstrap",
            "cluster": "item_id",
            "included_per_item": "all selected replications and both directions",
            "iterations": cli.num_bootstrap,
            "seed": cli.bootstrap_seed,
        },
    }
    bootstrap_results = {
        "metadata": summary["bootstrap"],
        "conditions": bootstrap_conditions,
        "channel_pairwise": pairwise_json,
        "causal_controls": control_json,
    }

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "bootstrap_results.json", bootstrap_results)
    atomic_write_csv(output / "paper_table_channels.csv", paper_rows)
    atomic_write_csv(output / "per_seed_metrics.csv", per_seed_rows)
    atomic_write_csv(output / "pairwise_channel_comparisons.csv", pairwise_rows)
    atomic_write_csv(output / "causal_controls.csv", control_rows)
    atomic_write_csv(output / "compute_costs.csv", compute_rows)

    lines = [
        f"# ICR MedQA300 — {len(cli.replications)}-seed channel comparison",
        "",
        f"Complete pooled comparison over {len(all_rows):,} directional-condition records. Confidence intervals use {cli.num_bootstrap:,} item-cluster resamples and retain both directions and every included seed pair.",
        "",
        "| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, condition in CHANNELS.items():
        value = metrics[condition]
        lines.append(
            f"| {label} | {fmt(value['accuracy'])} | {fmt(value['cr'])} | "
            f"{fmt(value['pr'])} | {fmt(value['si'])} | {fmt(value['sra'])} | "
            f"{fmt(value['fcs'])} | {fmt(value['fws'])} | "
            f"{fmt(value['follow_selectivity'])} |"
        )
    lines.extend(["", "## Pairwise channel differences", ""])
    for name, _, _ in pairs:
        lines.append(f"### {name}")
        lines.append("")
        for metric in PAIRWISE_METRICS:
            value = pairwise_json[name][metric]
            lines.append(
                f"- {metric}: {fmt(value['point_estimate'])}, 95% CI {fmt_ci(value['ci95'])}"
            )
        lines.append("")
    lines.extend(
        [
            "## Pattern evidence",
            "",
            f"- Classification: `{supported_pattern}`",
            f"- StateBridge CR−PR: {fmt(statebridge_cr_minus_pr)}",
            f"- LatentMAS CR−PR: {fmt(latentmas_cr_minus_pr)}",
            "- Criterion: A requires CR>PR for both latent channels; B requires CR>PR for StateBridge but not LatentMAS. This is a directional signature, not an unsupported high/low threshold claim.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
