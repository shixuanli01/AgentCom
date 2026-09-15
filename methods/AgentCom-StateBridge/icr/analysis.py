"""Item-clustered analysis for Independent -> Communicate -> Revise."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from agentcom.multipath import exact_mcnemar_p

from .merge import merge_revisions
from .protocol import atomic_write_json


RATE_KEYS = ("post_accuracy", "cr", "pr", "sr", "scr")
CATEGORY_FOR_RATE = {
    "cr": "correction_opportunity",
    "pr": "destruction_risk",
    "sr": "both_wrong",
    "scr": "both_correct",
}


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def condition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    post_correct = sum(bool(row["receiver_post_correct"]) for row in rows)
    pre_correct = sum(bool(row["receiver_pre_correct"]) for row in rows)
    result: dict[str, Any] = {
        "directional_examples": total,
        "pre_correct": pre_correct,
        "pre_accuracy": _safe_rate(pre_correct, total),
        "post_correct": post_correct,
        "post_accuracy": _safe_rate(post_correct, total),
    }
    for metric, category in CATEGORY_FOR_RATE.items():
        subset = [row for row in rows if row["pair_classification"] == category]
        numerator = sum(bool(row["receiver_post_correct"]) for row in subset)
        result[f"{metric}_numerator"] = numerator
        result[f"{metric}_denominator"] = len(subset)
        result[metric] = _safe_rate(numerator, len(subset))
    result["dr"] = 1.0 - result["pr"] if result["pr"] is not None else None

    rescues = sum(
        not bool(row["receiver_pre_correct"]) and bool(row["receiver_post_correct"])
        for row in rows
    )
    destructions = sum(
        bool(row["receiver_pre_correct"]) and not bool(row["receiver_post_correct"])
        for row in rows
    )
    changed = [row for row in rows if bool(row["answer_changed"])]
    result.update(
        {
            "rescues": rescues,
            "destructions": destructions,
            "net_correction": rescues - destructions,
            "answer_changes": len(changed),
            "answer_change_rate": _safe_rate(len(changed), total),
            "beneficial_changes": sum(
                not bool(row["receiver_pre_correct"])
                and bool(row["receiver_post_correct"])
                for row in changed
            ),
            "harmful_changes": sum(
                bool(row["receiver_pre_correct"])
                and not bool(row["receiver_post_correct"])
                for row in changed
            ),
            "neutral_changes": sum(
                bool(row["receiver_pre_correct"])
                == bool(row["receiver_post_correct"])
                for row in changed
            ),
            "invalid_outputs": sum(row["receiver_post_answer"] is None for row in rows),
        }
    )

    follow_correct_pool = [
        row
        for row in rows
        if row["pair_classification"] == "correction_opportunity"
        and row["sender_pre_answer"] is not None
        and row["receiver_pre_answer"] is not None
        and row["sender_pre_answer"] != row["receiver_pre_answer"]
    ]
    follow_wrong_pool = [
        row
        for row in rows
        if row["pair_classification"] == "destruction_risk"
        and row["sender_pre_answer"] is not None
        and row["receiver_pre_answer"] is not None
        and row["sender_pre_answer"] != row["receiver_pre_answer"]
    ]
    result.update(
        {
            "follow_correct_sender_numerator": sum(
                row["receiver_post_answer"] == row["sender_pre_answer"]
                for row in follow_correct_pool
            ),
            "follow_correct_sender_denominator": len(follow_correct_pool),
            "follow_correct_sender_rate": _safe_rate(
                sum(
                    row["receiver_post_answer"] == row["sender_pre_answer"]
                    for row in follow_correct_pool
                ),
                len(follow_correct_pool),
            ),
            "follow_wrong_sender_numerator": sum(
                row["receiver_post_answer"] == row["sender_pre_answer"]
                for row in follow_wrong_pool
            ),
            "follow_wrong_sender_denominator": len(follow_wrong_pool),
            "follow_wrong_sender_rate": _safe_rate(
                sum(
                    row["receiver_post_answer"] == row["sender_pre_answer"]
                    for row in follow_wrong_pool
                ),
                len(follow_wrong_pool),
            ),
        }
    )
    payloads = [row.get("communication_payload") or {} for row in rows]
    for key in ("tokens", "characters", "states", "hidden_dimension", "payload_bytes"):
        values = [float(payload[key]) for payload in payloads if payload.get(key) is not None]
        result[f"mean_payload_{key}"] = sum(values) / len(values) if values else None
    result["mean_generation_seconds"] = (
        sum(float(row["generation_seconds"]) for row in rows) / total if total else None
    )
    return result


def metric_fraction(row: Mapping[str, Any], metric: str) -> tuple[int, int]:
    if metric == "post_accuracy":
        return int(bool(row["receiver_post_correct"])), 1
    category = CATEGORY_FOR_RATE[metric]
    eligible = row["pair_classification"] == category
    return int(eligible and bool(row["receiver_post_correct"])), int(eligible)


def _item_sums(
    rows: Sequence[Mapping[str, Any]], item_ids: Sequence[int], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    position = {item_id: index for index, item_id in enumerate(item_ids)}
    numerator = np.zeros(len(item_ids), dtype=np.int64)
    denominator = np.zeros(len(item_ids), dtype=np.int64)
    for row in rows:
        num, den = metric_fraction(row, metric)
        index = position[int(row["item_id"])]
        numerator[index] += num
        denominator[index] += den
    return numerator, denominator


def _bootstrap_rates(
    numerator: np.ndarray, denominator: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    nums = numerator[samples].sum(axis=1)
    dens = denominator[samples].sum(axis=1)
    return np.divide(
        nums,
        dens,
        out=np.full(nums.shape, np.nan, dtype=np.float64),
        where=dens != 0,
    )


def _ci(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    low, high = np.quantile(finite, (0.025, 0.975))
    return [float(low), float(high)]


def clustered_bootstrap(
    by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    item_ids = sorted(
        {int(row["item_id"]) for rows in by_condition.values() for row in rows}
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(item_ids), size=(num_bootstrap, len(item_ids)))
    rates: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    result: dict[str, Any] = {"num_bootstrap": num_bootstrap, "cluster": "item"}
    result["conditions"] = {}
    for condition, rows in by_condition.items():
        result["conditions"][condition] = {}
        for metric in RATE_KEYS:
            numerator, denominator = _item_sums(rows, item_ids, metric)
            values = _bootstrap_rates(numerator, denominator, samples)
            rates[condition][metric] = values
            result["conditions"][condition][metric] = {"ci95": _ci(values)}

    comparisons = {
        "text_ce": ("true_text", "none"),
        "text_esv": ("true_text", "other_text"),
        "text_oav": ("true_text", "self_text"),
        "statebridge_ce": ("true_statebridge", "none"),
        "statebridge_esv": ("true_statebridge", "other_statebridge"),
        "statebridge_oav": ("true_statebridge", "self_statebridge"),
    }
    result["comparisons"] = {}
    for name, (primary, baseline) in comparisons.items():
        if primary not in rates or baseline not in rates:
            continue
        result["comparisons"][name] = {}
        for metric in RATE_KEYS:
            difference = rates[primary][metric] - rates[baseline][metric]
            result["comparisons"][name][metric] = {"ci95": _ci(difference)}
    return result


def paired_comparison(
    primary_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary = {
        (int(row["item_id"]), row["direction"]): row for row in primary_rows
    }
    baseline = {
        (int(row["item_id"]), row["direction"]): row for row in baseline_rows
    }
    keys = sorted(set(primary) & set(baseline))
    rescues = sum(
        bool(primary[key]["receiver_post_correct"])
        and not bool(baseline[key]["receiver_post_correct"])
        for key in keys
    )
    destructions = sum(
        not bool(primary[key]["receiver_post_correct"])
        and bool(baseline[key]["receiver_post_correct"])
        for key in keys
    )
    return {
        "paired_examples": len(keys),
        "rescues": rescues,
        "destructions": destructions,
        "delta_accuracy": _safe_rate(rescues - destructions, len(keys)),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(rescues, destructions),
        "mcnemar_note": "secondary direction-level statistic; item-cluster bootstrap is primary",
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ICR revisions")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    cli = parser.parse_args()
    rows = merge_revisions(cli.artifact_root, require_complete=False)
    if not rows:
        raise RuntimeError("No completed revision records found")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    metrics = {
        condition: condition_metrics(condition_rows)
        for condition, condition_rows in sorted(by_condition.items())
    }
    bootstrap = clustered_bootstrap(
        by_condition,
        num_bootstrap=cli.num_bootstrap,
        seed=cli.bootstrap_seed,
    )
    comparison_defs = {
        "text_ce": ("true_text", "none"),
        "text_esv": ("true_text", "other_text"),
        "text_oav": ("true_text", "self_text"),
        "statebridge_ce": ("true_statebridge", "none"),
        "statebridge_esv": ("true_statebridge", "other_statebridge"),
        "statebridge_oav": ("true_statebridge", "self_statebridge"),
    }
    comparisons = []
    for name, (primary, baseline) in comparison_defs.items():
        if primary not in metrics or baseline not in metrics:
            continue
        row: dict[str, Any] = {
            "comparison": name,
            "primary": primary,
            "baseline": baseline,
            **paired_comparison(by_condition[primary], by_condition[baseline]),
        }
        for metric in RATE_KEYS:
            left, right = metrics[primary][metric], metrics[baseline][metric]
            row[f"delta_{metric}"] = (
                left - right if left is not None and right is not None else None
            )
            row[f"{metric}_ci95"] = (
                bootstrap.get("comparisons", {})
                .get(name, {})
                .get(metric, {})
                .get("ci95")
            )
        comparisons.append(row)

    analysis_dir = cli.artifact_root / "analysis"
    atomic_write_json(
        analysis_dir / "summary.json",
        {"conditions": metrics, "comparisons": comparisons},
    )
    atomic_write_json(analysis_dir / "bootstrap_results.json", bootstrap)
    metric_fields = ["condition", *next(iter(metrics.values())).keys()]
    write_csv(
        analysis_dir / "per_condition_metrics.csv",
        [{"condition": key, **value} for key, value in metrics.items()],
        metric_fields,
    )
    comparison_fields = sorted({key for row in comparisons for key in row})
    write_csv(analysis_dir / "paired_comparisons.csv", comparisons, comparison_fields)
    write_csv(
        analysis_dir / "correction_preservation.csv",
        [
            {
                "condition": condition,
                "correction_rate": value["cr"],
                "preservation_rate": value["pr"],
            }
            for condition, value in metrics.items()
        ],
        ("condition", "correction_rate", "preservation_rate"),
    )

    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["item_id"]), row["direction"])
        case = grouped.setdefault(
            key,
            {
                "item_id": key[0],
                "direction": key[1],
                "sender_pre_answer": row["sender_pre_answer"],
                "sender_correct": row["sender_correct"],
                "receiver_pre_answer": row["receiver_pre_answer"],
                "receiver_pre_correct": row["receiver_pre_correct"],
                "classification": row["pair_classification"],
            },
        )
        prefix = row["condition"]
        case[f"{prefix}_post_answer"] = row["receiver_post_answer"]
        case[f"{prefix}_post_correct"] = row["receiver_post_correct"]
        case[f"{prefix}_changed"] = row["answer_changed"]
        case[f"{prefix}_followed_sender"] = row["followed_sender"]
    case_fields = sorted({field for case in grouped.values() for field in case})
    write_csv(
        analysis_dir / "conditional_cases.csv",
        [grouped[key] for key in sorted(grouped)],
        case_fields,
    )

    lines = [
        "# ICR Analysis",
        "",
        "Item-cluster bootstrap is the primary uncertainty estimate; McNemar is secondary.",
        "",
        "| Condition | Post Acc | CR | PR | DR | SR | SCR | Rescue | Destroy | Change Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, value in metrics.items():
        lines.append(
            f"| {condition} | {_fmt(value['post_accuracy'])} | {_fmt(value['cr'])} | "
            f"{_fmt(value['pr'])} | {_fmt(value['dr'])} | {_fmt(value['sr'])} | "
            f"{_fmt(value['scr'])} | {value['rescues']} | {value['destructions']} | "
            f"{_fmt(value['answer_change_rate'])} |"
        )
    lines.extend(["", "## Causal comparisons", ""])
    for comparison in comparisons:
        lines.append(
            f"- {comparison['comparison']}: ΔAcc={_fmt(comparison['delta_post_accuracy'])}, "
            f"rescues={comparison['rescues']}, destructions={comparison['destructions']}."
        )
    lines.extend(
        [
            "",
            "Interpret True≫Other as example-specific information; True≈Other as possible message-presence/perturbation effect; True≈Self as little unique other-agent value. High CR with low PR is aggressive revision; high PR with low CR is conservative revision.",
            "",
        ]
    )
    (analysis_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
