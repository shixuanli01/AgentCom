"""Communication-first Step-1 analysis for the frozen ICR seed-pair artifact."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import transformers

from .protocol import atomic_write_json


PRIMARY_METRICS = (
    "accuracy",
    "cr",
    "pr",
    "sr",
    "scr",
    "si",
    "sra",
    "fcs",
    "fws",
    "follow_selectivity",
)
PAIRWISE_METRICS = (
    "accuracy",
    "cr",
    "pr",
    "si",
    "sra",
    "fcs",
    "fws",
    "follow_selectivity",
)
CATEGORIES = (
    "correction_opportunity",
    "destruction_risk",
    "both_wrong",
    "both_correct",
)
EXPECTED_REGRESSION = {
    "none": {"accuracy": 0.725, "cr": 2 / 35, "pr": 34 / 35},
    "true_text": {"accuracy": 446 / 600, "cr": 32 / 35, "pr": 15 / 35},
    "true_statebridge": {
        "accuracy": 430 / 600,
        "cr": 29 / 35,
        "pr": 2 / 35,
    },
}


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def safe_delta(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def ci95(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    low, high = np.quantile(finite, (0.025, 0.975))
    return [float(low), float(high)]


def _eligible(row: Mapping[str, Any], metric: str) -> bool:
    category = row["pair_classification"]
    if metric == "accuracy":
        return True
    if metric == "cr":
        return category == "correction_opportunity"
    if metric == "pr":
        return category == "destruction_risk"
    if metric == "sr":
        return category == "both_wrong"
    if metric == "scr":
        return category == "both_correct"
    if metric == "sra":
        return bool(row["sender_correct"]) != bool(row["receiver_pre_correct"])
    valid_disagreement = (
        row["sender_pre_answer"] is not None
        and row["receiver_pre_answer"] is not None
        and row["sender_pre_answer"] != row["receiver_pre_answer"]
    )
    if metric == "fcs":
        return category == "correction_opportunity" and valid_disagreement
    if metric == "fws":
        return category == "destruction_risk" and valid_disagreement
    raise KeyError(metric)


def _success(row: Mapping[str, Any], metric: str) -> bool:
    if metric in {"accuracy", "cr", "pr", "sr", "scr", "sra"}:
        return bool(row["receiver_post_correct"])
    if metric in {"fcs", "fws"}:
        return row["receiver_post_answer"] == row["sender_pre_answer"]
    raise KeyError(metric)


def fraction(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[int, int]:
    eligible = [row for row in rows if _eligible(row, metric)]
    return sum(_success(row, metric) for row in eligible), len(eligible)


def condition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"directional_examples": len(rows)}
    for metric in ("accuracy", "cr", "pr", "sr", "scr", "sra", "fcs", "fws"):
        numerator, denominator = fraction(rows, metric)
        result[metric] = safe_rate(numerator, denominator)
        result[f"{metric}_numerator"] = numerator
        result[f"{metric}_denominator"] = denominator
    result["si"] = (
        (result["cr"] + result["pr"]) / 2
        if result["cr"] is not None and result["pr"] is not None
        else None
    )
    result["follow_selectivity"] = (
        result["fcs"] - result["fws"]
        if result["fcs"] is not None and result["fws"] is not None
        else None
    )
    result.update(
        {
            "answer_change_rate": safe_rate(
                sum(bool(row["answer_changed"]) for row in rows), len(rows)
            ),
            "rescues": sum(
                not bool(row["receiver_pre_correct"])
                and bool(row["receiver_post_correct"])
                for row in rows
            ),
            "destructions": sum(
                bool(row["receiver_pre_correct"])
                and not bool(row["receiver_post_correct"])
                for row in rows
            ),
            "invalid_outputs": sum(row["receiver_post_answer"] is None for row in rows),
        }
    )
    result["net_correction"] = result["rescues"] - result["destructions"]
    payloads = [row.get("communication_payload") or {} for row in rows]
    result.update(
        {
            "total_generations": len(rows),
            "mean_generated_tokens": safe_rate(
                sum(int(row["generation_length"]) for row in rows), len(rows)
            ),
            "mean_input_tokens": safe_rate(
                sum(int(row["prompt_tokens"]) for row in rows), len(rows)
            ),
            "mean_latency_seconds": safe_rate(
                sum(float(row["generation_seconds"]) for row in rows), len(rows)
            ),
            "total_generation_seconds": sum(
                float(row["generation_seconds"]) for row in rows
            ),
            "mean_payload_bytes": safe_rate(
                sum(float(payload.get("payload_bytes", 0)) for payload in payloads),
                len(payloads),
            ),
            "mean_payload_tokens": safe_rate(
                sum(float(payload["tokens"]) for payload in payloads if "tokens" in payload),
                sum("tokens" in payload for payload in payloads),
            ),
            "mean_payload_states": safe_rate(
                sum(float(payload["states"]) for payload in payloads if "states" in payload),
                sum("states" in payload for payload in payloads),
            ),
        }
    )
    return result


def evidence_payload_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    evidence_rows = [row for row in rows if row.get("condition") == "true_evidence"]
    if not evidence_rows:
        return None
    payloads = [row.get("communication_payload") or {} for row in evidence_rows]
    original_tokens = sum(int(payload.get("original_tokens", 0)) for payload in payloads)
    evidence_tokens = sum(int(payload.get("tokens", 0)) for payload in payloads)
    return {
        "directional_messages": len(evidence_rows),
        "explicit_answer_cue_present": sum(
            bool(payload.get("explicit_answer_cue_present_after_filter"))
            for payload in payloads
        ),
        "sender_answer_label_present": sum(
            bool(payload.get("sender_answer_label_present_after_filter"))
            for payload in payloads
        ),
        "sender_answer_text_present": sum(
            bool(payload.get("sender_answer_text_present_after_filter"))
            for payload in payloads
        ),
        "empty_messages": sum(
            int(payload.get("characters", 0)) == 0 for payload in payloads
        ),
        "source_binding_mismatches": sum(
            int(row.get("message_source_item_id", -1)) != int(row["item_id"])
            or str(row.get("message_source_agent_id"))
            != str(row.get("sender_agent_id"))
            for row in evidence_rows
        ),
        "removed_span_count": sum(
            int(payload.get("removed_span_count", 0)) for payload in payloads
        ),
        "original_tokens": original_tokens,
        "evidence_tokens": evidence_tokens,
        "token_retention_rate": safe_rate(evidence_tokens, original_tokens),
    }


def item_fraction_arrays(
    rows: Sequence[Mapping[str, Any]], item_ids: Sequence[int], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    position = {item_id: index for index, item_id in enumerate(item_ids)}
    numerator = np.zeros(len(item_ids), dtype=np.int64)
    denominator = np.zeros(len(item_ids), dtype=np.int64)
    for row in rows:
        if _eligible(row, metric):
            index = position[int(row["item_id"])]
            denominator[index] += 1
            numerator[index] += int(_success(row, metric))
    return numerator, denominator


def bootstrap_fraction(
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


def condition_bootstrap(
    rows: Sequence[Mapping[str, Any]], item_ids: Sequence[int], samples: np.ndarray
) -> dict[str, np.ndarray]:
    rates: dict[str, np.ndarray] = {}
    for metric in ("accuracy", "cr", "pr", "sr", "scr", "sra", "fcs", "fws"):
        rates[metric] = bootstrap_fraction(
            *item_fraction_arrays(rows, item_ids, metric), samples
        )
    rates["si"] = (rates["cr"] + rates["pr"]) / 2
    rates["follow_selectivity"] = rates["fcs"] - rates["fws"]
    return rates


def paired_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    left_map = {(int(row["item_id"]), str(row["direction"])): row for row in left}
    right_map = {(int(row["item_id"]), str(row["direction"])): row for row in right}
    if set(left_map) != set(right_map):
        raise RuntimeError("Paired conditions do not contain identical directional keys")
    return [(left_map[key], right_map[key]) for key in sorted(left_map)]


def influence_estimate(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    item_ids: Sequence[int],
    samples: np.ndarray,
    category: str,
) -> tuple[int, int, float | None, np.ndarray]:
    position = {item_id: index for index, item_id in enumerate(item_ids)}
    numerator = np.zeros(len(item_ids), dtype=np.int64)
    denominator = np.zeros(len(item_ids), dtype=np.int64)
    for true_row, control_row in pairs:
        if category != "all" and true_row["pair_classification"] != category:
            continue
        index = position[int(true_row["item_id"])]
        denominator[index] += 1
        numerator[index] += int(
            true_row["receiver_post_answer"] != control_row["receiver_post_answer"]
        )
    num, den = int(numerator.sum()), int(denominator.sum())
    values = bootstrap_fraction(numerator, denominator, samples)
    return num, den, safe_rate(num, den), values


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_derived_cases(
    rows: Sequence[Mapping[str, Any]], replication_id: str
) -> list[dict[str, Any]]:
    cases = []
    for row in rows:
        post = row["receiver_post_answer"]
        sender = row["sender_pre_answer"]
        receiver = row["receiver_pre_answer"]
        cases.append(
            {
                "benchmark": "medqa300",
                "replication_id": replication_id,
                "item_id": int(row["item_id"]),
                "direction": row["direction"],
                "condition": row["condition"],
                "gold": row["gold"],
                "sender_correct": bool(row["sender_correct"]),
                "receiver_pre_correct": bool(row["receiver_pre_correct"]),
                "receiver_post_correct": bool(row["receiver_post_correct"]),
                "sender_answer": sender,
                "receiver_pre_answer": receiver,
                "receiver_post_answer": post,
                "pre_category": row["pair_classification"],
                "answer_changed": bool(row["answer_changed"]),
                "followed_sender": bool(row["followed_sender"]),
                "changed_to_third_answer": bool(
                    post is not None and post != sender and post != receiver
                ),
                "rescued": bool(
                    not row["receiver_pre_correct"] and row["receiver_post_correct"]
                ),
                "destroyed": bool(
                    row["receiver_pre_correct"] and not row["receiver_post_correct"]
                ),
                "invalid_parse": post is None,
                "revision_seed": int(row["revision_seed"]),
                "prompt_sha256": row["prompt_sha256"],
                "message_source_item_id": row.get("message_source_item_id"),
                "message_source_agent_id": row.get("message_source_agent_id"),
                "generation_seconds": float(row["generation_seconds"]),
                "generation_length": int(row["generation_length"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "payload_bytes": int(
                    (row.get("communication_payload") or {}).get("payload_bytes", 0)
                ),
            }
        )
    return cases


def build_item_cases(
    cases: Sequence[Mapping[str, Any]], replication_id: str
) -> list[dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for case in cases:
        item_id = int(case["item_id"])
        target = result.setdefault(
            item_id,
            {
                "benchmark": "medqa300",
                "replication_id": replication_id,
                "item_id": item_id,
                "gold": case["gold"],
            },
        )
        stem = f"{case['direction']}__{case['condition']}"
        for field in (
            "sender_answer",
            "receiver_pre_answer",
            "receiver_post_answer",
            "sender_correct",
            "receiver_pre_correct",
            "receiver_post_correct",
            "pre_category",
            "answer_changed",
            "followed_sender",
            "changed_to_third_answer",
        ):
            target[f"{stem}__{field}"] = case[field]
    return [result[key] for key in sorted(result)]


def regression_check(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks = []
    for condition, expected in EXPECTED_REGRESSION.items():
        for metric, value in expected.items():
            actual = metrics[condition][metric]
            passed = actual is not None and abs(actual - value) < 1e-12
            checks.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "expected": value,
                    "actual": actual,
                    "passed": passed,
                }
            )
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"Frozen seed_pair_00 regression failed: {checks}")
    return {"passed": True, "checks": checks}


def command_output(command: Sequence[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    source_root: Path,
    config: Mapping[str, Any],
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/.hf_home"))
    model_ref = hf_home / "hub/models--Qwen--Qwen3-4B/refs/main"
    gpu_line = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        repo_root,
    )
    replication_id = config.get("replication_id") or "seed_pair_00"
    return {
        "run_id": f"medqa300_{replication_id}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source_root.resolve()),
        "source_config_fingerprint": config["fingerprint"],
        "git_commit_hash": command_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_dirty_status": command_output(
            ["git", "status", "--porcelain=v1"], repo_root
        ),
        "benchmark_name": "medqa300",
        "dataset_source": "repository-bundled data/medqa.json",
        "dataset_revision": config["dataset_sha256"],
        "dataset_split": "fixed 300-item diagnostic subset",
        "dataset_item_count": len(config["selected_item_ids"]),
        "model_name": config["model"],
        "model_revision": model_ref.read_text(encoding="utf-8").strip()
        if model_ref.is_file()
        else None,
        "tokenizer_revision": model_ref.read_text(encoding="utf-8").strip()
        if model_ref.is_file()
        else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu_model_and_driver": gpu_line.splitlines() if gpu_line else None,
        "cuda_version": torch.version.cuda,
        "global_seed": config["global_seed"],
        "replication_id": replication_id,
        "generation_config": config["generation"],
        "communication_conditions": config["completed_revision_conditions"],
        "prompt_template_versions": {
            "revision": config["revision_prompt_version"],
            "statebridge_injection": config["statebridge"]["receiver_injection_position"],
        },
        "answer_parser_version": "icr.protocol.parse_medqa_answer@ICR-MEDQA300-V2",
        "analysis": {
            "version": "communication_first_step1_v2",
            "bootstrap_cluster": "item_id (both directions retained)",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_iterations": bootstrap_iterations,
            "new_model_inference": False,
        },
    }


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def fmt_ci(value: Sequence[float] | None) -> str:
    return "—" if value is None else f"[{100*value[0]:.2f}%, {100*value[1]:.2f}%]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    cli = parser.parse_args()

    source_root = cli.source_root.resolve()
    seed_root = cli.seed_root.resolve()
    output = seed_root / "analysis_v2"
    config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    replication_id = config.get("replication_id") or "seed_pair_00"
    rows = [
        json.loads(line)
        for line in (source_root / "revisions/merged.jsonl")
        .read_text(encoding="utf-8")
        .split("\n")
        if line
    ]
    conditions = tuple(config["completed_revision_conditions"])
    keys = {(row["item_id"], row["direction"], row["condition"]) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError(f"Duplicate revision keys: rows={len(rows)} unique={len(keys)}")
    # Every condition must cover exactly the same directional pairs, whether or
    # not a both-correct subsample removed some of them. Without a subsample the
    # run must still be the full item x direction grid.
    per_condition = {
        condition: {(i, d) for i, d, c in keys if c == condition}
        for condition in conditions
    }
    reference = per_condition[conditions[0]]
    mismatched = [c for c, pairs in per_condition.items() if pairs != reference]
    if mismatched:
        raise RuntimeError(f"Conditions cover different directional pairs: {mismatched}")
    sampling = config.get("both_correct_sampling") or {}
    keep_one_in = int(sampling.get("keep_one_in", 1))
    skip_all_correct = bool(sampling.get("skip_all_correct_items", False))
    from . import DIRECTIONS

    full_grid = len(config["selected_item_ids"]) * len(DIRECTIONS)
    if keep_one_in <= 1 and not skip_all_correct and len(reference) != full_grid:
        raise RuntimeError(
            f"Incomplete run: {len(reference)} directional pairs, "
            f"expected {full_grid}"
        )

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    if set(by_condition) != set(conditions):
        raise RuntimeError("Source conditions do not match the frozen config")

    metrics = {
        condition: condition_metrics(condition_rows)
        for condition, condition_rows in sorted(by_condition.items())
    }
    # EXPECTED_REGRESSION holds the frozen numbers of exactly one run: the V2
    # MedQA300 seed_pair_00 replication. Later protocols reuse the replication
    # id, so the guard must also match the protocol, dataset, and item count.
    is_frozen_v2_baseline = (
        replication_id == "seed_pair_00"
        and config.get("revision_prompt_version")
        == "icr_v2_statebridge_front_injection"
        and config.get("dataset", "medqa") == "medqa"
        and len(config["selected_item_ids"]) == 300
    )
    regression = (
        regression_check(metrics)
        if is_frozen_v2_baseline
        else {
            "passed": True,
            "not_applicable": True,
            "reason": (
                "The frozen numeric regression baseline is the V2 MedQA300 "
                "seed_pair_00 replication only."
            ),
        }
    )
    item_ids = sorted(int(item_id) for item_id in config["selected_item_ids"])
    rng = np.random.default_rng(cli.bootstrap_seed)
    samples = rng.integers(0, len(item_ids), (cli.num_bootstrap, len(item_ids)))
    boot = {
        condition: condition_bootstrap(condition_rows, item_ids, samples)
        for condition, condition_rows in by_condition.items()
    }

    bootstrap_metrics = {
        "metadata": {
            "seed": cli.bootstrap_seed,
            "iterations": cli.num_bootstrap,
            "cluster": "item_id",
            "directions_per_cluster": 2,
        },
        "conditions": {
            condition: {
                metric: {
                    "point_estimate": metrics[condition][metric],
                    "ci95": ci95(boot[condition][metric]),
                }
                for metric in PRIMARY_METRICS
            }
            for condition in sorted(metrics)
        },
    }

    pairwise = {}
    for metric in PAIRWISE_METRICS:
        values = boot["true_text"][metric] - boot["true_statebridge"][metric]
        pairwise[metric] = {
            "point_estimate": safe_delta(
                metrics["true_text"][metric],
                metrics["true_statebridge"][metric],
            ),
            "ci95": ci95(values),
        }

    evidence_pairwise = {}
    if "true_evidence" in boot:
        for reference in (
            "none",
            "true_text",
            "true_statebridge",
            "true_latentmas",
        ):
            if reference not in boot:
                continue
            paired = paired_rows(
                by_condition["true_evidence"], by_condition[reference]
            )
            gained = sum(
                bool(evidence_row["receiver_post_correct"])
                and not bool(reference_row["receiver_post_correct"])
                for evidence_row, reference_row in paired
            )
            lost = sum(
                not bool(evidence_row["receiver_post_correct"])
                and bool(reference_row["receiver_post_correct"])
                for evidence_row, reference_row in paired
            )
            evidence_pairwise[reference] = {
                "gained": gained,
                "lost": lost,
                "ties": len(paired) - gained - lost,
                "metrics": {
                    metric: {
                        "point_estimate": safe_delta(
                            metrics["true_evidence"][metric],
                            metrics[reference][metric],
                        ),
                        "ci95": ci95(
                            boot["true_evidence"][metric]
                            - boot[reference][metric]
                        ),
                    }
                    for metric in PAIRWISE_METRICS
                },
            }

    # Baseline-only runs carry no self/other controls, so every comparison is
    # emitted if and only if both of its conditions were actually run.
    available = set(boot)
    utility_defs = {
        name: pair
        for name, pair in {
            "text_ce": ("true_text", "none"),
            "text_esv": ("true_text", "other_text"),
            "text_oav": ("true_text", "self_text"),
            "statebridge_ce": ("true_statebridge", "none"),
            "statebridge_esv": ("true_statebridge", "other_statebridge"),
            "statebridge_oav": ("true_statebridge", "self_statebridge"),
            "latentmas_ce": ("true_latentmas", "none"),
            "latentmas_esv": ("true_latentmas", "other_latentmas"),
            "latentmas_oav": ("true_latentmas", "self_latentmas"),
            "evidence_ce": ("true_evidence", "none"),
            "evidence_esv": ("true_evidence", "other_evidence"),
            "evidence_oav": ("true_evidence", "self_evidence"),
        }.items()
        if pair[0] in available and pair[1] in available
    }
    utility_rows = []
    utility_bootstrap = {}
    for name, (true_condition, control_condition) in utility_defs.items():
        utility_bootstrap[name] = {}
        for metric in ("accuracy", "cr", "pr", "sr", "scr", "si", "sra"):
            values = boot[true_condition][metric] - boot[control_condition][metric]
            estimate = safe_delta(
                metrics[true_condition][metric], metrics[control_condition][metric]
            )
            interval = ci95(values)
            utility_rows.append(
                {
                    "comparison": name,
                    "true_condition": true_condition,
                    "control_condition": control_condition,
                    "metric": metric,
                    "delta": estimate,
                    "ci95_low": interval[0] if interval else None,
                    "ci95_high": interval[1] if interval else None,
                }
            )
            utility_bootstrap[name][metric] = {
                "point_estimate": estimate,
                "ci95": interval,
            }

    influence_rows = []
    influence_bootstrap: dict[str, Any] = {}
    influence_defs = {
        "text": {
            "true_vs_none": ("true_text", "none"),
            "true_vs_self": ("true_text", "self_text"),
            "true_vs_other": ("true_text", "other_text"),
        },
        "statebridge": {
            "true_vs_none": ("true_statebridge", "none"),
            "true_vs_self": ("true_statebridge", "self_statebridge"),
            "true_vs_other": ("true_statebridge", "other_statebridge"),
        },
        "latentmas": {
            "true_vs_none": ("true_latentmas", "none"),
            "true_vs_self": ("true_latentmas", "self_latentmas"),
            "true_vs_other": ("true_latentmas", "other_latentmas"),
        },
        "evidence": {
            "true_vs_none": ("true_evidence", "none"),
            "true_vs_self": ("true_evidence", "self_evidence"),
            "true_vs_other": ("true_evidence", "other_evidence"),
        },
    }
    influence_defs = {
        modality: {
            name: pair
            for name, pair in comparisons.items()
            if pair[0] in available and pair[1] in available
        }
        for modality, comparisons in influence_defs.items()
    }
    influence_defs = {
        modality: comparisons
        for modality, comparisons in influence_defs.items()
        if comparisons
    }
    for modality, comparisons in influence_defs.items():
        influence_bootstrap[modality] = {}
        for name, (true_condition, control_condition) in comparisons.items():
            influence_bootstrap[modality][name] = {}
            pairs = paired_rows(by_condition[true_condition], by_condition[control_condition])
            for category in ("all", *CATEGORIES):
                numerator, denominator, rate, values = influence_estimate(
                    pairs, item_ids, samples, category
                )
                interval = ci95(values)
                record = {
                    "modality": modality,
                    "comparison": name,
                    "true_condition": true_condition,
                    "control_condition": control_condition,
                    "category": category,
                    "numerator": numerator,
                    "denominator": denominator,
                    "influence_rate": rate,
                    "ci95_low": interval[0] if interval else None,
                    "ci95_high": interval[1] if interval else None,
                }
                influence_rows.append(record)
                influence_bootstrap[modality][name][category] = {
                    "point_estimate": rate,
                    "numerator": numerator,
                    "denominator": denominator,
                    "ci95": interval,
                }

    transition_rows = []
    for condition in (
        "true_text",
        "true_evidence",
        "true_statebridge",
        "true_latentmas",
    ):
        if condition not in by_condition:
            continue
        condition_rows = by_condition[condition]
        transitions = {
            "wrong_to_wrong": lambda row: not row["receiver_pre_correct"]
            and not row["receiver_post_correct"],
            "wrong_to_correct": lambda row: not row["receiver_pre_correct"]
            and row["receiver_post_correct"],
            "correct_to_correct": lambda row: row["receiver_pre_correct"]
            and row["receiver_post_correct"],
            "correct_to_wrong": lambda row: row["receiver_pre_correct"]
            and not row["receiver_post_correct"],
        }
        for transition, predicate in transitions.items():
            count = sum(bool(predicate(row)) for row in condition_rows)
            transition_rows.append(
                {
                    "condition": condition,
                    "matrix": "correctness",
                    "transition": transition,
                    "count": count,
                    "denominator": len(condition_rows),
                    "rate": count / len(condition_rows),
                }
            )
        disagreements = [
            row
            for row in condition_rows
            if row["sender_pre_answer"] is not None
            and row["receiver_pre_answer"] is not None
            and row["sender_pre_answer"] != row["receiver_pre_answer"]
        ]
        answer_transitions: dict[str, Callable[[Mapping[str, Any]], bool]] = {
            "keeps_receiver_answer": lambda row: row["receiver_post_answer"]
            == row["receiver_pre_answer"],
            "adopts_sender_answer": lambda row: row["receiver_post_answer"]
            == row["sender_pre_answer"],
            "changes_to_third_answer": lambda row: row["receiver_post_answer"] is not None
            and row["receiver_post_answer"] != row["sender_pre_answer"]
            and row["receiver_post_answer"] != row["receiver_pre_answer"],
            "invalid_post_answer": lambda row: row["receiver_post_answer"] is None,
        }
        for transition, predicate in answer_transitions.items():
            count = sum(bool(predicate(row)) for row in disagreements)
            transition_rows.append(
                {
                    "condition": condition,
                    "matrix": "sender_receiver_answer_disagreement",
                    "transition": transition,
                    "count": count,
                    "denominator": len(disagreements),
                    "rate": safe_rate(count, len(disagreements)),
                }
            )

    conditional_rows = []
    for condition, condition_rows in sorted(by_condition.items()):
        for category in CATEGORIES:
            subset = [row for row in condition_rows if row["pair_classification"] == category]
            conditional_rows.append(
                {
                    "condition": condition,
                    "category": category,
                    "directional_examples": len(subset),
                    "post_accuracy": safe_rate(
                        sum(bool(row["receiver_post_correct"]) for row in subset),
                        len(subset),
                    ),
                    "answer_change_rate": safe_rate(
                        sum(bool(row["answer_changed"]) for row in subset), len(subset)
                    ),
                    "follow_sender_rate": safe_rate(
                        sum(bool(row["followed_sender"]) for row in subset), len(subset)
                    ),
                }
            )

    metric_rows = []
    for condition in sorted(metrics):
        row: dict[str, Any] = {"condition": condition, **metrics[condition]}
        for metric in PRIMARY_METRICS:
            interval = bootstrap_metrics["conditions"][condition][metric]["ci95"]
            row[f"{metric}_ci95_low"] = interval[0] if interval else None
            row[f"{metric}_ci95_high"] = interval[1] if interval else None
        metric_rows.append(row)

    sender_following_rows = []
    for condition in sorted(metrics):
        sender_following_rows.append(
            {
                "condition": condition,
                "fcs": metrics[condition]["fcs"],
                "fcs_numerator": metrics[condition]["fcs_numerator"],
                "fcs_denominator": metrics[condition]["fcs_denominator"],
                "fcs_ci95": bootstrap_metrics["conditions"][condition]["fcs"]["ci95"],
                "fws": metrics[condition]["fws"],
                "fws_numerator": metrics[condition]["fws_numerator"],
                "fws_denominator": metrics[condition]["fws_denominator"],
                "fws_ci95": bootstrap_metrics["conditions"][condition]["fws"]["ci95"],
                "follow_selectivity": metrics[condition]["follow_selectivity"],
                "follow_selectivity_ci95": bootstrap_metrics["conditions"][condition][
                    "follow_selectivity"
                ]["ci95"],
            }
        )

    cases = build_derived_cases(rows, replication_id)
    item_cases = build_item_cases(cases, replication_id)
    influence_utility_rows = []
    utility_lookup = {
        (row["comparison"], row["metric"]): row for row in utility_rows
    }
    for row in influence_rows:
        if row["category"] != "all":
            continue
        utility_name = f"{row['modality']}_{'ce' if row['comparison'] == 'true_vs_none' else 'oav' if row['comparison'] == 'true_vs_self' else 'esv'}"
        utility = utility_lookup[(utility_name, "accuracy")]
        influence_utility_rows.append(
            {
                **row,
                "accuracy_utility": utility["delta"],
                "utility_ci95_low": utility["ci95_low"],
                "utility_ci95_high": utility["ci95_high"],
            }
        )

    summary = {
        "status": "complete",
        "step": 1,
        "source_artifact": str(source_root),
        "new_model_inference": False,
        "directional_records": len(rows),
        "item_count": len(item_ids),
        "conditions": metrics,
        "regression_baseline": regression,
        "text_minus_statebridge": pairwise,
        "evidence_pairwise": evidence_pairwise,
        "evidence_payload_audit": evidence_payload_audit(rows),
        "causal_utility": utility_bootstrap,
        "causal_influence": influence_bootstrap,
        "bootstrap": bootstrap_metrics["metadata"],
    }

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        seed_root / "manifest.json",
        build_manifest(
            source_root,
            config,
            bootstrap_seed=cli.bootstrap_seed,
            bootstrap_iterations=cli.num_bootstrap,
        ),
    )
    atomic_write_json(
        seed_root / "source_artifact.json",
        {
            "replication_id": replication_id,
            "source_artifact": str(source_root),
            "source_config_fingerprint": config["fingerprint"],
            "preserved_unchanged": True,
        },
    )
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "bootstrap_metrics.json", bootstrap_metrics)
    atomic_write_json(
        output / "bootstrap_pairwise.json",
        {
            "metadata": bootstrap_metrics["metadata"],
            "text_minus_statebridge": pairwise,
            "evidence_pairwise": evidence_pairwise,
            "causal_utility": utility_bootstrap,
            "causal_influence": influence_bootstrap,
        },
    )
    atomic_write_csv(output / "metrics_full.csv", metric_rows)
    atomic_write_csv(output / "conditional_metrics.csv", conditional_rows)
    atomic_write_csv(output / "sender_following.csv", sender_following_rows)
    atomic_write_csv(output / "transition_matrix.csv", transition_rows)
    atomic_write_csv(output / "causal_influence.csv", influence_rows)
    atomic_write_csv(output / "causal_utility.csv", utility_rows)
    atomic_write_csv(
        output / "correction_preservation.csv",
        [
            {
                "condition": condition,
                "correction_rate": value["cr"],
                "preservation_rate": value["pr"],
                "selectivity_index": value["si"],
            }
            for condition, value in sorted(metrics.items())
        ],
    )
    atomic_write_csv(output / "influence_utility.csv", influence_utility_rows)
    atomic_write_parquet(output / "per_direction_cases.parquet", cases)
    atomic_write_parquet(output / "per_item_cases.parquet", item_cases)

    focus = ("accuracy", "cr", "pr", "si", "sra", "fcs", "fws", "follow_selectivity")
    lines = [
        "# ICR Communication-First Step 1",
        "",
        f"`{replication_id}` was analyzed without additional model inference. Confidence intervals use {cli.num_bootstrap:,} item-cluster bootstrap resamples; both directions are retained whenever an item is sampled.",
        "",
        "| Channel | Acc | CR | PR | SI | SRA | FCS | FWS | FollowSelectivity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_conditions = (
        ("No message", "none"),
        ("Full Text", "true_text"),
        ("Evidence", "true_evidence"),
        ("StateBridge", "true_statebridge"),
        ("LatentMAS", "true_latentmas"),
    )
    for label, condition in display_conditions:
        if condition not in metrics:
            continue
        value = metrics[condition]
        lines.append(
            f"| {label} | {fmt(value['accuracy'])} | {fmt(value['cr'])} | {fmt(value['pr'])} | "
            f"{fmt(value['si'])} | {fmt(value['sra'])} | {fmt(value['fcs'])} | "
            f"{fmt(value['fws'])} | {fmt(value['follow_selectivity'])} |"
        )
    if evidence_pairwise:
        lines.extend(
            [
                "",
                "## Evidence channel paired comparisons",
                "",
                "| Reference | Accuracy difference | 95% item-cluster bootstrap CI | Gained / Lost |",
                "|---|---:|---:|---:|",
            ]
        )
        labels = {
            "none": "No message",
            "true_text": "Full Text",
            "true_statebridge": "StateBridge",
            "true_latentmas": "LatentMAS",
        }
        for reference, comparison in evidence_pairwise.items():
            accuracy = comparison["metrics"]["accuracy"]
            lines.append(
                f"| {labels[reference]} | {fmt(accuracy['point_estimate'])} | "
                f"{fmt_ci(accuracy['ci95'])} | "
                f"{comparison['gained']} / {comparison['lost']} |"
            )
        audit = evidence_payload_audit(rows)
        assert audit is not None
        lines.extend(
            [
                "",
                "## Evidence payload audit",
                "",
                f"- Directional messages: {audit['directional_messages']}",
                f"- Explicit answer cues retained: {audit['explicit_answer_cue_present']}",
                f"- Sender answer labels retained: {audit['sender_answer_label_present']}",
                f"- Sender answer texts retained: {audit['sender_answer_text_present']}",
                f"- Empty messages: {audit['empty_messages']}",
                f"- Source-binding mismatches: {audit['source_binding_mismatches']}",
                f"- Token retention: {fmt(audit['token_retention_rate'])}",
                "",
                "Answer-label/text presence is diagnostic leakage, not proof that the "
                "payload is verified or answer-blind evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Text − StateBridge",
            "",
            "| Metric | Difference | 95% item-cluster bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for metric in focus:
        value = pairwise[metric]
        lines.append(
            f"| {metric} | {fmt(value['point_estimate'])} | {fmt_ci(value['ci95'])} |"
        )
    sb_influence = influence_bootstrap["statebridge"]["true_vs_none"]["all"]
    sb_utility = utility_bootstrap["statebridge_ce"]["accuracy"]
    lines.extend(
        [
            "",
            "## Evidence-bounded answers",
            "",
            f"- **High sender influence?** Yes in this artifact: true StateBridge differs from none on {sb_influence['numerator']}/{sb_influence['denominator']} answers ({fmt(sb_influence['point_estimate'])}, 95% CI {fmt_ci(sb_influence['ci95'])}).",
            f"- **Positive communication utility?** No on overall accuracy: StateBridge CE is {fmt(sb_utility['point_estimate'])}, 95% CI {fmt_ci(sb_utility['ci95'])}.",
            f"- **Sender-belief overwrite behavior?** The observed pattern is consistent with aggressive sender following: FCS={fmt(metrics['true_statebridge']['fcs'])}, FWS={fmt(metrics['true_statebridge']['fws'])}, FollowSelectivity={fmt(metrics['true_statebridge']['follow_selectivity'])}. This is a behavioral description of this run, not yet a cross-seed causal generalization.",
            "",
            (
                "The correction subset contains "
                f"{metrics['true_text']['cr_denominator']} directional cases and the "
                "destruction subset contains "
                f"{metrics['true_text']['pr_denominator']}, so cross-seed replication "
                "remains necessary."
            ),
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
