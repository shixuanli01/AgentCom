#!/usr/bin/env python3
"""Validate and freeze one completed cross-benchmark result artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = ("icr", "egr")
CODE_FILES = (
    "methods/state_bridge.py",
    "prompts.py",
    "utils.py",
    "scripts/run_cross_benchmark_first_wave.sh",
    "scripts/freeze_cross_benchmark_result.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"Required result file is missing: {path}")
    return path


def git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def code_snapshot() -> dict[str, Any]:
    paths: list[Path] = []
    for root_name in CODE_ROOTS:
        paths.extend(
            path
            for path in (REPO_ROOT / root_name).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    paths.extend(REPO_ROOT / value for value in CODE_FILES)
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in unique
    }
    return {
        "files": hashes,
        "aggregate_sha256": sha256_json(hashes),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(
            git_value("status", "--short").encode("utf-8")
        ).hexdigest(),
    }


def validate_egr(root: Path, expected_items: int, task: str) -> list[Path]:
    files: list[Path] = []
    expected = (("egr_contrast", "egr_contrast"),)
    if task not in {"mbppplus", "humanevalplus"}:
        expected += (("egr_permutation", "egr_permutation_gate"),)
    for directory, condition in expected:
        egr_root = root / directory
        status_path = require_file(egr_root / "run_status.json")
        status = read_json(status_path)
        if status.get("status") != "complete":
            raise RuntimeError(f"{directory} is not complete")
        if int(status.get("items", -1)) != expected_items:
            raise RuntimeError(f"{directory} item count is incomplete")
        revision_path = require_file(egr_root / "revisions" / f"{condition}.jsonl")
        if count_jsonl(revision_path) != expected_items * 2:
            raise RuntimeError(f"{condition} directional rows are incomplete")
        files.extend((status_path, revision_path, require_file(egr_root / "config.json")))
    return files


def relative_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(set(paths))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()

    frozen_path = root / "FROZEN_RESULT.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        print(f"already frozen: {frozen['freeze_fingerprint']}")
        return

    config_path = require_file(root / "config.json")
    config = read_json(config_path)
    selected_ids = [int(value) for value in config["selected_item_ids"]]
    item_count = len(selected_ids)
    conditions = list(config.get("completed_revision_conditions") or [])
    if not conditions:
        raise RuntimeError("Baseline revision conditions are not marked complete")

    prebelief_path = require_file(root / "prebeliefs" / "merged.jsonl")
    revision_path = require_file(root / "revisions" / "merged.jsonl")
    report_path = require_file(root / "cross_benchmark_analysis" / "report.json")
    report_markdown = require_file(root / "cross_benchmark_analysis" / "report.md")
    if count_jsonl(prebelief_path) != item_count * 2:
        raise RuntimeError("Prebelief row count is incomplete")
    expected_revisions = item_count * 2 * len(conditions)
    if count_jsonl(revision_path) != expected_revisions:
        raise RuntimeError("Baseline revision row count is incomplete")

    report = read_json(report_path)
    if report.get("selected_item_ids") != selected_ids:
        raise RuntimeError("Report selected IDs differ from config")
    missing_conditions = set(conditions) - set(report.get("conditions", {}))
    if missing_conditions:
        raise RuntimeError(f"Report is missing conditions: {sorted(missing_conditions)}")

    result_files = [
        config_path,
        prebelief_path,
        revision_path,
        report_path,
        report_markdown,
        *validate_egr(root, item_count, str(config["dataset"])),
    ]
    provenance = root / "subset_provenance.json"
    if provenance.is_file():
        result_files.append(provenance)
    hashes = relative_hashes(root, result_files)
    stable = {
        "schema": "agentcom_cross_benchmark_frozen_result_v1",
        "label": args.label,
        "artifact_root": str(root),
        "task": config["dataset"],
        "benchmark": config.get("benchmark"),
        "replication_id": config.get("replication_id", "seed_pair_00"),
        "item_count": item_count,
        "selected_item_ids_sha256": sha256_json(selected_ids),
        "selection": config.get("selection"),
        "dataset_sha256": config.get("dataset_sha256"),
        "config_fingerprint": config["fingerprint"],
        "baseline_conditions": conditions,
        "expected_baseline_revision_rows": expected_revisions,
        "result_file_sha256": hashes,
        "result_files_aggregate_sha256": sha256_json(hashes),
        "code_snapshot": code_snapshot(),
        "condition_exact_counts": {
            condition: {
                "post_correct": values["post_correct"],
                "directional_examples": values["directional_examples"],
                "rescues": values["rescues"],
                "destructions": values["destructions"],
            }
            for condition, values in report["conditions"].items()
        },
    }
    frozen = {
        **stable,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "freeze_fingerprint": sha256_json(stable),
    }
    temporary = frozen_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(frozen_path)
    print(f"frozen {args.label}: {frozen['freeze_fingerprint']}")


if __name__ == "__main__":
    main()
