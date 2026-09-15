"""Deterministically merge resumable ICR rank artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .protocol import atomic_write_jsonl, sha256_json


def _deduplicate(
    paths: Iterable[Path], key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    hashes: dict[tuple[Any, ...], str] = {}
    for path in sorted(paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") != "complete":
            continue
        key = tuple(row[field] for field in key_fields)
        digest = sha256_json(row)
        if key in rows and hashes[key] != digest:
            raise RuntimeError(f"Conflicting completed records for {key}")
        rows[key] = row
        hashes[key] = digest
    return [rows[key] for key in sorted(rows)]


def merge_prebeliefs(root: Path, *, require_complete: bool) -> list[dict[str, Any]]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    rows = _deduplicate(
        root.glob("prebeliefs/rank*/records/item_*.json"), ("item_id", "agent_id")
    )
    if require_complete:
        expected = {
            (item_id, agent_id)
            for item_id in config["selected_item_ids"]
            for agent_id in ("A", "B")
        }
        actual = {(row["item_id"], row["agent_id"]) for row in rows}
        if actual != expected:
            raise RuntimeError(
                f"Incomplete prebelief cache: have {len(actual)}, expected {len(expected)}"
            )
    atomic_write_jsonl(root / "prebeliefs" / "merged.jsonl", rows)
    return rows


def merge_revisions(root: Path, *, require_complete: bool) -> list[dict[str, Any]]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    rows = _deduplicate(
        root.glob("revisions/rank*/records/item_*.json"),
        ("item_id", "direction", "condition"),
    )
    if require_complete:
        conditions = config.get("revision_conditions_requested")
        if not conditions:
            raise RuntimeError("config.json does not declare requested revision conditions")
        expected = {
            (item_id, direction, condition)
            for item_id in config["selected_item_ids"]
            for direction in ("A_to_B", "B_to_A")
            for condition in conditions
        }
        actual = {
            (row["item_id"], row["direction"], row["condition"])
            for row in rows
            if row["condition"] in conditions
        }
        if actual != expected:
            raise RuntimeError(
                f"Incomplete revisions: have {len(actual)}, expected {len(expected)}"
            )
        config["completed_revision_conditions"] = list(conditions)
        from .protocol import atomic_write_json

        atomic_write_json(root / "config.json", config)
    atomic_write_jsonl(root / "revisions" / "merged.jsonl", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prebeliefs", "revisions"), required=True)
    parser.add_argument("--require-complete", action="store_true")
    cli = parser.parse_args()
    if cli.phase == "prebeliefs":
        rows = merge_prebeliefs(cli.artifact_root, require_complete=cli.require_complete)
    else:
        rows = merge_revisions(cli.artifact_root, require_complete=cli.require_complete)
    print(f"merged {len(rows)} {cli.phase} records")


if __name__ == "__main__":
    main()
