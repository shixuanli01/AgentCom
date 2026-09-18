"""Materialize a provenance-tracked ICR subset from a complete prebelief cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .benchmarks import select_item_ids
from .merge import merge_prebeliefs
from .protocol import atomic_write_json, sha256_json


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_records(root: Path) -> dict[tuple[int, str], tuple[Path, dict[str, Any]]]:
    records = {}
    for path in root.glob("prebeliefs/rank*/records/item_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        records[(int(row["item_id"]), str(row["agent_id"]))] = (path, row)
    return records


def selected_ids_from_cli(cli: argparse.Namespace, source_rows: list[dict[str, Any]]) -> list[int]:
    available = sorted({int(row["item_id"]) for row in source_rows})
    if cli.mode == "random":
        positions = select_item_ids(
            len(available), sample_size=cli.size, selection_seed=cli.selection_seed
        )
        return [available[position] for position in positions]
    by_item: dict[int, list[dict[str, Any]]] = {}
    for row in source_rows:
        by_item.setdefault(int(row["item_id"]), []).append(row)
    return sorted(
        item_id
        for item_id, rows in by_item.items()
        if len({row.get("parsed_answer") for row in rows}) > 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("random", "disagreement"), required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--selection-seed", type=int, default=42)
    cli = parser.parse_args()

    source_config = json.loads(
        (cli.source_root / "config.json").read_text(encoding="utf-8")
    )
    source_rows = merge_prebeliefs(cli.source_root, require_complete=True)
    selected_ids = selected_ids_from_cli(cli, source_rows)
    if cli.mode == "random" and len(selected_ids) != cli.size:
        raise RuntimeError("Random subset does not have the requested size")
    if not selected_ids:
        raise RuntimeError("Subset selection produced no items")

    stable = {
        key: value
        for key, value in source_config.items()
        if key
        not in {
            "fingerprint",
            "revision_conditions_requested",
            "completed_revision_conditions",
        }
    }
    stable["selected_item_ids"] = selected_ids
    stable["selection"] = {
        "mode": cli.mode,
        "size": len(selected_ids),
        "selection_seed": cli.selection_seed if cli.mode == "random" else None,
        "gold_used": False,
        "source_config_fingerprint": source_config["fingerprint"],
    }
    candidate = {**stable, "fingerprint": sha256_json(stable)}
    config_path = cli.output_root / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != candidate["fingerprint"]:
            raise RuntimeError("Output root contains a different subset configuration")
    else:
        atomic_write_json(config_path, candidate)

    source_records = read_records(cli.source_root)
    manifest_rows = []
    for item_id in selected_ids:
        for agent_id in ("A", "B"):
            source_path, source_row = source_records[(item_id, agent_id)]
            rank_name = source_path.parents[1].name
            destination = (
                cli.output_root / "prebeliefs" / rank_name / "records" / source_path.name
            )
            row = {
                **source_row,
                "config_fingerprint": candidate["fingerprint"],
                "subset_source_config_fingerprint": source_config["fingerprint"],
            }
            atomic_write_json(destination, row)
            prefix_relative = Path(str(source_row["statebridge_prefix_file"]))
            link_or_copy(
                cli.source_root / prefix_relative,
                cli.output_root / prefix_relative,
            )
            manifest_rows.append(
                {
                    "item_id": item_id,
                    "agent_id": agent_id,
                    "source_record": str(source_path.relative_to(cli.source_root)),
                    "prefix": str(prefix_relative),
                }
            )

    merged = merge_prebeliefs(cli.output_root, require_complete=True)
    atomic_write_json(
        cli.output_root / "subset_provenance.json",
        {
            "source_root": str(cli.source_root.resolve()),
            "source_config_fingerprint": source_config["fingerprint"],
            "subset_config_fingerprint": candidate["fingerprint"],
            "selection": stable["selection"],
            "selected_item_ids": selected_ids,
            "records": manifest_rows,
            "merged_prebelief_records": len(merged),
        },
    )
    print(
        f"materialized {len(selected_ids)} items / {len(merged)} prebeliefs "
        f"at {cli.output_root}"
    )


if __name__ == "__main__":
    main()
