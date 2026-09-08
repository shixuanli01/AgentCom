#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.state_bridge import load_dataset_by_name


TASKS = ("arc_challenge", "medqa", "gsm8k", "mbppplus", "humanevalplus")


def canonical_hash(rows):
    payload = json.dumps(
        rows,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproduction/reports/generated/dataset_audit.json"),
    )
    args = parser.parse_args()

    report = {}
    for task in TASKS:
        rows = load_dataset_by_name(task)
        report[task] = {
            "rows": len(rows),
            "canonical_loader_sha256": canonical_hash(rows),
            "keys": sorted(rows[0].keys()) if rows else [],
        }
        print(task, report[task])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
