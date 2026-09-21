"""Freeze the question-level development / evaluation split for CR-DNC.

The split must exist before any CR-DNC performance is observed, and must never
be revised afterwards. It is question-level: all six directed interactions of a
question stay together, because they share the question's beliefs and are not
independent samples.

Assignment is a deterministic hash of "medqa|<item_id>|cr_dnc_split_v1", so it
is reproducible without storing a random seed and does not depend on iteration
order. Stratification is by k, the number of agents that answered the question
correctly in phase 1; k is a property of the frozen phase-1 artifacts, and
using it here is an offline design choice, not information available to message
construction.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

AGENTS = ("A", "B", "C")
SPLIT_TAG = "cr_dnc_split_v1"
DEV_FRACTION = 0.20


def _hash_rank(dataset: str, item_id: int) -> int:
    payload = f"{dataset}|{item_id}|{SPLIT_TAG}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_split(artifact_root: Path, dataset: str = "medqa") -> dict:
    beliefs: dict[int, dict[str, bool]] = collections.defaultdict(dict)
    for line in (artifact_root / "prebeliefs" / "merged.jsonl").read_text(
        encoding="utf-8"
    ).split("\n"):
        if not line:
            continue
        row = json.loads(line)
        beliefs[int(row["item_id"])][str(row["agent_id"])] = bool(row["correct"])

    complete = {i: v for i, v in beliefs.items() if len(v) == len(AGENTS)}
    # Retained questions only: where every agent was already correct there is no
    # correction or destruction pair, and phase 2 does not run them.
    retained = {
        i: sum(v.values()) for i, v in complete.items() if not all(v.values())
    }

    by_k: dict[int, list[int]] = collections.defaultdict(list)
    for item_id, k in retained.items():
        by_k[k].append(item_id)

    dev: list[int] = []
    for k, items in sorted(by_k.items()):
        # Rank inside the stratum by the deterministic hash and take the lowest
        # ranks, so each stratum contributes its share exactly rather than in
        # expectation.
        ranked = sorted(items, key=lambda i: _hash_rank(dataset, i))
        n_dev = round(len(ranked) * DEV_FRACTION)
        dev.extend(ranked[:n_dev])

    dev_set = set(dev)
    evaluation = sorted(i for i in retained if i not in dev_set)

    return {
        "split_tag": SPLIT_TAG,
        "dataset": dataset,
        "artifact_root": str(artifact_root),
        "rule": f'sha256("{dataset}|<item_id>|{SPLIT_TAG}")[:8], ranked within each k stratum',
        "dev_fraction_target": DEV_FRACTION,
        "population": "retained questions only (not every agent correct in phase 1)",
        "unit": "question; all six directed pairs of a question stay in one split",
        "stratification": "k = number of agents correct in phase 1 (offline property, not used by message construction)",
        "counts": {
            "retained_questions": len(retained),
            "dev_questions": len(dev_set),
            "eval_questions": len(evaluation),
            "by_k": {
                str(k): {
                    "total": len(items),
                    "dev": sum(1 for i in items if i in dev_set),
                    "eval": sum(1 for i in items if i not in dev_set),
                }
                for k, items in sorted(by_k.items())
            },
        },
        "dev_item_ids": sorted(dev_set),
        "eval_item_ids": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("artifacts/icr_v3/medqa_full_seed42"))
    parser.add_argument("--out", type=Path,
                        default=Path("runs/medqa/cr_dnc_v1/split.json"))
    cli = parser.parse_args()

    if cli.out.exists():
        raise SystemExit(
            f"{cli.out} already exists. The split is frozen once written; "
            "delete it deliberately if it must be rebuilt."
        )
    split = build_split(cli.artifact_root, dataset="medqa")
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    c = split["counts"]
    print(f"wrote {cli.out}")
    print(f"  retained {c['retained_questions']}  dev {c['dev_questions']}  eval {c['eval_questions']}")
    for k, v in c["by_k"].items():
        print(f"    k={k}: total {v['total']:>3}  dev {v['dev']:>2}  eval {v['eval']:>3}")


if __name__ == "__main__":
    main()
