"""Accuracy over the whole benchmark, not just the pairs phase 2 retained.

`icr.analysis_v2` reports accuracy over the directional pairs in merged.jsonl.
Those pairs come only from mixed items -- the ones where at least one agent was
wrong -- because phase 2 skips items every agent already answered correctly.
That makes the published figure a conditional accuracy on the hardest slice
(26-42%), not the benchmark accuracy anyone would quote.

Recovering the benchmark figure needs the rate at which a receiver on a skipped
item still answers correctly. That rate cannot be borrowed from the mixed
items: they are harder by construction, and borrowing pushed ARC-Challenge's
full-set accuracy 5.7 points too low and manufactured a 5.9-point gap between
channels that direct measurement puts at 0.6. So it is measured on a sample of
the skipped items themselves, written by icr.merge to
revisions/all_correct_sample.jsonl.

Two units are reported, because one item has no single post-communication
answer under this protocol -- three agents each revise once per incoming
message, so an item carries six revised answers:

  receiver   one revised answer per (item, sender -> receiver); denominator is
             items x 6. Defined for every task.
  item       each agent revises on one incoming message, then the three answers
             are majority-voted into one answer per item; averaged over the
             eight ways to assign one sender to each receiver. Undefined for
             code tasks, where answers are programs and three distinct strings
             never form a majority.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import AGENTS, DIRECTIONS
from .benchmarks import SPECS

# Config stores the benchmark label ("humanevalplus_test"), not the task key,
# so the answer type is looked up through the registry rather than matched
# against a hand-kept name list.
CODE_LABELS = {spec.label for spec in SPECS.values() if spec.answer_type == "code"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line
    ]


def _wilson_half_width(successes: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson interval, which stays finite at p = 1.

    Every sampled rate here sits at or near the ceiling, where the normal
    approximation collapses to a zero-width interval and would claim the
    extrapolation is exact.
    """
    if n <= 0:
        return 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(half, abs(p - centre) + half) if n < 30 else half


def compute(source_root: Path, conditions: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    task = str(config.get("benchmark") or config.get("task") or "")

    beliefs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(source_root / "prebeliefs" / "merged.jsonl"):
        beliefs[int(row["item_id"])][str(row["agent_id"])] = row
    items = {i: v for i, v in beliefs.items() if len(v) == len(AGENTS)}
    if not items:
        raise RuntimeError("no items carry a belief from every agent")
    all_correct = {i for i, v in items.items() if all(v[a]["correct"] for a in AGENTS)}

    per_item = len(DIRECTIONS)
    total = len(items) * per_item
    skipped = len(all_correct) * per_item

    mixed_rows = [
        r for r in _read_jsonl(source_root / "revisions" / "merged.jsonl")
        if int(r["item_id"]) in items
    ]
    sample_rows = [
        r for r in _read_jsonl(source_root / "revisions" / "all_correct_sample.jsonl")
        if int(r["item_id"]) in all_correct
    ]

    gold = {i: v[AGENTS[0]]["gold"] for i, v in items.items()}
    post: dict[tuple[int, str, str], Any] = {
        (int(r["item_id"]), str(r["direction"]), str(r["condition"])): r
        for r in (*mixed_rows, *sample_rows)
    }

    pre_correct = sum(
        bool(v[a]["correct"]) for v in items.values() for a in AGENTS
    )
    result: dict[str, Any] = {
        "task": task,
        "items": len(items),
        "records_per_item": per_item,
        "total_records": total,
        "all_correct_items": len(all_correct),
        "skipped_records": skipped,
        "pre_communication_accuracy": pre_correct / (len(items) * len(AGENTS)),
        "item_unit_defined": task not in CODE_LABELS,
        "conditions": {},
    }

    for condition in conditions:
        mixed = [r for r in mixed_rows if r["condition"] == condition]
        sample = [r for r in sample_rows if r["condition"] == condition]
        n_sample = len(sample)
        c_sample = sum(bool(r["receiver_post_correct"]) for r in sample)
        c_mixed = sum(bool(r["receiver_post_correct"]) for r in mixed)
        unmeasured = skipped - n_sample

        entry: dict[str, Any] = {
            "mixed_correct": c_mixed,
            "mixed_records": len(mixed),
            "sample_correct": c_sample,
            "sample_records": n_sample,
            "unmeasured_records": unmeasured,
        }
        if n_sample:
            scr = c_sample / n_sample
            acc = (c_mixed + c_sample + unmeasured * scr) / total
            half = _wilson_half_width(c_sample, n_sample) * unmeasured / total
            entry.update(
                {
                    "skipped_item_accuracy": scr,
                    "accuracy": acc,
                    "accuracy_ci95": [max(acc - half, 0.0), min(acc + half, 1.0)],
                    "extrapolated": unmeasured > 0,
                }
            )
        else:
            # Nothing measured on the skipped population: report the interval
            # the data actually supports instead of inventing a point estimate.
            entry.update(
                {
                    "skipped_item_accuracy": None,
                    "accuracy": None,
                    "accuracy_ci95": None,
                    "accuracy_bounds": [
                        (c_mixed) / total,
                        (c_mixed + skipped) / total,
                    ],
                    "extrapolated": True,
                }
            )
        result["conditions"][condition] = entry

    if result["item_unit_defined"]:
        _add_item_unit(result, items, all_correct, gold, post, conditions)
    return result


def _add_item_unit(result, items, all_correct, gold, post, conditions) -> None:
    """Majority vote over the three agents, averaged across sender assignments."""
    mixed = [i for i in items if i not in all_correct]
    assignments = list(
        itertools.product(*[[s for s in AGENTS if s != r] for r in AGENTS])
    )
    for condition in conditions:
        # Rate at which a skipped item survives as a majority-correct item,
        # measured on whichever skipped items the sample covers in full.
        covered, survived = 0, 0
        for item in all_correct:
            answers = [
                post.get((item, f"{s}_to_{r}", condition)) for r in AGENTS
                for s in AGENTS if s != r
            ]
            if any(a is None for a in answers):
                continue
            covered += 1
            survived += _majority_ok(item, assignments, post, condition, gold, mean=True)
        survival = survived / covered if covered else None

        scores = []
        for pick in assignments:
            good = 0
            for item in mixed:
                votes = Counter()
                for receiver, sender in zip(AGENTS, pick):
                    row = post.get((item, f"{sender}_to_{receiver}", condition))
                    if row is not None:
                        votes[row["receiver_post_answer"]] += 1
                if votes:
                    top, n = votes.most_common(1)[0]
                    if n >= 2 and top == gold[item]:
                        good += 1
            scores.append(good)
        base = sum(scores) / len(scores)
        rate = 1.0 if survival is None else survival
        entry = result["conditions"][condition]
        entry["item_unit_accuracy"] = (base + len(all_correct) * rate) / len(items)
        entry["item_unit_range"] = [
            (min(scores) + len(all_correct) * rate) / len(items),
            (max(scores) + len(all_correct) * rate) / len(items),
        ]
        entry["item_unit_skipped_measured"] = covered
        entry["item_unit_skipped_survival"] = survival


def _majority_ok(item, assignments, post, condition, gold, mean=False):
    hits = 0
    for pick in assignments:
        votes = Counter()
        for receiver, sender in zip(AGENTS, pick):
            row = post.get((item, f"{sender}_to_{receiver}", condition))
            if row is not None:
                votes[row["receiver_post_answer"]] += 1
        if votes:
            top, n = votes.most_common(1)[0]
            if n >= 2 and top == gold[item]:
                hits += 1
    return hits / len(assignments) if mean else hits
