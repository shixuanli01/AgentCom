#!/usr/bin/env python3
"""Report what phase 2 could possibly measure, from phase-1 beliefs alone.

CR and FCS exist only on items where the sender is right and the receiver wrong;
PR and FWS only on the mirror image; SR only on items both agents got wrong.
Those denominators are fixed by the independent beliefs, so a dataset's
diagnostic yield is knowable before a single revision is generated.

Usage:
    python scripts/phase1_yield.py artifacts/icr_v3/*_full_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path) -> dict | None:
    merged = root / "prebeliefs" / "merged.jsonl"
    config = root / "config.json"
    if not merged.is_file() or not config.is_file():
        return None
    rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").split("\n") if line]
    if not rows:
        return None
    cfg = json.loads(config.read_text(encoding="utf-8"))
    selected = [int(v) for v in cfg["selected_item_ids"]]
    correct = {(int(r["item_id"]), str(r["agent_id"])): bool(r["correct"]) for r in rows}
    if len(correct) < 2 * len(selected):
        selected = sorted({item for item, _ in correct})

    both_correct = both_wrong = disagree = 0
    for item in selected:
        a, b = correct.get((item, "A")), correct.get((item, "B"))
        if a is None or b is None:
            continue
        if a and b:
            both_correct += 1
        elif not a and not b:
            both_wrong += 1
        else:
            disagree += 1

    truncated = sum(1 for r in rows if not r.get("hit_eos", True))
    unparsed = sum(1 for r in rows if r.get("parsed_answer") is None)
    items = both_correct + both_wrong + disagree
    keep_one_in = int((cfg.get("both_correct_sampling") or {}).get("keep_one_in", 10))
    sampled = sum(
        1
        for item in selected
        if correct.get((item, "A")) and correct.get((item, "B")) and item % keep_one_in == 0
    )
    retained = disagree + both_wrong + sampled
    return {
        "benchmark": cfg.get("benchmark", root.name),
        "items": items,
        "accuracy": sum(correct.values()) / max(len(correct), 1),
        "oracle2": both_correct + disagree,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "disagree": disagree,
        # Each disagreeing item contributes one correction case in one direction
        # and one destruction case in the other.
        "cr_pr_denominator": disagree,
        "sr_denominator": both_wrong * 2,
        "retained_items": retained,
        "phase2_records": retained * 8,
        "phase2_full": items * 8,
        # A truncated generation never states its answer, parses as None, and is
        # then scored wrong, so it inflates both_wrong and the correction subset
        # with a formatting failure rather than a reasoning one.
        "truncated": truncated / max(len(correct), 1),
        "unparsed": unparsed / max(len(correct), 1),
        "cap": int((cfg.get("generation") or {}).get("max_new_tokens") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    cli = parser.parse_args()

    reports = [r for r in (load(root) for root in cli.roots) if r]
    if not reports:
        print("No completed phase-1 caches found.")
        return

    print(
        f"{'benchmark':<18}{'items':>6}{'cap':>7}{'截断':>7}{'pre-acc':>9}"
        f"{'both✓':>7}{'both✗':>7}{'分歧':>6}{'CR/PR分母':>10}{'SR分母':>8}{'phase2':>9}"
    )
    print("-" * 100)
    for r in sorted(reports, key=lambda x: -x["cr_pr_denominator"]):
        print(
            f"{r['benchmark']:<18}{r['items']:>6}{r['cap']:>7}{100 * r['truncated']:>6.1f}%"
            f"{100 * r['accuracy']:>8.1f}%"
            f"{r['both_correct']:>7}{r['both_wrong']:>7}{r['disagree']:>6}"
            f"{r['cr_pr_denominator']:>10}{r['sr_denominator']:>8}"
            f"{r['phase2_records']:>9}"
        )
    print()
    print("CR/PR 分母 = A/B 正确性分歧的题数；每题在一个方向贡献一条 correction、")
    print("另一个方向贡献一条 destruction。分母越小，通信行为越测不准。")
    print("截断 = 生成到 token 上限仍未给出答案；这类记录会被判错并污染 both✗ 与分歧子集。")


if __name__ == "__main__":
    main()
