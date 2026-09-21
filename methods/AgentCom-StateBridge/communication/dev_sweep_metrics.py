"""CR/PR/SI for the CR-DNC alpha sweep against StateBridge on the same pairs."""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path

AGENTS = ("A", "B", "C")
CATS = {"cr": "correction_opportunity", "pr": "destruction_risk",
        "sr": "both_wrong", "scr": "both_correct"}


def load(path, keep_items):
    rows = []
    for p in Path(path).glob("*/records/item_*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if int(r["item_id"]) in keep_items:
            rows.append(r)
    return rows


def metrics(rows, condition):
    sub = [r for r in rows if r["condition"] == condition]
    out = {}
    for key, cat in CATS.items():
        s = [r for r in sub if r["pair_classification"] == cat]
        n = len(s); k = sum(bool(r["receiver_post_correct"]) for r in s)
        out[key] = (k, n, (k / n if n else None))
    cr, pr = out["cr"][2], out["pr"][2]
    out["si"] = (None, None, (cr + pr) / 2 if cr is not None and pr is not None else None)
    acc = sum(bool(r["receiver_post_correct"]) for r in sub)
    out["acc"] = (acc, len(sub), acc / len(sub) if sub else None)
    out["adopt_pr"] = None
    prs = [r for r in sub if r["pair_classification"] == "destruction_risk"]
    if prs:
        out["adopt_pr"] = sum(
            1 for r in prs if r["receiver_post_answer"] == r["sender_pre_answer"]
        ) / len(prs)
    crs = [r for r in sub if r["pair_classification"] == "correction_opportunity"]
    out["adopt_cr"] = (
        sum(1 for r in crs if r["receiver_post_answer"] == r["sender_pre_answer"]) / len(crs)
        if crs else None
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("runs/medqa/cr_dnc_v1"))
    ap.add_argument("--artifact-root", type=Path,
                    default=Path("artifacts/icr_v3/medqa_full_seed42"))
    ap.add_argument("--subset", default="dev")
    cli = ap.parse_args()

    split = json.loads((cli.run_root / "split.json").read_text())
    keep = set(split[f"{cli.subset}_item_ids"])
    baseline = [json.loads(l) for l in (cli.artifact_root / "revisions/merged.jsonl")
                .read_text().split("\n") if l and json.loads(l)["item_id"] in keep]
    crdnc = load(cli.run_root / "icr_root" / "revisions", keep)

    print(f"{cli.subset}: {len(keep)} questions, baseline {len(baseline)} records, "
          f"CR-DNC {len(crdnc)} records\n")
    header = f"{'condition':<16}{'CR':>14}{'PR':>14}{'SI':>9}{'Acc_ret':>14}{'PR采纳':>9}"
    print(header); print("-" * len(header))
    rows = []
    for label, src, cond in (
        ("No Message", baseline, "none"),
        ("StateBridge", baseline, "true_statebridge"),
        ("Full Text", baseline, "true_text"),
        ("CR-DNC a=0.25", crdnc, "cr_dnc_a025"),
        ("CR-DNC a=0.50", crdnc, "cr_dnc_a050"),
        ("CR-DNC a=0.75", crdnc, "cr_dnc_a075"),
        ("CR-DNC a=1.00", crdnc, "cr_dnc_a100"),
        ("CTRL rand a=1.0", crdnc, "cr_dnc_randdir_a100"),
        ("CTRL rand-norm", crdnc, "cr_dnc_randnorm_a100"),
    ):
        m = metrics(src, cond)
        if m["cr"][1] == 0:
            continue
        rows.append((label, cond, m))
        f = lambda t: f"{t[0]}/{t[1]} {100*t[2]:.1f}%" if t[2] is not None else "-"
        print(f"{label:<16}{f(m['cr']):>14}{f(m['pr']):>14}"
              f"{100*m['si'][2]:>8.2f}%{f(m['acc']):>14}"
              f"{100*m['adopt_pr']:>8.1f}%" if m["adopt_pr"] is not None else "")
    base = dict(rows)[""] if False else None
    sb = next(m for l, c, m in rows if c == "true_statebridge")
    print(f"\n{'vs StateBridge':<16}{'dCR':>10}{'dPR':>10}{'dSI':>10}{'dPR采纳':>10}")
    for label, cond, m in rows:
        if not cond.startswith("cr_dnc"):
            continue
        print(f"{label:<16}{100*(m['cr'][2]-sb['cr'][2]):>+9.2f}"
              f"{100*(m['pr'][2]-sb['pr'][2]):>+10.2f}"
              f"{100*(m['si'][2]-sb['si'][2]):>+10.2f}"
              f"{100*(m['adopt_pr']-sb['adopt_pr']):>+10.2f}")
    print(f"\n注: dev CR/PR 分母各 {sb['cr'][1]}，翻一条 = {100/sb['cr'][1]:.2f}pp")


if __name__ == "__main__":
    main()
