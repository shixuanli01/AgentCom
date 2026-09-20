#!/usr/bin/env python3
"""Raise an existing run's token budget without discarding its finished records.

max_new_tokens decides only when generation stops, not how it samples, so a
record that reached EOS under the old budget is exactly what the same seed would
produce under a larger one. Only the truncated records need regenerating, which
on GPQA is 28% of the prebeliefs and on ARC-Challenge 5%.

Raising the budget changes the config fingerprint, which the resume checks use
to reject stale artifacts. The old fingerprint is therefore recorded as
superseded so the finished records stay valid, and the runners are told to
regenerate truncated ones with --rerun-truncated.

Usage:
    python scripts/raise_token_budget.py ARTIFACT_ROOT NEW_MAX_NEW_TOKENS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icr.prebeliefs import fingerprint_payload
from icr.protocol import atomic_write_json, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("max_new_tokens", type=int)
    cli = parser.parse_args()

    path = cli.root / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    current = int(config["generation"]["max_new_tokens"])
    if cli.max_new_tokens < current:
        raise SystemExit(
            f"Refusing to lower the budget: {current} -> {cli.max_new_tokens}. "
            "Only an increase leaves finished records valid."
        )
    if cli.max_new_tokens == current:
        # A repair may be rerun after a failure part-way through, so reaching the
        # target budget again is success, not an error. Treating it as an error
        # once aborted a repair that had already deleted the revisions it was
        # about to regenerate.
        print(f"{cli.root.name}: already at max_new_tokens {current}; nothing to raise")
        return

    old_fingerprint = config["fingerprint"]
    raised = {**config, "generation": {**config["generation"], "max_new_tokens": cli.max_new_tokens}}
    # The fingerprint must cover exactly what build_config hashes, or the
    # rebuilt candidate will never match what is on disk.
    fingerprint = sha256_json(fingerprint_payload(raised))
    superseded = list(
        dict.fromkeys([*config.get("superseded_fingerprints", []), old_fingerprint])
    )
    updated = {
        **raised,
        "superseded_fingerprints": superseded,
        "max_new_tokens_history": [
            *config.get("max_new_tokens_history", []),
            {
                "from": current,
                "to": cli.max_new_tokens,
                "reason": "truncated generations never stated an answer and were scored wrong",
            },
        ],
        "fingerprint": fingerprint,
    }
    atomic_write_json(path, updated)

    print(f"{cli.root.name}: max_new_tokens {current} -> {cli.max_new_tokens}")
    print(f"  fingerprint {old_fingerprint[:16]} -> {updated['fingerprint'][:16]}")
    print(f"  superseded fingerprints kept valid: {len(superseded)}")
    print("  rerun with --rerun-truncated to regenerate only the truncated records")


if __name__ == "__main__":
    main()
