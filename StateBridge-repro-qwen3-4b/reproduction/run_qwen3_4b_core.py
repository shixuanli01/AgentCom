#!/usr/bin/env python3
"""Run the five published Qwen3-4B StateBridge tasks one at a time."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_ROWS = {
    "arc_challenge": 1172,
    "medqa": 300,
    "gsm8k": 1319,
    "mbppplus": 378,
    "humanevalplus": 164,
}

DEFAULT_ORDER = (
    "medqa",
    "humanevalplus",
    "mbppplus",
    "arc_challenge",
    "gsm8k",
)


def is_complete(path: Path, task: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    config = result.get("config", {})
    return (
        result.get("total") == EXPECTED_ROWS[task]
        and len(result.get("results", [])) == EXPECTED_ROWS[task]
        and config.get("model_name") == "Qwen/Qwen3-4B"
        and config.get("task") == task
        and config.get("seed") == seed
        and config.get("max_prefix_tokens") == 64
        and config.get("snap_ratio") == 0.3
        and config.get("adaptive_reg") == 1e-3
        and config.get("temperature") == 0.6
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=EXPECTED_ROWS, default=DEFAULT_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=Path("reproduction/runs/main_seed42"))
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "Use upstream log resume for a partial task. This restarts the RNG stream and "
            "must be reported as a resumed run, not an uninterrupted author-style run."
        ),
    )
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        result_path = args.run_dir / f"{task}.json"
        log_path = args.run_dir / f"{task}.log"
        if is_complete(result_path, task, args.seed):
            print(f"[skip complete] {task}: {result_path}", flush=True)
            continue
        if log_path.exists() and not args.resume_partial:
            raise RuntimeError(
                f"Partial log exists for {task}: {log_path}. Remove/archive it for a clean "
                "rerun, or pass --resume-partial and report the changed RNG stream."
            )

        command = [
            sys.executable,
            "-u",
            "-m",
            "methods.state_bridge",
            "--model",
            "Qwen/Qwen3-4B",
            "--task",
            task,
            "--gpus",
            str(args.gpu),
            "--seed",
            str(args.seed),
            "--resume",
            str(log_path),
            "--output",
            str(result_path),
        ]
        print(f"[start] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        if not is_complete(result_path, task, args.seed):
            raise RuntimeError(f"Task exited without a complete validated result: {task}")
        print(f"[complete] {task}: {result_path}", flush=True)


if __name__ == "__main__":
    main()
