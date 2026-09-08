"""Fixed-width multi-path evaluation on top of the frozen StateBridge method."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from agentcom import UPSTREAM_STATEBRIDGE_COMMIT
from methods.state_bridge import StateBridge, TASK_CONFIG, load_dataset_by_name
from models import ModelWrapper
from utils import set_seed


ROLES = ("planner", "critic", "refiner", "judger")
HANDOFF_ROLES = ROLES[:-1]
INVALID_LABEL = "INVALID"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(
    base_seed: int,
    dataset: str,
    item_id: int,
    branch_id: int,
    stage: str,
) -> int:
    """Derive a process-independent Torch-compatible seed."""
    payload = f"{base_seed}\0{dataset}\0{item_id}\0{branch_id}\0{stage}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def branch_seeds(
    base_seed: int,
    dataset: str,
    item_id: int,
    branch_id: int,
) -> Dict[str, int]:
    return {
        role: stable_seed(base_seed, dataset, item_id, branch_id, role)
        for role in ROLES
    }


def canonical_prediction(value: Optional[str]) -> str:
    if value is None:
        return INVALID_LABEL
    normalized = str(value).strip().lower()
    return normalized if normalized else INVALID_LABEL


def plurality_vote(predictions: Sequence[Optional[str]]) -> Dict[str, Any]:
    """Vote with earliest-branch tie breaking, as preregistered for V1."""
    if not predictions:
        raise ValueError("At least one prediction is required")
    labels = [canonical_prediction(value) for value in predictions]
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}
    winner = next(label for label in labels if label in tied)
    return {
        "winner": winner,
        "counts": dict(sorted(counts.items())),
        "tied": len(tied) > 1,
        "tied_labels": sorted(tied),
    }


def pairwise_disagreement(predictions: Sequence[Optional[str]]) -> float:
    labels = [canonical_prediction(value) for value in predictions]
    pairs = math.comb(len(labels), 2)
    if pairs == 0:
        return 0.0
    disagreements = sum(
        labels[left] != labels[right]
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    )
    return disagreements / pairs


def exact_mcnemar_p(corrections: int, harms: int) -> float:
    """Two-sided exact McNemar p-value under a Binomial(n, 0.5) null."""
    discordant = corrections + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(corrections, harms) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_delta_interval(
    paired_deltas: Sequence[int],
    *,
    seed: int = 42,
    samples: int = 10_000,
) -> List[float]:
    if not paired_deltas:
        return [0.0, 0.0]
    values = np.asarray(paired_deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    cpu_tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in tensors.items()
    }
    save_file(cpu_tensors, str(temporary))
    os.replace(temporary, path)


def current_git_commit(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class InstrumentedStateBridge(StateBridge):
    """StateBridge with per-stage seeding and non-invasive handoff capture."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stage_seeds: Dict[str, int] = {}
        self._generation_index = 0
        self._alignment_index = 0
        self._captured_tensors: Dict[str, torch.Tensor] = {}
        self._handoff_metadata: List[Dict[str, Any]] = []

    def begin_branch(self, seeds: Mapping[str, int]) -> None:
        missing = set(ROLES) - set(seeds)
        if missing:
            raise ValueError(f"Missing stage seeds: {sorted(missing)}")
        self._stage_seeds = {role: int(seeds[role]) for role in ROLES}
        self._generation_index = 0
        self._alignment_index = 0
        self._captured_tensors = {}
        self._handoff_metadata = []

    def _generate_with_prefix(self, *args: Any, **kwargs: Any):
        if self._generation_index >= len(ROLES):
            raise RuntimeError("StateBridge generated more stages than expected")
        role = ROLES[self._generation_index]
        set_seed(self._stage_seeds[role])
        output = super()._generate_with_prefix(*args, **kwargs)
        _, _, token_ids, _ = output
        if token_ids is not None and token_ids.numel() > 0:
            self._captured_tensors[f"{role}_generated_token_ids"] = (
                token_ids.detach().cpu().contiguous()
            )
        self._generation_index += 1
        return output

    def _align_hidden_sequence(
        self,
        hidden_seq: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._alignment_index >= len(HANDOFF_ROLES):
            raise RuntimeError("StateBridge aligned more handoffs than expected")
        role = HANDOFF_ROLES[self._alignment_index]
        self._captured_tensors[f"{role}_transfer_hidden"] = (
            hidden_seq.detach().cpu().contiguous()
        )
        self._captured_tensors[f"{role}_transfer_token_ids"] = (
            token_ids.detach().cpu().contiguous()
        )
        self._handoff_metadata.append(
            {
                "sender_role": role,
                "prefix_length": int(hidden_seq.shape[1]),
                "hidden_size": int(hidden_seq.shape[2]),
                "hidden_dtype": str(hidden_seq.dtype),
                "token_count": int(token_ids.shape[1]),
            }
        )
        self._alignment_index += 1
        return super()._align_hidden_sequence(hidden_seq, token_ids)

    def captured_tensors(self) -> Dict[str, torch.Tensor]:
        return dict(self._captured_tensors)

    def handoff_metadata(self) -> List[Dict[str, Any]]:
        return list(self._handoff_metadata)


def summarize_item(
    item_index: int,
    gold: str,
    branch_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not branch_results:
        raise ValueError("Cannot summarize an item without branch results")
    predictions = [result.get("prediction") for result in branch_results]
    correctness = [canonical_prediction(pred) == canonical_prediction(gold) for pred in predictions]
    vote_all = plurality_vote(predictions)
    vote_three = plurality_vote(predictions[:3]) if len(predictions) >= 3 else None
    branch_zero_correct = correctness[0]
    vote_correct = vote_all["winner"] == canonical_prediction(gold)
    return {
        "item_index": item_index,
        "gold": canonical_prediction(gold),
        "branch_predictions": [canonical_prediction(pred) for pred in predictions],
        "branch_correct": correctness,
        "branch_0_correct": branch_zero_correct,
        "mean_branch_accuracy": sum(correctness) / len(correctness),
        "oracle_at_m": any(correctness),
        "vote_at_m": vote_all,
        "vote_at_m_correct": vote_correct,
        "vote_at_3": vote_three,
        "vote_at_3_correct": (
            vote_three["winner"] == canonical_prediction(gold)
            if vote_three is not None
            else None
        ),
        "unique_prediction_count": len(set(canonical_prediction(pred) for pred in predictions)),
        "pairwise_disagreement": pairwise_disagreement(predictions),
        "correction": (not branch_zero_correct) and vote_correct,
        "harm": branch_zero_correct and (not vote_correct),
    }


def aggregate_items(items: Sequence[Mapping[str, Any]], *, base_seed: int) -> Dict[str, Any]:
    if not items:
        return {"total": 0}
    branch_count = len(items[0]["branch_correct"])
    for item in items:
        if len(item["branch_correct"]) != branch_count:
            raise ValueError("Inconsistent branch count in item summaries")

    total = len(items)
    branch_correct = [
        sum(bool(item["branch_correct"][branch]) for item in items)
        for branch in range(branch_count)
    ]
    branch_zero_correct = sum(bool(item["branch_0_correct"]) for item in items)
    vote_correct = sum(bool(item["vote_at_m_correct"]) for item in items)
    oracle_correct = sum(bool(item["oracle_at_m"]) for item in items)
    corrections = sum(bool(item["correction"]) for item in items)
    harms = sum(bool(item["harm"]) for item in items)
    paired_deltas = [
        int(bool(item["vote_at_m_correct"])) - int(bool(item["branch_0_correct"]))
        for item in items
    ]
    vote_three_values = [item.get("vote_at_3_correct") for item in items]
    vote_three_correct = sum(value is True for value in vote_three_values)

    return {
        "total": total,
        "branches": branch_count,
        "branch_0": {
            "correct": branch_zero_correct,
            "accuracy": branch_zero_correct / total,
        },
        "per_branch": [
            {"branch": branch, "correct": count, "accuracy": count / total}
            for branch, count in enumerate(branch_correct)
        ],
        "mean_branch_accuracy": sum(branch_correct) / (total * branch_count),
        "vote_at_3": (
            {"correct": vote_three_correct, "accuracy": vote_three_correct / total}
            if all(value is not None for value in vote_three_values)
            else None
        ),
        "vote_at_m": {
            "correct": vote_correct,
            "accuracy": vote_correct / total,
        },
        "oracle_at_m": {
            "correct": oracle_correct,
            "accuracy": oracle_correct / total,
        },
        "delta_vote_minus_branch_0": (vote_correct - branch_zero_correct) / total,
        "corrections": corrections,
        "harms": harms,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(corrections, harms),
        "paired_bootstrap_95ci": bootstrap_delta_interval(
            paired_deltas,
            seed=base_seed,
        ),
        "tie_items": sum(bool(item["vote_at_m"]["tied"]) for item in items),
        "invalid_predictions": sum(
            prediction == INVALID_LABEL
            for item in items
            for prediction in item["branch_predictions"]
        ),
        "mean_unique_predictions": sum(item["unique_prediction_count"] for item in items) / total,
        "mean_pairwise_disagreement": sum(item["pairwise_disagreement"] for item in items) / total,
    }


class MultiPathRunner:
    def __init__(
        self,
        bridge: InstrumentedStateBridge,
        *,
        task: str,
        branches: int,
        base_seed: int,
        run_dir: Path,
        manifest_fingerprint: str,
    ) -> None:
        if branches < 1:
            raise ValueError("branches must be positive")
        self.bridge = bridge
        self.task = task
        self.branches = branches
        self.base_seed = base_seed
        self.run_dir = run_dir
        self.manifest_fingerprint = manifest_fingerprint
        self.stop_requested = False
        self.log_path = run_dir / "run.log"

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.log(f"Stop requested by signal {signum}; finishing the current branch")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _trace_path(self, item_index: int, branch_id: int) -> Path:
        return self.run_dir / "traces" / f"item_{item_index:04d}_branch_{branch_id}.json"

    def _state_path(self, item_index: int, branch_id: int) -> Path:
        return self.run_dir / "states" / f"item_{item_index:04d}_branch_{branch_id}.safetensors"

    def _item_path(self, item_index: int) -> Path:
        return self.run_dir / "items" / f"item_{item_index:04d}.json"

    def _load_completed_branch(self, item_index: int, branch_id: int) -> Optional[Dict[str, Any]]:
        trace_path = self._trace_path(item_index, branch_id)
        state_path = self._state_path(item_index, branch_id)
        if not trace_path.exists() or not state_path.exists():
            return None
        try:
            value = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            value.get("status") != "complete"
            or value.get("item_index") != item_index
            or value.get("branch_id") != branch_id
            or value.get("manifest_fingerprint") != self.manifest_fingerprint
        ):
            return None
        return value

    def run_branch(self, item_index: int, item: Mapping[str, Any], branch_id: int) -> Dict[str, Any]:
        cached = self._load_completed_branch(item_index, branch_id)
        if cached is not None:
            self.log(f"resume item={item_index} branch={branch_id} prediction={cached.get('prediction')}")
            return cached

        seeds = branch_seeds(self.base_seed, self.task, item_index, branch_id)
        self.bridge.begin_branch(seeds)
        branch_item = dict(item)
        branch_item["idx"] = item_index
        started = time.time()
        self.log(f"start item={item_index} branch={branch_id} seeds={seeds}")
        try:
            result = self.bridge.run_item(branch_item)
        except BaseException as error:
            failure = {
                "status": "error",
                "manifest_fingerprint": self.manifest_fingerprint,
                "item_index": item_index,
                "branch_id": branch_id,
                "stage_seeds": seeds,
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
            }
            atomic_write_json(self._trace_path(item_index, branch_id), failure)
            raise

        tensors = self.bridge.captured_tensors()
        if len(self.bridge.handoff_metadata()) != len(HANDOFF_ROLES):
            raise RuntimeError(
                f"Expected {len(HANDOFF_ROLES)} handoffs, got "
                f"{len(self.bridge.handoff_metadata())}"
            )
        atomic_save_tensors(self._state_path(item_index, branch_id), tensors)
        record = {
            "status": "complete",
            "manifest_fingerprint": self.manifest_fingerprint,
            "item_index": item_index,
            "branch_id": branch_id,
            "question_hash": hashlib.sha256(item["question"].encode("utf-8")).hexdigest(),
            "stage_seeds": seeds,
            "duration": time.time() - started,
            "handoffs": self.bridge.handoff_metadata(),
            **result,
        }
        atomic_write_json(self._trace_path(item_index, branch_id), record)
        self.log(
            f"complete item={item_index} branch={branch_id} "
            f"prediction={record.get('prediction')} correct={record.get('correct')} "
            f"duration={record['duration']:.2f}s"
        )
        return record

    def run(self, indexed_items: Iterable[tuple[int, Mapping[str, Any]]]) -> Dict[str, Any]:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        summaries: List[Dict[str, Any]] = []
        for item_index, item in indexed_items:
            branch_results = []
            for branch_id in range(self.branches):
                branch_results.append(self.run_branch(item_index, item, branch_id))
                if self.stop_requested:
                    self.log("Stopped at a durable branch boundary")
                    return self.write_summary(summaries, status="paused")
            item_summary = summarize_item(item_index, item["gold"], branch_results)
            atomic_write_json(self._item_path(item_index), item_summary)
            summaries.append(item_summary)
            self.log(
                f"item={item_index} branch0={item_summary['branch_predictions'][0]} "
                f"vote={item_summary['vote_at_m']['winner']} gold={item_summary['gold']} "
                f"oracle={item_summary['oracle_at_m']}"
            )
            self.write_summary(summaries, status="running")
        return self.write_summary(summaries, status="complete")

    def write_summary(self, items: Sequence[Mapping[str, Any]], *, status: str) -> Dict[str, Any]:
        aggregate = aggregate_items(items, base_seed=self.base_seed)
        summary = {
            "status": status,
            "manifest_fingerprint": self.manifest_fingerprint,
            "updated_at": datetime.now().isoformat(),
            "metrics": aggregate,
            "items": list(items),
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        return summary


def build_manifest(
    *,
    protocol: str,
    repo_root: Path,
    model: str,
    task: str,
    data: Sequence[Mapping[str, Any]],
    branches: int,
    base_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_prefix_tokens: int,
    adaptive_reg: float,
    snap_ratio: float,
    start_index: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    prompts_path = repo_root / "prompts.py"
    implementation_path = Path(__file__).resolve()
    manifest = {
        "protocol": protocol,
        "created_at": datetime.now().isoformat(),
        "upstream_commit": UPSTREAM_STATEBRIDGE_COMMIT,
        "method_commit": current_git_commit(repo_root),
        "implementation_hash": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
        "model": model,
        "task": task,
        "dataset_items": len(data),
        "dataset_hash": sha256_json(
            [{"question": item["question"], "gold": item["gold"]} for item in data]
        ),
        "prompt_hash": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "branches": branches,
        "base_seed": base_seed,
        "start_index": start_index,
        "limit": limit,
        "configuration": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "max_prefix_tokens": max_prefix_tokens,
            "enable_thinking": True,
            "prefix_strategy": "scale",
            "prefix_scale": 1.0,
            "adaptive_reg": adaptive_reg,
            "snap_ratio": snap_ratio,
            "use_hook": True,
            "branch_execution": "sequential",
            "vote_tie_break": "earliest_branch",
        },
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    manifest["fingerprint"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "created_at"}
    )
    return manifest


def ensure_manifest(run_dir: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    path = run_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != candidate.get("fingerprint"):
            raise RuntimeError(
                "Run directory contains a different manifest: "
                f"{existing.get('fingerprint')} != {candidate.get('fingerprint')}"
            )
        return existing
    atomic_write_json(path, candidate)
    return dict(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MP-StateBridge V1 evaluator")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--task", default="medqa", choices=sorted(TASK_CONFIG))
    parser.add_argument("--branches", type=int, default=5)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Allow non-protocol branch counts for throughput benchmarks",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prefix-tokens", type=int, default=64)
    parser.add_argument("--adaptive-reg", type=float, default=1e-3)
    parser.add_argument("--snap-ratio", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    cli = parse_args()
    if cli.branches != 5 and not cli.benchmark:
        raise ValueError("MP-SB-V1 freezes --branches at 5")
    if not torch.cuda.is_available():
        raise RuntimeError("MP-StateBridge requires a CUDA device")
    torch.cuda.set_device(cli.gpu)

    repo_root = Path(__file__).resolve().parents[1]
    full_data = load_dataset_by_name(cli.task)
    start = cli.start_index
    stop = len(full_data) if cli.limit is None else min(len(full_data), start + cli.limit)
    selected = list(enumerate(full_data))[start:stop]
    if not selected:
        raise ValueError("The selected dataset slice is empty")
    max_new_tokens = cli.max_new_tokens or TASK_CONFIG[cli.task]["max_new_tokens"]
    candidate_manifest = build_manifest(
        protocol=("MP-SB-CONCURRENCY-BENCHMARK" if cli.benchmark else "MP-SB-V1"),
        repo_root=repo_root,
        model=cli.model,
        task=cli.task,
        data=full_data,
        branches=cli.branches,
        base_seed=cli.base_seed,
        max_new_tokens=max_new_tokens,
        temperature=cli.temperature,
        top_p=cli.top_p,
        max_prefix_tokens=cli.max_prefix_tokens,
        adaptive_reg=cli.adaptive_reg,
        snap_ratio=cli.snap_ratio,
        start_index=cli.start_index,
        limit=cli.limit,
    )
    manifest = ensure_manifest(cli.run_dir, candidate_manifest)

    args = argparse.Namespace(
        model=cli.model,
        model_name=cli.model,
        task=cli.task,
        prompt="sequential",
        batch_size=1,
    )
    model = ModelWrapper(cli.model, device=torch.device(f"cuda:{cli.gpu}"), args=args)
    bridge = InstrumentedStateBridge(
        model,
        max_new_tokens=max_new_tokens,
        temperature=cli.temperature,
        top_p=cli.top_p,
        max_prefix_tokens=cli.max_prefix_tokens,
        enable_thinking=True,
        prefix_strategy="scale",
        adaptive_reg=cli.adaptive_reg,
        snap_ratio=cli.snap_ratio,
        use_hook=True,
        args=args,
    )
    runner = MultiPathRunner(
        bridge,
        task=cli.task,
        branches=cli.branches,
        base_seed=cli.base_seed,
        run_dir=cli.run_dir,
        manifest_fingerprint=manifest["fingerprint"],
    )
    runner.log(
        f"MP-SB-V1 start task={cli.task} items={start}:{stop} "
        f"branches={cli.branches} model={cli.model}"
    )
    summary = runner.run(selected)
    runner.log(f"MP-SB-V1 status={summary['status']} metrics={summary['metrics']}")
    return summary


if __name__ == "__main__":
    main()
