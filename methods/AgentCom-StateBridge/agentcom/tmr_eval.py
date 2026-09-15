"""Controlled single-path evaluation for Trajectory Memory Relay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from agentcom.multipath import (
    INVALID_LABEL,
    atomic_write_json,
    branch_seeds,
    canonical_prediction,
    current_git_commit,
    exact_mcnemar_p,
    sha256_json,
)
from methods.state_bridge import (
    TASK_CONFIG,
    StateBridge,
    load_dataset_by_name,
    strip_thinking,
)
from methods.trajectory_memory_relay import (
    TMRConfig,
    TMRGenerationHooks,
    assert_no_new_trainable_parameters,
    decoder_layers,
    parameter_identity_snapshot,
    select_tmr_memory,
)
from models import ModelWrapper
from prompts import EMBEDDING_HINT_MARKER, build_agent_message_embedding_mas
from utils import (
    extract_gsm8k_answer,
    extract_gsm8k_answer_1,
    normalize_answer,
    set_seed,
)


PROTOCOL = "TMR-MEDQA300-V1"
ROLES = ("planner", "critic", "refiner", "judger")


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class TrajectoryMemoryRelay:
    """Planner-to-Judger pipeline with native external-memory attention."""

    def __init__(
        self,
        model: ModelWrapper,
        *,
        args: argparse.Namespace,
        config: TMRConfig,
        task: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model = model
        self.args = args
        self.config = config
        self.task = task
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        layers = decoder_layers(model.model)
        config.validate(num_layers=len(layers))
        self._parameter_snapshot = parameter_identity_snapshot(model.model)
        new_parameters = assert_no_new_trainable_parameters(
            self._parameter_snapshot, model.model
        )
        print(f"[TMR] num_new_trainable_parameters={new_parameters}", flush=True)

    def _render_prompt(self, role: str, question: str, *, has_memory: bool) -> tuple[torch.Tensor, torch.Tensor]:
        messages = build_agent_message_embedding_mas(
            role=role,
            question=question,
            context="",
            method="embedding_mas",
            args=self.args,
            has_prefix=has_memory,
        )
        prompts, _, _, _ = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        prompt = f"{prompts[0]}<think>"
        if has_memory:
            prompt = prompt.replace(EMBEDDING_HINT_MARKER + "\n\n", "").replace(
                EMBEDDING_HINT_MARKER, ""
            )
        encoded = self.model.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        return (
            encoded["input_ids"].to(self.model.device),
            encoded["attention_mask"].to(self.model.device),
        )

    def _generate(
        self,
        *,
        role: str,
        question: str,
        memory: Any,
        capture_trajectory: bool,
        seed: int,
    ) -> Dict[str, Any]:
        set_seed(seed)
        input_ids, attention_mask = self._render_prompt(
            role, question, has_memory=memory is not None
        )
        prompt_length = int(input_ids.shape[1])
        started = time.time()
        with TMRGenerationHooks(
            self.model.model,
            self.config,
            memory=memory,
            capture_trajectory=capture_trajectory,
        ) as hooks:
            with torch.no_grad():
                generated = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
            trajectories = hooks.trajectories()
            attention_summary = hooks.attention_summary()
        elapsed = time.time() - started

        sequences = generated.sequences
        if sequences.shape[1] < prompt_length:
            raise RuntimeError("Generated sequence is shorter than the receiver prompt")
        generated_ids = sequences[:, prompt_length:]
        generated_count = int(generated_ids.shape[1])
        if capture_trajectory:
            trajectory_lengths = {
                layer: int(states.shape[1]) for layer, states in trajectories.items()
            }
            if set(trajectory_lengths.values()) != {generated_count}:
                raise RuntimeError(
                    "Generation/capture event mismatch. TMR requires one normalized "
                    f"decision state per generated token: tokens={generated_count}, "
                    f"trajectories={trajectory_lengths}"
                )

        decoded = self.model.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
        output = strip_thinking(decoded)
        last_token = int(generated_ids[0, -1]) if generated_count else None
        return {
            "output": output,
            "generated_token_ids": generated_ids,
            "trajectories": trajectories,
            "generated_tokens": generated_count,
            "prompt_tokens": int(attention_mask[0].sum()),
            "hit_eos": last_token == self.model.tokenizer.eos_token_id,
            "inference_time": elapsed,
            "external_attention": attention_summary,
        }

    @torch.no_grad()
    def run_item(self, item: Mapping[str, Any], seeds: Mapping[str, int]) -> Dict[str, Any]:
        if set(seeds) != set(ROLES):
            raise ValueError(f"Expected role seeds for {ROLES}, got {sorted(seeds)}")
        question = str(item["question"])
        current_memory = None
        pending_handoff: Optional[Dict[str, Any]] = None
        trace: list[Dict[str, Any]] = []
        handoffs: list[Dict[str, Any]] = []
        final_text = ""

        for role in ROLES:
            generation = self._generate(
                role=role,
                question=question,
                memory=current_memory,
                capture_trajectory=role != "judger",
                seed=int(seeds[role]),
            )
            if pending_handoff is not None:
                pending_handoff["receiver_role"] = role
                pending_handoff["external_attention"] = generation[
                    "external_attention"
                ]
                handoffs.append(pending_handoff)
                pending_handoff = None

            trace.append(
                {
                    "role": role,
                    "output": generation["output"],
                    "prompt_tokens": generation["prompt_tokens"],
                    "generated_tokens": generation["generated_tokens"],
                    "hit_eos": generation["hit_eos"],
                    "inference_time": generation["inference_time"],
                    "received_tmr_states": (
                        len(current_memory.selected_indices)
                        if current_memory is not None
                        else 0
                    ),
                }
            )
            if role == "judger":
                final_text = generation["output"]
                continue

            current_memory = select_tmr_memory(
                generation["trajectories"],
                generation["generated_token_ids"],
                self.model.tokenizer,
                self.config,
            )
            pending_handoff = {
                "sample_id": int(item.get("idx", -1)),
                "sender_role": role,
                **current_memory.selection_diagnostic,
            }

        assert_no_new_trainable_parameters(self._parameter_snapshot, self.model.model)
        prediction = normalize_answer(extract_gsm8k_answer(final_text))
        prediction_1 = normalize_answer(extract_gsm8k_answer_1(final_text))
        gold = str(item.get("gold", ""))
        return {
            "question": question,
            "gold": gold,
            "solution": item.get("solution", ""),
            "prediction": prediction,
            "prediction_1": prediction_1,
            "correct": bool(prediction and gold and prediction == gold),
            "correct_1": bool(prediction_1 and gold and prediction_1 == gold),
            "final_response": final_text,
            "idx": int(item.get("idx", -1)),
            "trace": trace,
            "handoffs": handoffs,
            "num_new_trainable_parameters": 0,
            "efficiency": {
                "prompt_tokens": sum(stage["prompt_tokens"] for stage in trace),
                "generated_tokens": sum(stage["generated_tokens"] for stage in trace),
                "tmr_memory_states": sum(
                    handoff["selected_position_count"] for handoff in handoffs
                ),
                "inference_time": sum(stage["inference_time"] for stage in trace),
            },
        }


class RoleSeededStateBridge(StateBridge):
    """Unmodified StateBridge behavior with deterministic per-role reseeding."""

    def begin_item(self, seeds: Mapping[str, int]) -> None:
        self._tmr_eval_seeds = {role: int(seeds[role]) for role in ROLES}
        self._tmr_eval_generation_index = 0

    def _generate_with_prefix(self, *args: Any, **kwargs: Any):
        role = ROLES[self._tmr_eval_generation_index]
        set_seed(self._tmr_eval_seeds[role])
        result = super()._generate_with_prefix(*args, **kwargs)
        self._tmr_eval_generation_index += 1
        return result


def _baseline_items(summary: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for item in summary.get("items", []):
        predictions = item.get("branch_predictions", [])
        correctness = item.get("branch_correct", [])
        if not predictions or not correctness:
            raise ValueError("Baseline summary lacks branch-0 paired outcomes")
        result[int(item["item_index"])] = {
            "prediction": canonical_prediction(predictions[0]),
            "correct": bool(correctness[0]),
            "gold": canonical_prediction(item["gold"]),
        }
    return result


def shard_indexed_items(
    indexed_items: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Select a deterministic, balanced execution shard.

    Sharding changes only which process evaluates an item. Per-item role seeds
    remain derived from the original dataset index, so process count and resume
    order cannot change generation inputs.
    """
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return [
        item
        for position, item in enumerate(indexed_items)
        if position % num_shards == shard_index
    ]


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    total = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    paired_records = [
        record for record in records if int(record["item_index"]) in baseline
    ]
    corrections = sum(
        bool(record["correct"]) and not bool(baseline[int(record["item_index"])]["correct"])
        for record in paired_records
    )
    harms = sum(
        not bool(record["correct"]) and bool(baseline[int(record["item_index"])]["correct"])
        for record in paired_records
    )
    latency = [float(record["duration"]) for record in records]

    attention_rows = [
        handoff["external_attention"]["aggregate"]
        for record in records
        for handoff in record.get("handoffs", [])
    ]
    attention_positions = sum(int(row["positions"]) for row in attention_rows)

    def weighted_mean(key: str) -> float:
        if not attention_positions:
            return 0.0
        return sum(float(row[key]) * int(row["positions"]) for row in attention_rows) / attention_positions

    baseline_correct = sum(
        bool(baseline[int(record["item_index"])]["correct"])
        for record in paired_records
    )
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "paired_statebridge_items": len(paired_records),
        "baseline_correct": baseline_correct if paired_records else None,
        "baseline_accuracy": (
            baseline_correct / len(paired_records) if paired_records else None
        ),
        "delta_vs_statebridge": (
            (corrections - harms) / len(paired_records) if paired_records else None
        ),
        "rescues_vs_statebridge": corrections if paired_records else None,
        "destructions_vs_statebridge": harms if paired_records else None,
        "mcnemar_exact_two_sided_p": (
            exact_mcnemar_p(corrections, harms) if paired_records else None
        ),
        "invalid_outputs": sum(
            canonical_prediction(record.get("prediction")) == INVALID_LABEL
            for record in records
        ),
        "mean_latency_seconds": sum(latency) / len(latency) if latency else 0.0,
        "external_attention_positions": attention_positions,
        "mean_external_attention_entropy": weighted_mean(
            "mean_external_attention_entropy"
        ),
        "mean_tmr_gate": weighted_mean("mean_tmr_gate"),
        "max_tmr_gate": max(
            (float(row["max_tmr_gate"]) for row in attention_rows), default=0.0
        ),
        "mean_external_residual_norm": weighted_mean(
            "mean_external_residual_norm"
        ),
        "mean_self_attention_residual_norm": weighted_mean(
            "mean_self_attention_residual_norm"
        ),
        "norm_cap_applied_fraction": weighted_mean("norm_cap_applied_fraction"),
    }


class EvaluationRunner:
    def __init__(
        self,
        pipeline: Any,
        *,
        communication_method: str,
        task: str,
        base_seed: int,
        run_dir: Path,
        manifest_fingerprint: str,
        baseline: Mapping[int, Mapping[str, Any]],
        expected_item_ids: Sequence[int],
    ) -> None:
        self.pipeline = pipeline
        self.communication_method = communication_method
        self.task = task
        self.base_seed = base_seed
        self.run_dir = run_dir
        self.manifest_fingerprint = manifest_fingerprint
        self.baseline = baseline
        self.expected_item_ids = frozenset(int(item_id) for item_id in expected_item_ids)
        self.stop_requested = False

    def request_stop(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.stop_requested = True
        self.log("Stop requested; finishing the current item")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _record_path(self, item_index: int) -> Path:
        return self.run_dir / "records" / f"item_{item_index:04d}.json"

    def records(self) -> list[Dict[str, Any]]:
        rows = []
        for path in sorted((self.run_dir / "records").glob("item_*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                value.get("status") == "complete"
                and value.get("manifest_fingerprint") == self.manifest_fingerprint
            ):
                rows.append(value)
        return rows

    def write_summary(self, *, status: str) -> Dict[str, Any]:
        records = self.records()
        completed_item_ids = {int(record["item_index"]) for record in records}
        if status == "complete" and completed_item_ids != self.expected_item_ids:
            status = "running"
        metrics = aggregate_records(records, self.baseline)
        summary = {
            "status": status,
            "manifest_fingerprint": self.manifest_fingerprint,
            "updated_at": datetime.now().isoformat(),
            "expected_records": len(self.expected_item_ids),
            "completed_records": len(records),
            "metrics": metrics,
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        paired = [
            {
                "item_index": int(record["item_index"]),
                "gold": canonical_prediction(record["gold"]),
                "statebridge_prediction": self.baseline[int(record["item_index"])][
                    "prediction"
                ],
                "statebridge_correct": self.baseline[int(record["item_index"])][
                    "correct"
                ],
                "method_prediction": canonical_prediction(record.get("prediction")),
                "method_correct": bool(record["correct"]),
                "paired_delta": int(bool(record["correct"]))
                - int(bool(self.baseline[int(record["item_index"])]["correct"])),
            }
            for record in records
            if int(record["item_index"]) in self.baseline
        ]
        _atomic_write_jsonl(self.run_dir / "paired_vs_statebridge.jsonl", paired)
        diagnostics = [
            {"item_index": int(record["item_index"]), **handoff}
            for record in records
            for handoff in record.get("handoffs", [])
        ]
        _atomic_write_jsonl(self.run_dir / "diagnostics.jsonl", diagnostics)
        return summary

    def run(self, indexed_items: Sequence[tuple[int, Mapping[str, Any]]]) -> Dict[str, Any]:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        for item_index, source_item in indexed_items:
            path = self._record_path(item_index)
            if path.exists():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached = {}
                if (
                    cached.get("status") == "complete"
                    and cached.get("manifest_fingerprint") == self.manifest_fingerprint
                ):
                    self.log(f"resume item={item_index} prediction={cached.get('prediction')}")
                    continue

            item = dict(source_item)
            item["idx"] = item_index
            seeds = branch_seeds(self.base_seed, self.task, item_index, 0)
            started = time.time()
            try:
                if self.communication_method == "statebridge":
                    self.pipeline.begin_item(seeds)
                    result = self.pipeline.run_item(item)
                else:
                    result = self.pipeline.run_item(item, seeds)
            except BaseException as error:
                atomic_write_json(
                    path,
                    {
                        "status": "error",
                        "manifest_fingerprint": self.manifest_fingerprint,
                        "item_index": item_index,
                        "stage_seeds": seeds,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                raise
            record = {
                "status": "complete",
                "manifest_fingerprint": self.manifest_fingerprint,
                "item_index": item_index,
                "stage_seeds": seeds,
                "duration": time.time() - started,
                **result,
            }
            atomic_write_json(path, record)
            self.log(
                f"item={item_index} pred={canonical_prediction(record.get('prediction'))} "
                f"gold={canonical_prediction(record.get('gold'))} "
                f"correct={record['correct']} seconds={record['duration']:.2f}"
            )
            self.write_summary(status="running")
            if self.stop_requested:
                return self.write_summary(status="paused")
            torch.cuda.empty_cache()
        return self.write_summary(status="complete")


def build_manifest(
    *,
    repo_root: Path,
    cli: argparse.Namespace,
    config: TMRConfig,
    selected_ids: Sequence[int],
    baseline_summary: Mapping[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    implementation_paths = [
        Path(__file__).resolve(),
        repo_root / "methods" / "trajectory_memory_relay.py",
    ]
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now().isoformat(),
        "method_commit": current_git_commit(repo_root),
        "implementation_hashes": {
            str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in implementation_paths
        },
        "communication_method": cli.communication_method,
        "model": cli.model,
        "task": cli.task,
        "selected_item_ids": list(selected_ids),
        "base_seed": cli.base_seed,
        "max_new_tokens": max_new_tokens,
        "temperature": cli.temperature,
        "top_p": cli.top_p,
        "tmr": {
            "selection": config.tmr_selection,
            "layers": list(config.tmr_layers),
            "norm_ratio": config.tmr_norm_ratio,
            "entropy_gate": config.tmr_entropy_gate,
            "force_gate_zero": config.tmr_force_gate_zero,
            "k": config.tmr_k,
            "coverage_k_tail": config.coverage_k_tail,
            "coverage_k_representative": config.coverage_k_representative,
            "coverage_selection_layer": config.coverage_selection_layer,
            "coverage_scope": config.coverage_scope,
            "rope_on_external_memory": False,
            "inputs_embeds_prefix": False,
            "new_trainable_parameters": 0,
        },
        "baseline_summary_hash": (
            sha256_json(baseline_summary) if baseline_summary else None
        ),
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(cli.gpu) if torch.cuda.is_available() else None,
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
            raise RuntimeError("Run directory contains a different TMR manifest")
        return existing
    atomic_write_json(path, candidate)
    return dict(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROTOCOL)
    parser.add_argument("--communication-method", choices=("statebridge", "tmr"), default="tmr")
    parser.add_argument("--tmr-selection", choices=("last64", "coverage64"), default="last64")
    parser.add_argument("--tmr-layers", nargs="+", type=int, default=[11, 23, 35])
    parser.add_argument("--tmr-norm-ratio", type=float, default=0.25)
    parser.add_argument("--tmr-entropy-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tmr-force-gate-zero", action="store_true")
    parser.add_argument("--tmr-k", type=int, default=64)
    parser.add_argument("--coverage-k-tail", type=int, default=16)
    parser.add_argument("--coverage-k-representative", type=int, default=48)
    parser.add_argument("--coverage-selection-layer", type=int, default=35)
    parser.add_argument("--coverage-scope", choices=("full", "post_think"), default="full")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--task", default="medqa", choices=sorted(TASK_CONFIG))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--item-ids", nargs="+", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        help="Optional MP-StateBridge summary used for paired metrics",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TMR evaluation requires CUDA")
    torch.cuda.set_device(cli.gpu)
    config = TMRConfig(
        communication_method=cli.communication_method,
        tmr_selection=cli.tmr_selection,
        tmr_layers=tuple(cli.tmr_layers),
        tmr_norm_ratio=cli.tmr_norm_ratio,
        tmr_entropy_gate=cli.tmr_entropy_gate,
        tmr_force_gate_zero=cli.tmr_force_gate_zero,
        tmr_k=cli.tmr_k,
        coverage_k_tail=cli.coverage_k_tail,
        coverage_k_representative=cli.coverage_k_representative,
        coverage_selection_layer=cli.coverage_selection_layer,
        coverage_scope=cli.coverage_scope,
    )
    config.validate()

    data = load_dataset_by_name(cli.task)
    if cli.item_ids is not None:
        if cli.start_index != 0 or cli.limit is not None:
            raise ValueError("--item-ids cannot be combined with --start-index/--limit")
        selected = [(item_id, data[item_id]) for item_id in cli.item_ids]
    else:
        stop = len(data) if cli.limit is None else min(len(data), cli.start_index + cli.limit)
        selected = list(enumerate(data))[cli.start_index:stop]
    if not selected:
        raise ValueError("The selected dataset slice is empty")
    if any(not 0 <= item_id < len(data) for item_id, _ in selected):
        raise ValueError("Selected item ID is outside the dataset")
    execution_items = shard_indexed_items(
        selected,
        num_shards=cli.num_shards,
        shard_index=cli.shard_index,
    )
    if not execution_items:
        raise ValueError("The selected execution shard is empty")

    baseline_summary: Dict[str, Any] = {}
    baseline: Dict[int, Dict[str, Any]] = {}
    if cli.baseline_summary is not None:
        baseline_summary = json.loads(
            cli.baseline_summary.read_text(encoding="utf-8")
        )
        baseline = _baseline_items(baseline_summary)
        missing_baseline = [item_id for item_id, _ in selected if item_id not in baseline]
        if missing_baseline:
            raise ValueError(f"Baseline summary is missing items: {missing_baseline[:10]}")
    else:
        print(
            "[TMR] no --baseline-summary supplied; paired metrics will be null",
            flush=True,
        )

    max_new_tokens = cli.max_new_tokens or TASK_CONFIG[cli.task]["max_new_tokens"]
    repo_root = Path(__file__).resolve().parents[1]
    manifest = ensure_manifest(
        cli.run_dir,
        build_manifest(
            repo_root=repo_root,
            cli=cli,
            config=config,
            selected_ids=[item_id for item_id, _ in selected],
            baseline_summary=baseline_summary,
            max_new_tokens=max_new_tokens,
        ),
    )
    args = argparse.Namespace(
        model=cli.model,
        model_name=cli.model,
        task=cli.task,
        prompt="sequential",
        batch_size=1,
    )
    model = ModelWrapper(cli.model, device=torch.device(f"cuda:{cli.gpu}"), args=args)
    if cli.communication_method == "statebridge":
        pipeline = RoleSeededStateBridge(
            model,
            max_new_tokens=max_new_tokens,
            temperature=cli.temperature,
            top_p=cli.top_p,
            max_prefix_tokens=64,
            enable_thinking=True,
            prefix_strategy="scale",
            adaptive_reg=1e-3,
            snap_ratio=0.3,
            use_hook=True,
            args=args,
        )
    else:
        pipeline = TrajectoryMemoryRelay(
            model,
            args=args,
            config=config,
            task=cli.task,
            max_new_tokens=max_new_tokens,
            temperature=cli.temperature,
            top_p=cli.top_p,
        )
    runner = EvaluationRunner(
        pipeline,
        communication_method=cli.communication_method,
        task=cli.task,
        base_seed=cli.base_seed,
        run_dir=cli.run_dir,
        manifest_fingerprint=manifest["fingerprint"],
        baseline=baseline,
        expected_item_ids=[item_id for item_id, _ in selected],
    )
    runner.log(
        f"start method={cli.communication_method} selection={cli.tmr_selection} "
        f"items={len(execution_items)}/{len(selected)} model={cli.model} "
        f"shard={cli.shard_index}/{cli.num_shards}"
    )
    summary = runner.run(execution_items)
    runner.log(f"finished status={summary['status']} metrics={summary['metrics']}")
    return summary


if __name__ == "__main__":
    main()
