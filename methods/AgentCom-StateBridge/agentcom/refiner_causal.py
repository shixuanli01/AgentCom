"""Causal pilot for testing whether Refiner prefixes carry answer-relevant variation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from safetensors.torch import load_file
from transformers import GenerationConfig

from agentcom.multipath import (
    INVALID_LABEL,
    atomic_write_json,
    bootstrap_delta_interval,
    canonical_prediction,
    current_git_commit,
    exact_mcnemar_p,
    pairwise_disagreement,
    plurality_vote,
    sha256_json,
)
from methods.state_bridge import StateBridge, strip_thinking
from models import ModelWrapper
from prompts import EMBEDDING_HINT_MARKER, build_agent_message_embedding_mas
from utils import set_seed


PROTOCOL = "MP-SB-REFINER-CAUSAL-ORACLE-PILOT"
VALID_LABELS = frozenset("abcd")


def parse_boxed_choice(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{\s*([A-Da-d])\s*\}", text)
    return boxes[-1].lower() if boxes else None


def select_source_items(
    source_summary: Mapping[str, Any], subset: str
) -> List[Mapping[str, Any]]:
    items = list(source_summary["items"])
    if subset == "all":
        return items
    if subset == "oracle-misses":
        return [
            item
            for item in items
            if item["oracle_at_m"] and not item["vote_at_m_correct"]
        ]
    raise ValueError(f"Unsupported subset: {subset}")


def paired_metrics(
    primary: Sequence[bool], baseline: Sequence[bool], *, seed: int
) -> Dict[str, Any]:
    if len(primary) != len(baseline):
        raise ValueError("Paired metric inputs must have the same length")
    corrections = sum((not old) and new for new, old in zip(primary, baseline))
    harms = sum(old and (not new) for new, old in zip(primary, baseline))
    deltas = [int(new) - int(old) for new, old in zip(primary, baseline)]
    return {
        "corrections": corrections,
        "harms": harms,
        "delta_accuracy": sum(deltas) / len(deltas) if deltas else 0.0,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(corrections, harms),
        "paired_bootstrap_95ci": bootstrap_delta_interval(deltas, seed=seed),
    }


def aggregate_items(
    items: Sequence[Mapping[str, Any]], *, base_seed: int
) -> Dict[str, Any]:
    if not items:
        return {"total": 0}
    total = len(items)
    branch_count = len(items[0]["fixed_branch_correct"])
    branch_correct = [
        sum(bool(item["fixed_branch_correct"][branch]) for item in items)
        for branch in range(branch_count)
    ]
    fixed_oracle = [bool(item["fixed_oracle_at_m"]) for item in items]
    fixed_vote = [bool(item["fixed_vote_correct"]) for item in items]
    no_prefix = [bool(item["no_prefix_correct"]) for item in items]
    zero_prefix = [bool(item["zero_prefix_correct"]) for item in items]
    original_vote = [bool(item["original_vote_correct"]) for item in items]

    original_positive_paths = sum(
        sum(bool(value) for value in item["original_branch_correct"])
        for item in items
    )
    retained_positive_paths = sum(
        sum(
            bool(original) and bool(fixed)
            for original, fixed in zip(
                item["original_branch_correct"], item["fixed_branch_correct"]
            )
        )
        for item in items
    )
    original_negative_paths = total * branch_count - original_positive_paths
    repaired_negative_paths = sum(
        sum(
            (not bool(original)) and bool(fixed)
            for original, fixed in zip(
                item["original_branch_correct"], item["fixed_branch_correct"]
            )
        )
        for item in items
    )

    return {
        "total": total,
        "branches": branch_count,
        "fixed_per_branch": [
            {"branch": branch, "correct": count, "accuracy": count / total}
            for branch, count in enumerate(branch_correct)
        ],
        "fixed_mean_branch_accuracy": sum(branch_correct) / (total * branch_count),
        "fixed_vote_at_m": {
            "correct": sum(fixed_vote),
            "accuracy": sum(fixed_vote) / total,
        },
        "fixed_oracle_at_m": {
            "correct": sum(fixed_oracle),
            "accuracy": sum(fixed_oracle) / total,
        },
        "no_prefix": {
            "correct": sum(no_prefix),
            "accuracy": sum(no_prefix) / total,
        },
        "zero_prefix": {
            "correct": sum(zero_prefix),
            "accuracy": sum(zero_prefix) / total,
        },
        "items_with_prefix_dependent_branch_answers": sum(
            item["fixed_unique_prediction_count"] > 1 for item in items
        ),
        "items_any_branch_differs_from_no_prefix": sum(
            bool(item["any_branch_differs_from_no_prefix"]) for item in items
        ),
        "items_any_branch_differs_from_zero_prefix": sum(
            bool(item["any_branch_differs_from_zero_prefix"]) for item in items
        ),
        "mean_pairwise_branch_disagreement": sum(
            float(item["fixed_pairwise_disagreement"]) for item in items
        )
        / total,
        "original_correct_path_retention": {
            "retained": retained_positive_paths,
            "available": original_positive_paths,
            "rate": retained_positive_paths / original_positive_paths,
        },
        "original_wrong_path_repair": {
            "repaired": repaired_negative_paths,
            "available": original_negative_paths,
            "rate": repaired_negative_paths / original_negative_paths,
        },
        "fixed_vote_vs_no_prefix": paired_metrics(
            fixed_vote, no_prefix, seed=base_seed
        ),
        "fixed_vote_vs_zero_prefix": paired_metrics(
            fixed_vote, zero_prefix, seed=base_seed + 1
        ),
        "fixed_vote_vs_original_vote": paired_metrics(
            fixed_vote, original_vote, seed=base_seed + 2
        ),
        "parse_failures": sum(
            prediction == INVALID_LABEL
            for item in items
            for prediction in [
                *item["fixed_branch_predictions"],
                item["no_prefix_prediction"],
                item["zero_prefix_prediction"],
            ]
        ),
    }


class FixedJudger:
    def __init__(
        self,
        model: ModelWrapper,
        bridge: StateBridge,
        *,
        args: argparse.Namespace,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.bridge = bridge
        self.args = args
        self.max_new_tokens = max_new_tokens
        self.embedding_layer = model.model.get_input_embeddings()

    def align_saved_refiner(self, state_path: Path) -> torch.Tensor:
        tensors = load_file(state_path)
        hidden = tensors["refiner_transfer_hidden"].to(self.model.device)
        token_ids = tensors["refiner_transfer_token_ids"].to(self.model.device)
        aligned = self.bridge._align_hidden_sequence(hidden, token_ids)
        return self.bridge._process_prefix(aligned)

    def _prepare(
        self,
        question: str,
        prefix: Optional[torch.Tensor],
        *,
        enable_thinking: bool,
        messages: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_prefix = prefix is not None
        if messages is None:
            messages = build_agent_message_embedding_mas(
                role="judger",
                question=question,
                args=self.args,
                has_prefix=has_prefix,
            )
        prompt = self.model.render_chat(
            list(messages),
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if enable_thinking:
            prompt = f"{prompt}<think>"

        insert_position = None
        if has_prefix:
            marker_index = prompt.find(EMBEDDING_HINT_MARKER)
            if marker_index < 0:
                raise RuntimeError("Prefix prompt is missing the embedding marker")
            left = prompt[:marker_index]
            insert_position = len(
                self.model.tokenizer(left, add_special_tokens=False)["input_ids"]
            )
            prompt = prompt.replace(EMBEDDING_HINT_MARKER + "\n\n", "").replace(
                EMBEDDING_HINT_MARKER, ""
            )

        encoded = self.model.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        prompt_embeds = self.embedding_layer(input_ids)
        if prefix is None:
            return prompt_embeds, attention_mask

        assert insert_position is not None
        full_embeds = torch.cat(
            [
                prompt_embeds[:, :insert_position],
                prefix,
                prompt_embeds[:, insert_position:],
            ],
            dim=1,
        )
        prefix_mask = torch.ones(
            (attention_mask.shape[0], prefix.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat(
            [
                attention_mask[:, :insert_position],
                prefix_mask,
                attention_mask[:, insert_position:],
            ],
            dim=1,
        )
        return full_embeds, full_mask

    @torch.no_grad()
    def _generate_once(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            bos_token_id=self.model.tokenizer.bos_token_id,
        )
        started = time.time()
        output = self.model.model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            use_model_defaults=False,
        )
        torch.cuda.synchronize()
        duration = time.time() - started
        input_length = prompt_embeds.shape[1]
        sequence = output[0]
        generated = sequence[input_length:] if sequence.shape[0] >= input_length else sequence
        raw_response = self.model.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
        response = strip_thinking(raw_response)
        return {
            "raw_response": raw_response,
            "response": response,
            "generated_tokens": int(generated.shape[0]),
            "seconds": duration,
        }

    def run(
        self,
        question: str,
        prefix: Optional[torch.Tensor],
        *,
        seed: int,
    ) -> Dict[str, Any]:
        set_seed(seed)
        embeds, mask = self._prepare(question, prefix, enable_thinking=True)
        generation = self._generate_once(
            embeds, mask, max_new_tokens=self.max_new_tokens
        )
        prediction = parse_boxed_choice(generation["response"])
        generation["repair_used"] = False
        generation["repair_response"] = ""

        if prediction not in VALID_LABELS:
            has_prefix = prefix is not None
            messages = build_agent_message_embedding_mas(
                role="judger",
                question=question,
                args=self.args,
                has_prefix=has_prefix,
            )
            repair_messages = [
                *messages,
                {"role": "assistant", "content": generation["raw_response"]},
                {
                    "role": "user",
                    "content": (
                        "Using your analysis above, output only the final option in "
                        "exactly this form: \\boxed{A}. Replace A with one of A, B, C, D."
                    ),
                },
            ]
            set_seed(seed)
            repair_embeds, repair_mask = self._prepare(
                question,
                prefix,
                enable_thinking=False,
                messages=repair_messages,
            )
            repair = self._generate_once(
                repair_embeds, repair_mask, max_new_tokens=64
            )
            prediction = parse_boxed_choice(repair["response"])
            generation["repair_used"] = True
            generation["repair_response"] = repair["response"]
            generation["generated_tokens"] += repair["generated_tokens"]
            generation["seconds"] += repair["seconds"]

        generation["prediction"] = (
            prediction if prediction in VALID_LABELS else INVALID_LABEL
        )
        return generation


class CausalRunner:
    def __init__(
        self,
        judger: FixedJudger,
        *,
        source_run_dir: Path,
        run_dir: Path,
        manifest_fingerprint: str,
        base_seed: int,
    ) -> None:
        self.judger = judger
        self.source_run_dir = source_run_dir
        self.run_dir = run_dir
        self.manifest_fingerprint = manifest_fingerprint
        self.base_seed = base_seed
        self.stop_requested = False
        self.log_path = run_dir / "run.log"

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.log(f"Stop requested by signal {signum}; finishing current generation")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _record_path(self, item_index: int, variant: str) -> Path:
        return self.run_dir / "generations" / f"item_{item_index:04d}_{variant}.json"

    def _load_cached(self, item_index: int, variant: str) -> Optional[Dict[str, Any]]:
        path = self._record_path(item_index, variant)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            record.get("status") == "complete"
            and record.get("manifest_fingerprint") == self.manifest_fingerprint
        ):
            return record
        return None

    def _run_variant(
        self,
        *,
        item_index: int,
        question: str,
        gold: str,
        variant: str,
        prefix: Optional[torch.Tensor],
        original_prediction: Optional[str] = None,
    ) -> Dict[str, Any]:
        cached = self._load_cached(item_index, variant)
        if cached is not None:
            return cached
        seed = self.base_seed + item_index
        result = self.judger.run(question, prefix, seed=seed)
        prediction = canonical_prediction(result["prediction"])
        record = {
            "status": "complete",
            "manifest_fingerprint": self.manifest_fingerprint,
            "item_index": item_index,
            "variant": variant,
            "gold": gold,
            "prediction": prediction,
            "correct": prediction == gold,
            "original_prediction": (
                canonical_prediction(original_prediction)
                if original_prediction is not None
                else None
            ),
            **result,
        }
        atomic_write_json(self._record_path(item_index, variant), record)
        self.log(
            f"item={item_index} variant={variant} pred={prediction} gold={gold} "
            f"correct={record['correct']} duration={record['seconds']:.2f}s"
        )
        return record

    def run_item(self, source_item: Mapping[str, Any]) -> Dict[str, Any]:
        item_index = int(source_item["item_index"])
        gold = canonical_prediction(source_item["gold"])
        trace_path = (
            self.source_run_dir / "traces" / f"item_{item_index:04d}_branch_0.json"
        )
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        question = str(trace["question"])

        no_prefix = self._run_variant(
            item_index=item_index,
            question=question,
            gold=gold,
            variant="no_prefix",
            prefix=None,
        )
        zero_prefix_tensor = torch.zeros(
            (1, 64, self.judger.bridge.hidden_size),
            dtype=self.judger.bridge.dtype,
            device=self.judger.model.device,
        )
        zero_prefix = self._run_variant(
            item_index=item_index,
            question=question,
            gold=gold,
            variant="zero_prefix",
            prefix=zero_prefix_tensor,
        )

        branches = []
        for branch_id in range(5):
            state_path = (
                self.source_run_dir
                / "states"
                / f"item_{item_index:04d}_branch_{branch_id}.safetensors"
            )
            prefix = self.judger.align_saved_refiner(state_path)
            branches.append(
                self._run_variant(
                    item_index=item_index,
                    question=question,
                    gold=gold,
                    variant=f"branch_{branch_id}",
                    prefix=prefix,
                    original_prediction=source_item["branch_predictions"][branch_id],
                )
            )
            del prefix
            torch.cuda.empty_cache()

        predictions = [branch["prediction"] for branch in branches]
        correctness = [bool(branch["correct"]) for branch in branches]
        vote = plurality_vote(predictions)
        summary = {
            "item_index": item_index,
            "gold": gold,
            "original_branch_predictions": list(source_item["branch_predictions"]),
            "original_branch_correct": list(source_item["branch_correct"]),
            "original_vote_prediction": source_item["vote_at_m"]["winner"],
            "original_vote_correct": bool(source_item["vote_at_m_correct"]),
            "fixed_branch_predictions": predictions,
            "fixed_branch_correct": correctness,
            "fixed_vote_at_m": vote,
            "fixed_vote_correct": vote["winner"] == gold,
            "fixed_oracle_at_m": any(correctness),
            "fixed_unique_prediction_count": len(set(predictions)),
            "fixed_pairwise_disagreement": pairwise_disagreement(predictions),
            "no_prefix_prediction": no_prefix["prediction"],
            "no_prefix_correct": bool(no_prefix["correct"]),
            "zero_prefix_prediction": zero_prefix["prediction"],
            "zero_prefix_correct": bool(zero_prefix["correct"]),
            "any_branch_differs_from_no_prefix": any(
                prediction != no_prefix["prediction"] for prediction in predictions
            ),
            "any_branch_differs_from_zero_prefix": any(
                prediction != zero_prefix["prediction"] for prediction in predictions
            ),
        }
        atomic_write_json(
            self.run_dir / "items" / f"item_{item_index:04d}.json", summary
        )
        return summary

    def write_summary(
        self, items: Sequence[Mapping[str, Any]], *, status: str
    ) -> Dict[str, Any]:
        summary = {
            "status": status,
            "manifest_fingerprint": self.manifest_fingerprint,
            "updated_at": datetime.now().isoformat(),
            "metrics": aggregate_items(items, base_seed=self.base_seed),
            "items": list(items),
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        return summary

    def run(self, source_items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        summaries = []
        for item in source_items:
            summaries.append(self.run_item(item))
            self.write_summary(summaries, status="running")
            if self.stop_requested:
                return self.write_summary(summaries, status="paused")
        return self.write_summary(summaries, status="complete")


def build_manifest(
    *,
    repo_root: Path,
    source_run_dir: Path,
    source_summary: Mapping[str, Any],
    model: str,
    subset: str,
    base_seed: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    implementation_path = Path(__file__).resolve()
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now().isoformat(),
        "method_commit": current_git_commit(repo_root),
        "implementation_hash": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
        "source_run_dir": str(source_run_dir.resolve()),
        "source_manifest_fingerprint": source_summary["manifest_fingerprint"],
        "source_summary_hash": sha256_json(source_summary),
        "model": model,
        "subset": subset,
        "base_seed": base_seed,
        "max_new_tokens": max_new_tokens,
        "configuration": {
            "decoder": "greedy",
            "temperature": None,
            "top_p": None,
            "same_judger_seed_across_variants": True,
            "saved_state": "refiner_transfer_hidden_and_token_ids",
            "alignment": "source_adaptive_whitened_procrustes",
            "prefix_strategy": "scale",
            "prefix_scale": 1.0,
            "zero_prefix_length": 64,
            "format_repair": "greedy_no-thinking_64-token",
            "gold_available_to_generation": False,
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
            raise RuntimeError("Run directory contains a different causal manifest")
        return existing
    atomic_write_json(path, candidate)
    return dict(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROTOCOL)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subset", choices=("oracle-misses", "all"), default="oracle-misses")
    parser.add_argument("--base-seed", type=int, default=7300)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Refiner causal evaluation requires CUDA")
    torch.cuda.set_device(cli.gpu)
    source_summary = json.loads(
        (cli.source_run_dir / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary.get("status") != "complete":
        raise RuntimeError("Source multi-path run must be complete")
    selected = select_source_items(source_summary, cli.subset)
    if cli.limit is not None:
        selected = selected[: cli.limit]
    if not selected:
        raise ValueError("Selected source subset is empty")

    repo_root = Path(__file__).resolve().parents[1]
    candidate_manifest = build_manifest(
        repo_root=repo_root,
        source_run_dir=cli.source_run_dir,
        source_summary=source_summary,
        model=cli.model,
        subset=cli.subset,
        base_seed=cli.base_seed,
        max_new_tokens=cli.max_new_tokens,
    )
    manifest = ensure_manifest(cli.run_dir, candidate_manifest)
    args = argparse.Namespace(
        model=cli.model,
        model_name=cli.model,
        task="medqa",
        prompt="sequential",
        batch_size=1,
    )
    model = ModelWrapper(cli.model, device=torch.device(f"cuda:{cli.gpu}"), args=args)
    bridge = StateBridge(
        model,
        max_new_tokens=cli.max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        max_prefix_tokens=64,
        enable_thinking=True,
        prefix_strategy="scale",
        adaptive_reg=1e-3,
        snap_ratio=0.3,
        use_hook=True,
        args=args,
    )
    judger = FixedJudger(
        model, bridge, args=args, max_new_tokens=cli.max_new_tokens
    )
    runner = CausalRunner(
        judger,
        source_run_dir=cli.source_run_dir,
        run_dir=cli.run_dir,
        manifest_fingerprint=manifest["fingerprint"],
        base_seed=cli.base_seed,
    )
    runner.log(
        f"Refiner-Causal start subset={cli.subset} items={len(selected)} model={cli.model}"
    )
    summary = runner.run(selected)
    runner.log(f"Refiner-Causal status={summary['status']} metrics={summary['metrics']}")
    return summary


if __name__ == "__main__":
    main()
