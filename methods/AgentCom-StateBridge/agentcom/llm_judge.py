"""Candidate-balanced LLM-as-Judge evaluation over frozen multi-path traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from transformers import GenerationConfig

from agentcom.multipath import (
    INVALID_LABEL,
    atomic_write_json,
    bootstrap_delta_interval,
    canonical_prediction,
    current_git_commit,
    exact_mcnemar_p,
    sha256_json,
    stable_seed,
)
from models import ModelWrapper
from utils import set_seed


VALID_MEDQA_LABELS = frozenset("abcd")
PROTOCOL = "MP-SB-V1-CANDIDATE-BALANCED-LLM-JUDGE"
JUDGE_PROMPT_VERSION = "v1.1"


def deterministic_candidate_order(
    labels: Sequence[str], *, base_seed: int, item_index: int
) -> List[str]:
    """Return a reproducible order that does not depend on vote count."""
    unique = sorted(set(labels))
    return sorted(
        unique,
        key=lambda label: hashlib.sha256(
            f"{base_seed}\0{item_index}\0candidate-order\0{label}".encode("utf-8")
        ).digest(),
    )


def choose_representative_branch(
    branch_ids: Sequence[int],
    *,
    base_seed: int,
    item_index: int,
    label: str,
) -> int:
    """Choose one rationale per answer without inspecting rationale contents."""
    if not branch_ids:
        raise ValueError("A candidate must have at least one supporting branch")
    return min(
        branch_ids,
        key=lambda branch_id: hashlib.sha256(
            (
                f"{base_seed}\0{item_index}\0representative\0"
                f"{label}\0{branch_id}"
            ).encode("utf-8")
        ).digest(),
    )


def build_candidate_packet(
    item_summary: Mapping[str, Any],
    branch_records: Sequence[Mapping[str, Any]],
    *,
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Build one count-balanced rationale packet for every valid answer."""
    item_index = int(item_summary["item_index"])
    by_label: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in branch_records:
        label = canonical_prediction(record.get("prediction"))
        if label in VALID_MEDQA_LABELS:
            by_label[label].append(record)

    ordered_labels = deterministic_candidate_order(
        list(by_label), base_seed=base_seed, item_index=item_index
    )
    packet: List[Dict[str, Any]] = []
    for label in ordered_labels:
        records = by_label[label]
        branch_id = choose_representative_branch(
            [int(record["branch_id"]) for record in records],
            base_seed=base_seed,
            item_index=item_index,
            label=label,
        )
        representative = next(
            record for record in records if int(record["branch_id"]) == branch_id
        )
        packet.append(
            {
                "label": label,
                "representative_branch_id": branch_id,
                "rationale": str(representative.get("final_response", "")).strip(),
            }
        )
    return packet


def truncate_balanced_rationale(
    text: str, tokenizer: Any, *, max_tokens: int
) -> tuple[str, int, bool]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, len(token_ids), False
    head_size = max_tokens // 2
    tail_size = max_tokens - head_size
    head = tokenizer.decode(token_ids[:head_size], skip_special_tokens=True)
    tail = tokenizer.decode(token_ids[-tail_size:], skip_special_tokens=True)
    return f"{head}\n[...middle omitted...]\n{tail}", max_tokens, True


def build_judge_messages(
    question: str,
    packet: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    rationale_tokens: int,
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Create an anonymous prompt with equal evidence capacity per candidate."""
    rendered_candidates: List[str] = []
    prompt_packet: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(packet, start=1):
        rationale, used_tokens, truncated = truncate_balanced_rationale(
            str(candidate["rationale"]), tokenizer, max_tokens=rationale_tokens
        )
        label = str(candidate["label"]).upper()
        rendered_candidates.append(
            f"Candidate {candidate_index}\n"
            f"Proposed option: {label}\n"
            f"Reasoning:\n{rationale}"
        )
        prompt_packet.append(
            {
                "candidate_index": candidate_index,
                "label": str(candidate["label"]),
                "representative_branch_id": int(candidate["representative_branch_id"]),
                "rationale_tokens": used_tokens,
                "truncated": truncated,
            }
        )

    allowed = ", ".join(str(candidate["label"]).upper() for candidate in packet)
    user_prompt = f"""You are the final adjudicator for a medical multiple-choice problem.

Question:
{question}

Below are anonymized candidate solutions. Each distinct proposed answer is represented exactly once. Their original frequency and source identity are hidden. A majority can be confidently wrong, so judge the medical evidence rather than apparent consensus.

{chr(10).join(chr(10) + candidate for candidate in rendered_candidates)}

Instructions:
1. Independently verify each proposed option against the question.
2. Identify factual errors, unsupported assumptions, and the most discriminative clinical evidence.
3. Select exactly one option from this allowed set: {allowed}.
4. End with exactly one final selection in the form \\boxed{{A}}.
Do not select an option outside the allowed set.
"""
    messages = [
        {
            "role": "system",
            "content": "You are a careful, evidence-focused medical answer adjudicator.",
        },
        {"role": "user", "content": user_prompt},
    ]
    return messages, prompt_packet


def parse_judge_choice(text: str, allowed_labels: Sequence[str]) -> Optional[str]:
    allowed = {canonical_prediction(label) for label in allowed_labels}
    boxes = re.findall(r"\\boxed\{\s*([A-Da-d])\s*\}", text)
    if not boxes:
        return None
    choice = boxes[-1].lower()
    return choice if choice in allowed else None


def aggregate_judge_items(
    items: Sequence[Mapping[str, Any]], *, base_seed: int
) -> Dict[str, Any]:
    if not items:
        return {"total": 0}
    total = len(items)
    judge_correct = sum(bool(item["judge_correct"]) for item in items)
    vote_correct = sum(bool(item["vote_correct"]) for item in items)
    branch_zero_correct = sum(bool(item["branch_0_correct"]) for item in items)

    judge_vs_vote_corrections = sum(
        (not bool(item["vote_correct"])) and bool(item["judge_correct"])
        for item in items
    )
    judge_vs_vote_harms = sum(
        bool(item["vote_correct"]) and (not bool(item["judge_correct"]))
        for item in items
    )
    judge_vs_branch_zero_corrections = sum(
        (not bool(item["branch_0_correct"])) and bool(item["judge_correct"])
        for item in items
    )
    judge_vs_branch_zero_harms = sum(
        bool(item["branch_0_correct"]) and (not bool(item["judge_correct"]))
        for item in items
    )
    judge_vote_deltas = [
        int(bool(item["judge_correct"])) - int(bool(item["vote_correct"]))
        for item in items
    ]
    judge_branch_zero_deltas = [
        int(bool(item["judge_correct"])) - int(bool(item["branch_0_correct"]))
        for item in items
    ]
    called = [item for item in items if item["decision_mode"] == "llm_judge"]
    oracle_misses = [
        item for item in items if item["oracle_at_m"] and not item["vote_correct"]
    ]

    return {
        "total": total,
        "llm_judge": {"correct": judge_correct, "accuracy": judge_correct / total},
        "vote_at_5": {"correct": vote_correct, "accuracy": vote_correct / total},
        "branch_0": {
            "correct": branch_zero_correct,
            "accuracy": branch_zero_correct / total,
        },
        "delta_judge_minus_vote": (judge_correct - vote_correct) / total,
        "delta_judge_minus_branch_0": (judge_correct - branch_zero_correct) / total,
        "judge_vs_vote": {
            "corrections": judge_vs_vote_corrections,
            "harms": judge_vs_vote_harms,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                judge_vs_vote_corrections, judge_vs_vote_harms
            ),
            "paired_bootstrap_95ci": bootstrap_delta_interval(
                judge_vote_deltas, seed=base_seed
            ),
        },
        "judge_vs_branch_0": {
            "corrections": judge_vs_branch_zero_corrections,
            "harms": judge_vs_branch_zero_harms,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                judge_vs_branch_zero_corrections, judge_vs_branch_zero_harms
            ),
            "paired_bootstrap_95ci": bootstrap_delta_interval(
                judge_branch_zero_deltas, seed=base_seed + 1
            ),
        },
        "judge_calls": len(called),
        "unanimous_passthrough": total - len(called),
        "judge_parse_failures": sum(
            item["judge_prediction"] == INVALID_LABEL for item in called
        ),
        "oracle_miss_recovery": {
            "available": len(oracle_misses),
            "recovered": sum(bool(item["judge_correct"]) for item in oracle_misses),
            "rate": (
                sum(bool(item["judge_correct"]) for item in oracle_misses)
                / len(oracle_misses)
                if oracle_misses
                else 0.0
            ),
        },
        "mean_judge_seconds": (
            sum(float(item["judge_seconds"]) for item in called) / len(called)
            if called
            else 0.0
        ),
        "mean_judge_generated_tokens": (
            sum(int(item["judge_generated_tokens"]) for item in called) / len(called)
            if called
            else 0.0
        ),
    }


class JudgeRunner:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        source_run_dir: Path,
        run_dir: Path,
        manifest_fingerprint: str,
        base_seed: int,
        max_new_tokens: int,
        rationale_tokens: int,
    ) -> None:
        self.model = model
        self.source_run_dir = source_run_dir
        self.run_dir = run_dir
        self.manifest_fingerprint = manifest_fingerprint
        self.base_seed = base_seed
        self.max_new_tokens = max_new_tokens
        self.rationale_tokens = rationale_tokens
        self.stop_requested = False
        self.log_path = run_dir / "run.log"

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.log(f"Stop requested by signal {signum}; finishing current judge call")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _decision_path(self, item_index: int) -> Path:
        return self.run_dir / "decisions" / f"item_{item_index:04d}.json"

    def _load_cached(self, item_index: int) -> Optional[Dict[str, Any]]:
        path = self._decision_path(item_index)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            record.get("status") == "complete"
            and record.get("item_index") == item_index
            and record.get("manifest_fingerprint") == self.manifest_fingerprint
        ):
            return record
        return None

    def _load_branches(self, item_index: int) -> List[Dict[str, Any]]:
        records = []
        for branch_id in range(5):
            path = (
                self.source_run_dir
                / "traces"
                / f"item_{item_index:04d}_branch_{branch_id}.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                raise RuntimeError(f"Incomplete source trace: {path}")
            records.append(record)
        return records

    @torch.no_grad()
    def _generate(
        self,
        messages: Sequence[Mapping[str, str]],
        seed: int,
        *,
        enable_thinking: bool,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        set_seed(seed)
        prompt = self.model.render_chat(
            list(messages),
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if enable_thinking:
            prompt = f"{prompt}<think>"
        encoded = self.model.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        started = time.time()
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
        )
        output = self.model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            use_model_defaults=False,
        )
        torch.cuda.synchronize()
        duration = time.time() - started
        generated = output[0, input_ids.shape[1] :]
        text = self.model.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "response": text,
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(generated.shape[0]),
            "seconds": duration,
        }

    def run_item(self, source_item: Mapping[str, Any]) -> Dict[str, Any]:
        item_index = int(source_item["item_index"])
        cached = self._load_cached(item_index)
        if cached is not None:
            return cached

        branch_records = self._load_branches(item_index)
        packet = build_candidate_packet(
            source_item, branch_records, base_seed=self.base_seed
        )
        valid_labels = [str(candidate["label"]) for candidate in packet]
        gold = canonical_prediction(source_item["gold"])
        vote_prediction = canonical_prediction(source_item["vote_at_m"]["winner"])
        repair_generation = None

        if len(valid_labels) == 1:
            prediction = valid_labels[0]
            mode = "unanimous_passthrough"
            generation = {
                "response": "",
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "seconds": 0.0,
            }
            prompt_packet = [
                {
                    "candidate_index": 1,
                    "label": packet[0]["label"],
                    "representative_branch_id": packet[0]["representative_branch_id"],
                    "rationale_tokens": 0,
                    "truncated": False,
                }
            ]
        else:
            mode = "llm_judge"
            messages, prompt_packet = build_judge_messages(
                str(branch_records[0]["question"]),
                packet,
                tokenizer=self.model.tokenizer,
                rationale_tokens=self.rationale_tokens,
            )
            seed = stable_seed(
                self.base_seed, "medqa", item_index, 0, "llm-judge"
            )
            generation = self._generate(
                messages,
                seed,
                enable_thinking=True,
                max_new_tokens=self.max_new_tokens,
            )
            parsed = parse_judge_choice(generation["response"], valid_labels)
            if parsed is None:
                allowed = ", ".join(label.upper() for label in valid_labels)
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": generation["response"]},
                    {
                        "role": "user",
                        "content": (
                            "Your adjudication was not in the required parseable format. "
                            "Using your analysis above, output only one final choice from "
                            f"{allowed} in exactly this form: \\boxed{{A}}"
                        ),
                    },
                ]
                repair_seed = stable_seed(
                    self.base_seed, "medqa", item_index, 0, "llm-judge-format-repair"
                )
                repair_generation = self._generate(
                    repair_messages,
                    repair_seed,
                    enable_thinking=False,
                    max_new_tokens=64,
                )
                parsed = parse_judge_choice(
                    repair_generation["response"], valid_labels
                )
                generation["seconds"] += repair_generation["seconds"]
                generation["prompt_tokens"] += repair_generation["prompt_tokens"]
                generation["generated_tokens"] += repair_generation[
                    "generated_tokens"
                ]
            prediction = parsed if parsed is not None else INVALID_LABEL

        record = {
            "status": "complete",
            "manifest_fingerprint": self.manifest_fingerprint,
            "item_index": item_index,
            "gold": gold,
            "candidate_order": valid_labels,
            "candidate_packet": prompt_packet,
            "decision_mode": mode,
            "judge_prediction": prediction,
            "judge_correct": prediction == gold,
            "vote_prediction": vote_prediction,
            "vote_correct": bool(source_item["vote_at_m_correct"]),
            "branch_0_prediction": canonical_prediction(
                source_item["branch_predictions"][0]
            ),
            "branch_0_correct": bool(source_item["branch_0_correct"]),
            "oracle_at_m": bool(source_item["oracle_at_m"]),
            "judge_response": generation["response"],
            "judge_repair_used": repair_generation is not None,
            "judge_repair_response": (
                repair_generation["response"] if repair_generation is not None else ""
            ),
            "judge_prompt_tokens": generation["prompt_tokens"],
            "judge_generated_tokens": generation["generated_tokens"],
            "judge_seconds": generation["seconds"],
        }
        atomic_write_json(self._decision_path(item_index), record)
        return record

    def write_summary(
        self, items: Sequence[Mapping[str, Any]], *, status: str
    ) -> Dict[str, Any]:
        summary = {
            "status": status,
            "manifest_fingerprint": self.manifest_fingerprint,
            "updated_at": datetime.now().isoformat(),
            "metrics": aggregate_judge_items(items, base_seed=self.base_seed),
            "items": list(items),
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        return summary

    def run(self, source_items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        results: List[Dict[str, Any]] = []
        for source_item in source_items:
            item_index = int(source_item["item_index"])
            result = self.run_item(source_item)
            results.append(result)
            self.log(
                f"item={item_index} mode={result['decision_mode']} "
                f"judge={result['judge_prediction']} vote={result['vote_prediction']} "
                f"gold={result['gold']} correct={result['judge_correct']} "
                f"duration={result['judge_seconds']:.2f}s"
            )
            self.write_summary(results, status="running")
            if self.stop_requested:
                return self.write_summary(results, status="paused")
        return self.write_summary(results, status="complete")


def build_manifest(
    *,
    repo_root: Path,
    source_run_dir: Path,
    source_summary: Mapping[str, Any],
    model: str,
    base_seed: int,
    max_new_tokens: int,
    rationale_tokens: int,
    start_index: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    implementation_path = Path(__file__).resolve()
    protocol_description = {
        "candidate_frequency_hidden": True,
        "branch_identity_hidden": True,
        "one_representative_rationale_per_unique_valid_answer": True,
        "representative_selected_without_content_inspection": True,
        "candidate_order_deterministically_shuffled": True,
        "judge_constrained_to_observed_valid_answers": True,
        "unanimous_valid_answers_pass_through_without_call": True,
        "parse_failure": "INVALID_and_incorrect",
        "format_repair": (
            "on_parse_failure_reuse_prior_analysis_then_greedy_no-thinking_64-token_choice"
        ),
        "gold_available_to_judge": False,
        "decoding": "greedy",
    }
    manifest = {
        "protocol": PROTOCOL,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "created_at": datetime.now().isoformat(),
        "method_commit": current_git_commit(repo_root),
        "implementation_hash": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
        "source_run_dir": str(source_run_dir.resolve()),
        "source_manifest_fingerprint": source_summary["manifest_fingerprint"],
        "source_summary_hash": sha256_json(source_summary),
        "model": model,
        "base_seed": base_seed,
        "start_index": start_index,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "rationale_tokens_per_candidate": rationale_tokens,
        "protocol_description": protocol_description,
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
            raise RuntimeError("Run directory contains a different judge manifest")
        return existing
    atomic_write_json(path, candidate)
    return dict(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PROTOCOL)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=4242)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--rationale-tokens", type=int, default=1024)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> Dict[str, Any]:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LLM Judge requires a CUDA device")
    torch.cuda.set_device(cli.gpu)
    source_summary_path = cli.source_run_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete":
        raise RuntimeError("Source multi-path run must be complete")
    all_items = list(source_summary["items"])
    stop = (
        len(all_items)
        if cli.limit is None
        else min(len(all_items), cli.start_index + cli.limit)
    )
    selected = all_items[cli.start_index:stop]
    if not selected:
        raise ValueError("Selected source slice is empty")

    repo_root = Path(__file__).resolve().parents[1]
    candidate_manifest = build_manifest(
        repo_root=repo_root,
        source_run_dir=cli.source_run_dir,
        source_summary=source_summary,
        model=cli.model,
        base_seed=cli.base_seed,
        max_new_tokens=cli.max_new_tokens,
        rationale_tokens=cli.rationale_tokens,
        start_index=cli.start_index,
        limit=cli.limit,
    )
    manifest = ensure_manifest(cli.run_dir, candidate_manifest)
    model = ModelWrapper(cli.model, device=torch.device(f"cuda:{cli.gpu}"))
    runner = JudgeRunner(
        model,
        source_run_dir=cli.source_run_dir,
        run_dir=cli.run_dir,
        manifest_fingerprint=manifest["fingerprint"],
        base_seed=cli.base_seed,
        max_new_tokens=cli.max_new_tokens,
        rationale_tokens=cli.rationale_tokens,
    )
    runner.log(
        f"LLM-Judge start source_items={cli.start_index}:{stop} model={cli.model}"
    )
    summary = runner.run(selected)
    runner.log(f"LLM-Judge status={summary['status']} metrics={summary['metrics']}")
    return summary


if __name__ == "__main__":
    main()
