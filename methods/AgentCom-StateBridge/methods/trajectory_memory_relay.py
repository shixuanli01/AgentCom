"""Training-free native hidden-space communication for Qwen-style decoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


TMR_SELECTIONS = ("last64", "coverage64")
COVERAGE_SCOPES = ("full", "post_think")


@dataclass(frozen=True)
class TMRConfig:
    communication_method: str = "tmr"
    tmr_selection: str = "last64"
    tmr_layers: tuple[int, ...] = (11, 23, 35)
    tmr_norm_ratio: float = 0.25
    tmr_entropy_gate: bool = True
    tmr_force_gate_zero: bool = False
    tmr_k: int = 64
    coverage_k_tail: int = 16
    coverage_k_representative: int = 48
    coverage_selection_layer: int = 35
    coverage_scope: str = "full"
    eps: float = 1e-6

    def validate(self, *, num_layers: Optional[int] = None) -> None:
        if self.communication_method not in ("statebridge", "tmr"):
            raise ValueError("communication_method must be 'statebridge' or 'tmr'")
        if self.tmr_selection not in TMR_SELECTIONS:
            raise ValueError(f"tmr_selection must be one of {TMR_SELECTIONS}")
        if not self.tmr_layers or len(set(self.tmr_layers)) != len(self.tmr_layers):
            raise ValueError("tmr_layers must contain unique layer indices")
        if min(self.tmr_layers) < 0:
            raise ValueError("tmr_layers must be non-negative")
        if num_layers is not None and max(self.tmr_layers) >= num_layers:
            raise ValueError(
                f"TMR layer {max(self.tmr_layers)} is invalid for {num_layers} layers"
            )
        if self.coverage_selection_layer not in self.tmr_layers:
            raise ValueError("coverage_selection_layer must be one of tmr_layers")
        if self.coverage_scope not in COVERAGE_SCOPES:
            raise ValueError(f"coverage_scope must be one of {COVERAGE_SCOPES}")
        if self.tmr_k <= 0:
            raise ValueError("tmr_k must be positive")
        if self.coverage_k_tail < 0 or self.coverage_k_representative < 0:
            raise ValueError("coverage budgets must be non-negative")
        if self.coverage_k_tail + self.coverage_k_representative != self.tmr_k:
            raise ValueError("coverage budgets must sum to tmr_k")
        if self.tmr_norm_ratio < 0:
            raise ValueError("tmr_norm_ratio must be non-negative")


@dataclass
class TMRMemory:
    """A position-free set of native, layer-matched sender states."""

    states: Dict[int, torch.Tensor]
    selected_indices: list[int]
    selection_diagnostic: Dict[str, Any]

    def validate(self, layers: Sequence[int]) -> None:
        expected = tuple(layers)
        if tuple(sorted(self.states)) != tuple(sorted(expected)):
            raise ValueError(
                f"Memory layers {sorted(self.states)} do not match {sorted(expected)}"
            )
        lengths = {layer: int(self.states[layer].shape[1]) for layer in expected}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"TMR memory layer lengths disagree: {lengths}")
        if next(iter(lengths.values()), 0) != len(self.selected_indices):
            raise ValueError(
                "TMR memory length does not match selected position count: "
                f"{lengths} vs {len(self.selected_indices)}"
            )
        for layer in expected:
            if self.states[layer].ndim != 3:
                raise ValueError("TMR memory tensors must have shape [batch, K, hidden]")


@dataclass
class ProjectedMemory:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class AttentionStatistics:
    count: int = 0
    entropy_sum: float = 0.0
    gate_sum: float = 0.0
    max_gate: float = 0.0
    external_norm_sum: float = 0.0
    raw_external_norm_sum: float = 0.0
    self_norm_sum: float = 0.0
    cap_applied_count: int = 0

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        gate = values["gate"].detach().float()
        count = gate.numel()
        self.count += count
        self.entropy_sum += float(values["entropy"].detach().float().sum().cpu())
        self.gate_sum += float(gate.sum().cpu())
        self.max_gate = max(self.max_gate, float(gate.max().cpu()))
        self.external_norm_sum += float(
            values["external_norm"].detach().float().sum().cpu()
        )
        self.raw_external_norm_sum += float(
            values["raw_external_norm"].detach().float().sum().cpu()
        )
        self.self_norm_sum += float(values["self_norm"].detach().float().sum().cpu())
        self.cap_applied_count += int(values["cap_applied"].detach().sum().cpu())

    def summary(self) -> Dict[str, float | int]:
        denominator = max(self.count, 1)
        return {
            "positions": self.count,
            "mean_external_attention_entropy": self.entropy_sum / denominator,
            "mean_tmr_gate": self.gate_sum / denominator,
            "max_tmr_gate": self.max_gate,
            "mean_external_residual_norm": self.external_norm_sum / denominator,
            "mean_raw_external_residual_norm": (
                self.raw_external_norm_sum / denominator
            ),
            "mean_self_attention_residual_norm": self.self_norm_sum / denominator,
            "norm_cap_applied_fraction": self.cap_applied_count / denominator,
        }


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern:
        return -1
    for index in range(len(sequence) - len(pattern) + 1):
        if list(sequence[index : index + len(pattern)]) == list(pattern):
            return index
    return -1


def post_think_start(token_ids: torch.Tensor, tokenizer: Any) -> int:
    """Mirror StateBridge's first-</think> filtering boundary."""
    ids = token_ids.reshape(-1).tolist()
    marker = tokenizer.encode("</think>", add_special_tokens=False)
    position = _find_subsequence(ids, marker)
    boundary = position + len(marker) if position >= 0 else 0
    return boundary if 0 < boundary < len(ids) else 0


def last_k_indices(length: int, *, k: int, start: int = 0) -> list[int]:
    if not 0 <= start <= length:
        raise ValueError("start must be within the trajectory")
    scope_length = length - start
    first = start if scope_length <= k else length - k
    return list(range(first, length))


def facility_location_coverage(
    states: torch.Tensor,
    *,
    k_tail: int,
    k_representative: int,
    eps: float = 1e-6,
) -> tuple[list[int], Dict[str, Any]]:
    """Greedy facility-location coverage with fixed final-tail anchors."""
    if states.ndim == 3:
        if states.shape[0] != 1:
            raise ValueError("Coverage selection currently requires batch size one")
        sequence = states[0]
    elif states.ndim == 2:
        sequence = states
    else:
        raise ValueError("states must have shape [T, hidden] or [1, T, hidden]")
    length = int(sequence.shape[0])
    total_budget = k_tail + k_representative
    if length <= total_budget:
        indices = list(range(length))
        tail_start = max(0, length - k_tail)
        tail_indices = list(range(tail_start, length))
        if length and tail_indices:
            normalized = F.normalize(sequence.float(), dim=-1, eps=eps)
            similarities_to_tail = (
                normalized @ normalized[tail_start:].T + 1.0
            ) * 0.5
            objective_before = float(
                similarities_to_tail.max(dim=1).values.sum().cpu()
            )
        else:
            objective_before = 0.0
        return indices, {
            "tail_indices": tail_indices,
            "coverage_indices": list(range(tail_start)),
            "facility_objective_before": objective_before,
            "facility_objective_after": float(length),
            "facility_objective_history": [objective_before, float(length)],
        }
    if k_tail <= 0:
        raise ValueError("coverage64 requires at least one tail anchor")

    normalized = F.normalize(sequence.float(), dim=-1, eps=eps)
    pre_tail_count = length - k_tail
    tail_indices = list(range(pre_tail_count, length))
    similarities_to_tail = (
        normalized @ normalized[pre_tail_count:].T + 1.0
    ) * 0.5
    best = similarities_to_tail.max(dim=1).values
    objective_before = float(best.sum().cpu())

    candidates = normalized[:pre_tail_count]
    similarities = (normalized @ candidates.T + 1.0) * 0.5
    available = torch.ones(pre_tail_count, dtype=torch.bool, device=sequence.device)
    selected: list[int] = []
    history = [objective_before]
    steps = min(k_representative, pre_tail_count)
    for _ in range(steps):
        gains = (similarities - best[:, None]).clamp_min(0).sum(dim=0)
        gains = gains.masked_fill(~available, -1.0)
        chosen = int(torch.argmax(gains).item())
        selected.append(chosen)
        available[chosen] = False
        best = torch.maximum(best, similarities[:, chosen])
        history.append(float(best.sum().cpu()))

    indices = sorted(selected + tail_indices)
    if len(indices) != len(set(indices)):
        raise RuntimeError("Coverage selector produced duplicate indices")
    return indices, {
        "tail_indices": tail_indices,
        "coverage_indices": sorted(selected),
        "facility_objective_before": objective_before,
        "facility_objective_after": history[-1],
        "facility_objective_history": history,
    }


def select_tmr_memory(
    trajectories: Mapping[int, torch.Tensor],
    token_ids: torch.Tensor,
    tokenizer: Any,
    config: TMRConfig,
) -> TMRMemory:
    config.validate()
    lengths = {layer: int(value.shape[1]) for layer, value in trajectories.items()}
    if set(lengths) != set(config.tmr_layers):
        raise ValueError(f"Captured trajectory layers disagree with config: {lengths}")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Captured trajectory lengths disagree: {lengths}")
    length = next(iter(lengths.values()))
    if int(token_ids.shape[1]) != length:
        raise ValueError(
            f"Trajectory/token length mismatch: trajectories={length}, "
            f"tokens={token_ids.shape[1]}"
        )

    think_start = post_think_start(token_ids, tokenizer)
    coverage: Dict[str, Any] = {}
    if config.tmr_selection == "last64":
        indices = last_k_indices(length, k=config.tmr_k, start=think_start)
        scope_start = think_start
    else:
        scope_start = think_start if config.coverage_scope == "post_think" else 0
        reference = trajectories[config.coverage_selection_layer][:, scope_start:, :]
        local_indices, coverage = facility_location_coverage(
            reference,
            k_tail=config.coverage_k_tail,
            k_representative=config.coverage_k_representative,
            eps=config.eps,
        )
        indices = [scope_start + index for index in local_indices]
        for key in ("tail_indices", "coverage_indices"):
            coverage[key] = [scope_start + index for index in coverage[key]]

    index_tensor = torch.tensor(indices, dtype=torch.long, device=token_ids.device)
    states = {
        layer: trajectory.index_select(1, index_tensor.to(trajectory.device)).detach()
        for layer, trajectory in trajectories.items()
    }
    final64_start = max(0, length - 64)
    diagnostic = {
        "generated_token_count": length,
        "selected_position_indices": indices,
        "selected_position_count": len(indices),
        "tmr_selection": config.tmr_selection,
        "coverage_scope": config.coverage_scope,
        "scope_start": scope_start,
        "post_think_start": think_start,
        "memory_tensor_shapes": {
            str(layer): list(states[layer].shape) for layer in config.tmr_layers
        },
        "fraction_from_final_64": (
            sum(index >= final64_start for index in indices) / len(indices)
            if indices
            else 0.0
        ),
        **coverage,
    }
    memory = TMRMemory(states=states, selected_indices=indices, selection_diagnostic=diagnostic)
    memory.validate(config.tmr_layers)
    return memory


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, groups, length, head_dim
    )
    return expanded.reshape(batch, kv_heads * groups, length, head_dim)


def project_external_memory(attention: Any, memory: torch.Tensor) -> ProjectedMemory:
    """Apply the frozen native K/V projections without RoPE."""
    batch, length, _ = memory.shape
    head_dim = int(attention.head_dim)
    kv_heads = attention.k_proj.out_features // head_dim
    shape = (batch, length, kv_heads, head_dim)
    key = attention.k_proj(memory).view(shape)
    if hasattr(attention, "k_norm"):
        key = attention.k_norm(key)
    key = key.transpose(1, 2)
    value = attention.v_proj(memory).view(shape).transpose(1, 2)
    groups = int(getattr(attention, "num_key_value_groups", 1))
    return ProjectedMemory(key=_repeat_kv(key, groups), value=_repeat_kv(value, groups))


def external_memory_residual(
    attention: Any,
    normalized_hidden: torch.Tensor,
    projected_memory: ProjectedMemory,
    self_attention_output: torch.Tensor,
    *,
    entropy_gate: bool = True,
    norm_ratio: float = 0.25,
    force_gate_zero: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute one position-free frozen-attention residual branch."""
    batch, query_length, _ = normalized_hidden.shape
    if force_gate_zero:
        zeros = torch.zeros(
            batch, query_length, device=normalized_hidden.device, dtype=torch.float32
        )
        self_norm = self_attention_output.float().norm(dim=-1)
        return torch.zeros_like(self_attention_output), {
            "entropy": torch.ones_like(zeros),
            "gate": zeros,
            "external_norm": zeros,
            "raw_external_norm": zeros,
            "self_norm": self_norm,
            "cap_applied": torch.zeros_like(zeros, dtype=torch.bool),
        }

    head_dim = int(attention.head_dim)
    query_heads = attention.q_proj.out_features // head_dim
    query = attention.q_proj(normalized_hidden).view(
        batch, query_length, query_heads, head_dim
    )
    if hasattr(attention, "q_norm"):
        query = attention.q_norm(query)
    query = query.transpose(1, 2)

    key = projected_memory.key
    value = projected_memory.value
    if key.shape[1] != query_heads or value.shape[1] != query_heads:
        raise ValueError(
            f"GQA expansion mismatch: Q={query.shape}, K={key.shape}, V={value.shape}"
        )
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores = scores * float(getattr(attention, "scaling", head_dim**-0.5))
    probabilities = torch.softmax(scores, dim=-1)

    mean_probability = probabilities.mean(dim=1)
    memory_length = int(mean_probability.shape[-1])
    if memory_length == 1:
        entropy = torch.zeros_like(mean_probability[..., 0])
    else:
        entropy = -(
            mean_probability
            * torch.log(mean_probability.clamp_min(eps))
        ).sum(dim=-1) / math.log(memory_length)
    gate = (1.0 - entropy).clamp(0.0, 1.0) if entropy_gate else torch.ones_like(entropy)

    context = torch.matmul(probabilities, value.float())
    context = context.transpose(1, 2).reshape(batch, query_length, -1)
    context = context.to(attention.o_proj.weight.dtype)
    context = context * gate.unsqueeze(-1).to(context.dtype)
    external = attention.o_proj(context)

    raw_external_norm = external.float().norm(dim=-1)
    self_norm = self_attention_output.float().norm(dim=-1)
    allowed = norm_ratio * (self_norm + eps)
    cap_scale = torch.minimum(
        torch.ones_like(allowed), allowed / (raw_external_norm + eps)
    )
    safe_external = external * cap_scale.unsqueeze(-1).to(external.dtype)
    external_norm = safe_external.float().norm(dim=-1)
    return safe_external, {
        "entropy": entropy,
        "gate": gate,
        "external_norm": external_norm,
        "raw_external_norm": raw_external_norm,
        "self_norm": self_norm,
        "cap_applied": cap_scale < (1.0 - 1e-6),
    }


def decoder_layers(model: Any) -> Sequence[Any]:
    current = model
    if hasattr(current, "model"):
        current = current.model
    if hasattr(current, "model") and hasattr(current.model, "layers"):
        return current.model.layers
    if hasattr(current, "layers"):
        return current.layers
    raise ValueError(f"Cannot locate decoder layers in {type(model)}")


class TMRGenerationHooks:
    """Scoped hooks for sender capture and receiver external-memory retrieval."""

    def __init__(
        self,
        model: Any,
        config: TMRConfig,
        *,
        memory: Optional[TMRMemory],
        capture_trajectory: bool,
    ) -> None:
        self.model = model
        self.config = config
        self.layers = decoder_layers(model)
        config.validate(num_layers=len(self.layers))
        if memory is not None:
            memory.validate(config.tmr_layers)
        self.memory = memory
        self.capture_trajectory = capture_trajectory
        self._handles: list[Any] = []
        self._pending_hidden: Dict[int, torch.Tensor] = {}
        self._captures: Dict[int, list[torch.Tensor]] = {
            layer: [] for layer in config.tmr_layers
        }
        self._statistics: Dict[int, AttentionStatistics] = {
            layer: AttentionStatistics() for layer in config.tmr_layers
        }
        self._projected: Dict[int, ProjectedMemory] = {}

    def __enter__(self) -> "TMRGenerationHooks":
        if self.memory is not None:
            for layer_index in self.config.tmr_layers:
                attention = self.layers[layer_index].self_attn
                self._projected[layer_index] = project_external_memory(
                    attention, self.memory.states[layer_index]
                )
        for layer_index in self.config.tmr_layers:
            attention = self.layers[layer_index].self_attn
            self._handles.append(
                attention.register_forward_pre_hook(
                    partial(self._pre_hook, layer_index), with_kwargs=True
                )
            )
            if self.memory is not None:
                self._handles.append(
                    attention.register_forward_hook(
                        partial(self._post_hook, layer_index), with_kwargs=True
                    )
                )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_hidden.clear()

    def _pre_hook(
        self,
        layer_index: int,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        del module
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if hidden is None or hidden.ndim != 3:
            raise RuntimeError(f"Layer {layer_index} did not receive [B,T,H] states")
        if self.capture_trajectory:
            self._captures[layer_index].append(hidden[:, -1:, :].detach().clone())
        if self.memory is not None:
            self._pending_hidden[layer_index] = hidden

    def _post_hook(
        self,
        layer_index: int,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Any:
        del args, kwargs
        if layer_index not in self._pending_hidden:
            raise RuntimeError(f"Missing normalized receiver states for layer {layer_index}")
        if not isinstance(output, tuple) or not output:
            raise RuntimeError("Qwen attention hook expected a non-empty tuple output")
        self_output = output[0]
        residual, values = external_memory_residual(
            module,
            self._pending_hidden.pop(layer_index),
            self._projected[layer_index],
            self_output,
            entropy_gate=self.config.tmr_entropy_gate,
            norm_ratio=self.config.tmr_norm_ratio,
            force_gate_zero=self.config.tmr_force_gate_zero,
            eps=self.config.eps,
        )
        self._statistics[layer_index].update(values)
        return (self_output + residual, *output[1:])

    def trajectories(self) -> Dict[int, torch.Tensor]:
        result: Dict[int, torch.Tensor] = {}
        lengths: Dict[int, int] = {}
        for layer_index, captures in self._captures.items():
            if captures:
                result[layer_index] = torch.cat(captures, dim=1)
                lengths[layer_index] = int(result[layer_index].shape[1])
            elif self.capture_trajectory:
                raise RuntimeError(f"No trajectory captured at layer {layer_index}")
        if lengths and len(set(lengths.values())) != 1:
            raise RuntimeError(f"TMR layer trajectory lengths disagree: {lengths}")
        return result

    def attention_summary(self) -> Dict[str, Any]:
        per_layer = {
            str(layer): stats.summary() for layer, stats in self._statistics.items()
        }
        total_count = sum(stats.count for stats in self._statistics.values())
        denominator = max(total_count, 1)
        aggregate = {
            "positions": total_count,
            "mean_external_attention_entropy": sum(
                stats.entropy_sum for stats in self._statistics.values()
            )
            / denominator,
            "mean_tmr_gate": sum(
                stats.gate_sum for stats in self._statistics.values()
            )
            / denominator,
            "max_tmr_gate": max(
                (stats.max_gate for stats in self._statistics.values()), default=0.0
            ),
            "mean_external_residual_norm": sum(
                stats.external_norm_sum for stats in self._statistics.values()
            )
            / denominator,
            "mean_raw_external_residual_norm": sum(
                stats.raw_external_norm_sum for stats in self._statistics.values()
            )
            / denominator,
            "mean_self_attention_residual_norm": sum(
                stats.self_norm_sum for stats in self._statistics.values()
            )
            / denominator,
            "norm_cap_applied_fraction": sum(
                stats.cap_applied_count for stats in self._statistics.values()
            )
            / denominator,
        }
        return {"aggregate": aggregate, "per_layer": per_layer}


def parameter_identity_snapshot(model: Any) -> Dict[int, tuple[str, bool, int]]:
    return {
        id(parameter): (name, bool(parameter.requires_grad), parameter.numel())
        for name, parameter in model.named_parameters()
    }


def assert_no_new_trainable_parameters(
    before: Mapping[int, tuple[str, bool, int]], model: Any
) -> int:
    after = parameter_identity_snapshot(model)
    new_trainable = [
        metadata for identity, metadata in after.items()
        if identity not in before and metadata[1]
    ]
    if new_trainable:
        raise AssertionError(f"TMR introduced trainable parameters: {new_trainable}")
    return 0
