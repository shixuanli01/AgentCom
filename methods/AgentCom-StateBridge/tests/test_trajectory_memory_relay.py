import torch
from torch import nn

from methods.state_bridge import select_hidden_states
from methods.trajectory_memory_relay import (
    TMRConfig,
    TMRGenerationHooks,
    TMRMemory,
    assert_no_new_trainable_parameters,
    external_memory_residual,
    facility_location_coverage,
    parameter_identity_snapshot,
    project_external_memory,
    select_tmr_memory,
)


class MarkerTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text == "</think>"
        assert not add_special_tokens
        return [999]


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.num_key_value_groups = 2
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoder()])


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()


def _config(selection="last64"):
    return TMRConfig(
        tmr_selection=selection,
        tmr_layers=(0,),
        coverage_selection_layer=0,
    )


def test_tmr_last64_matches_statebridge_post_think_last_k():
    trajectory = torch.arange(100 * 8, dtype=torch.float32).reshape(1, 100, 8)
    token_ids = torch.arange(100, dtype=torch.long).reshape(1, 100)
    token_ids[0, 20] = 999

    memory = select_tmr_memory(
        {0: trajectory}, token_ids, MarkerTokenizer(), _config()
    )
    filtered_hidden = trajectory[:, 21:, :]
    filtered_tokens = token_ids[:, 21:]
    _, _, local_indices = select_hidden_states(
        filtered_hidden, filtered_tokens, k=64, method="last_k"
    )
    expected = [21 + index for index in local_indices]

    assert memory.selected_indices == expected == list(range(36, 100))
    assert torch.equal(memory.states[0], trajectory[:, -64:, :])


def test_tmr_memory_requires_layer_length_consistency():
    memory = TMRMemory(
        states={0: torch.zeros(1, 3, 8), 1: torch.zeros(1, 2, 8)},
        selected_indices=[0, 1, 2],
        selection_diagnostic={},
    )
    try:
        memory.validate((0, 1))
    except ValueError as error:
        assert "lengths disagree" in str(error)
    else:
        raise AssertionError("Expected inconsistent TMR layers to fail")


def test_tmr_hooks_add_no_trainable_parameters():
    model = FakeCausalLM()
    before = parameter_identity_snapshot(model)
    config = _config()
    memory = TMRMemory(
        states={0: torch.randn(1, 4, 8)},
        selected_indices=[0, 1, 2, 3],
        selection_diagnostic={},
    )
    with TMRGenerationHooks(
        model, config, memory=memory, capture_trajectory=True
    ):
        assert assert_no_new_trainable_parameters(before, model) == 0
    assert assert_no_new_trainable_parameters(before, model) == 0


def test_gate_zero_returns_exact_zero_residual():
    torch.manual_seed(2)
    attention = FakeAttention()
    hidden = torch.randn(1, 5, 8)
    memory = torch.randn(1, 7, 8)
    self_output = torch.randn(1, 5, 8)
    projected = project_external_memory(attention, memory)

    residual, values = external_memory_residual(
        attention,
        hidden,
        projected,
        self_output,
        force_gate_zero=True,
    )

    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(self_output + residual, self_output)
    assert torch.count_nonzero(values["gate"]) == 0


def test_norm_cap_is_respected_per_receiver_position():
    torch.manual_seed(3)
    attention = FakeAttention()
    hidden = torch.randn(1, 6, 8)
    memory = torch.randn(1, 9, 8)
    self_output = torch.randn(1, 6, 8)
    projected = project_external_memory(attention, memory)

    residual, _ = external_memory_residual(
        attention,
        hidden,
        projected,
        self_output,
        entropy_gate=False,
        norm_ratio=0.25,
    )

    residual_norm = residual.float().norm(dim=-1)
    self_norm = self_output.float().norm(dim=-1)
    assert torch.all(residual_norm <= 0.25 * (self_norm + 1e-6) + 2e-6)


def test_position_free_memory_attention_is_permutation_invariant():
    torch.manual_seed(5)
    attention = FakeAttention()
    hidden = torch.randn(1, 4, 8)
    memory = torch.randn(1, 12, 8)
    self_output = torch.randn(1, 4, 8)
    permutation = torch.randperm(memory.shape[1])

    first, _ = external_memory_residual(
        attention,
        hidden,
        project_external_memory(attention, memory),
        self_output,
    )
    second, _ = external_memory_residual(
        attention,
        hidden,
        project_external_memory(attention, memory[:, permutation, :]),
        self_output,
    )

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-5)


def test_coverage64_is_deterministic_complete_and_monotonic():
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(1, 100, 12, generator=generator)

    first_indices, first = facility_location_coverage(
        states, k_tail=16, k_representative=48
    )
    second_indices, second = facility_location_coverage(
        states, k_tail=16, k_representative=48
    )

    assert first_indices == second_indices
    assert first == second
    assert len(first_indices) == len(set(first_indices)) == 64
    assert first["tail_indices"] == list(range(84, 100))
    assert len(first["coverage_indices"]) == 48
    assert all(index < 84 for index in first["coverage_indices"])
    assert first_indices == sorted(first_indices)
    history = first["facility_objective_history"]
    assert all(right >= left for left, right in zip(history, history[1:]))
    assert first["facility_objective_after"] >= first["facility_objective_before"]


def test_coverage64_gathers_identical_indices_from_every_layer():
    generator = torch.Generator().manual_seed(11)
    trajectories = {
        0: torch.randn(1, 90, 8, generator=generator),
        1: torch.randn(1, 90, 8, generator=generator),
        2: torch.randn(1, 90, 8, generator=generator),
    }
    token_ids = torch.arange(90).reshape(1, 90)
    config = TMRConfig(
        tmr_selection="coverage64",
        tmr_layers=(0, 1, 2),
        coverage_selection_layer=2,
    )

    memory = select_tmr_memory(
        trajectories, token_ids, MarkerTokenizer(), config
    )

    assert len(memory.selected_indices) == 64
    for layer in config.tmr_layers:
        expected = trajectories[layer][:, memory.selected_indices, :]
        assert torch.equal(memory.states[layer], expected)
