import torch

from methods.state_bridge import select_hidden_states, turning_point_scores


def _sequence(length: int, width: int = 3):
    hidden = torch.arange(length * width, dtype=torch.float32).reshape(1, length, width)
    tokens = torch.arange(length, dtype=torch.long).reshape(1, length)
    return hidden, tokens


def test_last_k_reproduces_original_indices():
    hidden, tokens = _sequence(100)

    selected_hidden, selected_tokens, indices = select_hidden_states(
        hidden, tokens, k=64, method="last_k"
    )

    assert indices == list(range(36, 100))
    assert torch.equal(selected_hidden, hidden[:, -64:, :])
    assert torch.equal(selected_tokens, tokens[:, -64:])


def test_short_sequence_is_returned_unchanged():
    hidden, tokens = _sequence(32)

    selected_hidden, selected_tokens, indices = select_hidden_states(
        hidden, tokens, k=64, method="turning_point"
    )

    assert indices == list(range(32))
    assert torch.equal(selected_hidden, hidden)
    assert torch.equal(selected_tokens, tokens)


def test_turning_point_peaks_near_known_transition():
    direction_a = torch.tensor([1.0, 0.0, 0.0]).repeat(50, 1)
    direction_b = torch.tensor([0.0, 1.0, 0.0]).repeat(50, 1)
    hidden = torch.cat([direction_a, direction_b], dim=0).unsqueeze(0)

    scores = turning_point_scores(hidden, window_size=8)

    assert int(scores.argmax()) in {49, 50}
    assert scores[49] > scores[40]
    assert scores[50] > scores[60]


def test_turning_point_selection_returns_chronological_indices():
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, 100, 8, generator=generator).to(torch.bfloat16)
    tokens = torch.arange(100).reshape(1, 100)

    selected_hidden, _, indices = select_hidden_states(
        hidden, tokens, k=16, method="turning_point", window_size=4
    )

    assert len(indices) == 16
    assert indices == sorted(indices)
    assert selected_hidden.dtype == torch.bfloat16


def test_hidden_states_and_tokens_use_identical_indices():
    hidden, tokens = _sequence(100, width=2)

    selected_hidden, selected_tokens, indices = select_hidden_states(
        hidden, tokens, k=12, method="turning_point", window_size=5
    )

    assert selected_tokens[0].tolist() == indices
    assert selected_hidden[0, :, 0].tolist() == [float(index * 2) for index in indices]
