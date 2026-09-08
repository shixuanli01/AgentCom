# Patch Ledger

The upstream base is StateBridge release `0.1.0`, commit
`3f6bf5442c6e8848555a6132516e6d36f35444fb`.

## A001: Turning-point state selection

- File: `methods/state_bridge.py`
- Adds `select_hidden_states` with `last_k` and `turning_point` modes.
- Adds local trajectory-change scores based on left/right window cosine
  distance.
- Preserves selected state/token pairing and chronological order.
- Adds CLI and worker configuration for selector, window size, and optional
  diagnostics.
- Does not change Procrustes alignment, norm calibration, vocabulary anchoring,
  receiver prompts, decoding, or message budget.

Tests: `tests/test_state_selection.py`.

## Existing AgentCom modules

The `agentcom/` modules and associated tests implement earlier StateBridge
multi-path, LLM-judge, and Refiner causal analyses. They are retained because
they directly motivate the current position-selection experiment; they are not
part of the turning-point intervention itself.
