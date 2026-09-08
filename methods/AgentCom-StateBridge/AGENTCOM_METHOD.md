# AgentCom on StateBridge

## Starting point

This worktree starts from StateBridge release `0.1.0`, commit
`3f6bf5442c6e8848555a6132516e6d36f35444fb`, on branch
`method/agentcom-statebridge`.

The official reproduction checkout is separate. Changes made here must never
alter or overwrite official StateBridge results.

## Scientific objective

Build the next AgentCom method on StateBridge's input-embedding communication
interface while preserving these evidence requirements:

1. derive a reusable rule- or policy-specific message from support evidence;
2. keep the Sender query-blind with respect to held-out Receiver states;
3. reuse the same message across multiple Receiver states;
4. train with independently resampled support views when training is needed;
5. evaluate matched, same-family-wrong, random, and no-message controls; and
6. separate rule induction from Receiver execution and base-model capability.

## Repository boundary

- Upstream StateBridge files remain the reference implementation.
- New implementation code belongs in `agentcom/`.
- New experiment configs and launchers belong in `experiments/`.
- New causal and leakage tests belong in `tests/`.
- Every deliberate upstream-file change must be listed in a patch ledger.

## Current checkpoint

The current minimal intervention challenges StateBridge's fixed last-64 state
selection. `last_k` remains the exact control. `turning_point` scores changes
along the final-layer hidden-state trajectory, selects the top 64 positions,
restores chronological order, and sends those states through the unchanged
StateBridge alignment and receiver injection path.

See `experiments/TURNING_POINT_SELECTION.md` and `PATCH_LEDGER.md`.
