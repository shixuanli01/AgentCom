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

The active method is **Trajectory Memory Relay (TMR)**, a training-free
alternative to StateBridge's final-layer Procrustes prefix. It captures a
single Planner trajectory at layers 11, 23, and 35 and exposes that trajectory
to each Receiver through the model's own frozen attention projections. No
learned projector, extra decoding path, or additional model is introduced.

The first controlled comparison keeps the communication budget at `K=64`:

- `tmr_last64`: the last 64 valid Planner positions from each captured layer;
- `tmr_coverage64`: 48 chronological coverage representatives plus a
  16-position tail anchor.

The upstream-style StateBridge path remains available as the frozen conceptual
control through `--communication-method statebridge`. Generated traces and
results must remain outside Git.

See `experiments/TMR_V1_PROTOCOL_ZH.md` for the frozen v1 protocol and
`experiments/README.md` for launch commands.
