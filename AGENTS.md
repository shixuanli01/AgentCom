# Coding Agent Instructions

This repository has one active method and one frozen control.

## Start Here

1. Read `README.md`, this file, and `docs/REMOTE_REPRODUCTION.md`.
2. Run `nvidia-smi` and verify that Python 3.10 is available.
3. Run `bash scripts/bootstrap_remote.sh cu128` on RTX 5090, then
   `source .venv-agentcom/bin/activate`.
4. Run `bash scripts/verify_checkout.sh`.
5. Start GPU work with `--limit 1`; inspect the JSON/log before a full run.

## Repository Boundaries

- All new method work belongs in `methods/AgentCom-StateBridge/`.
- `StateBridge-repro-qwen3-4b/` is the frozen upstream control. Do not add TMR
  or other active-method changes to it.
- Generated files belong in ignored `artifacts/`, `results/`, `logs_new/`, or
  `reproduction/runs/` directories.
- Do not reintroduce historical Task 5-14, LatentMAS, KVComm, or C2C code into
  this focused repository.

## Current Experiment

The active method is training-free Trajectory Memory Relay:

- transport control: StateBridge post-think last-64 + Procrustes prefix;
- transport intervention: `tmr_last64` native external memory;
- selection intervention: `tmr_coverage64` with 48 coverage + 16 tail states;
- source/receiver layers: `11,23,35`;
- fixed communication budget: `K=64` per layer;
- model/task: Qwen3-4B on MedQA 300;
- seed: `42`;
- prompts, parser, decoding, roles, and LLM-call count: unchanged.

TMR uses frozen native Q/K/V/O attention without RoPE or positional IDs on the
external memory. It adds an entropy gate and a fixed `0.25` residual norm cap.
No training or new parameters are allowed.

Relevant files:

- `methods/AgentCom-StateBridge/methods/trajectory_memory_relay.py`
- `methods/AgentCom-StateBridge/agentcom/tmr_eval.py`
- `methods/AgentCom-StateBridge/tests/test_trajectory_memory_relay.py`
- `methods/AgentCom-StateBridge/experiments/TMR_V1_PROTOCOL_ZH.md`

## Scientific Guardrails

- Do not tune selection, gate, layers, or norm cap on full test accuracy.
- Keep model, data order, prompt, seed, parser, decoding, and `K` identical in
  the control and intervention.
- Report exact counts and paired per-item changes, not only percentages.
- Label smoke, partial, resumed, and full runs explicitly.
- Do not claim that final accuracy alone proves communication quality.
- Never commit model weights, hidden-state tensors, caches, or generated
  per-sample traces.

## Validation

```bash
bash scripts/verify_checkout.sh

cd methods/AgentCom-StateBridge
pytest -q tests

cd ../../StateBridge-repro-qwen3-4b
pytest -q reproduction/test_protocol.py
```

Before committing, inspect `git status --short` and never force-add ignored
artifacts.
