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

The active track is the **ICR-V3 baseline sweep**: four communication channels
on the full test set of every benchmark, one replication.

- channels: `none`, `true_text`, `true_statebridge`, `true_latentmas`;
- benchmarks: MedQA 300, ARC-Challenge 1,165, GSM8K 1,319, GPQA-Diamond 198,
  HumanEval+ 164;
- model/seed: Qwen3-4B, global seed 42, replication `seed_pair_00`;
- decoding: temperature 0.6, top-p 0.95.

V3 exists because V2 compared channels under visibly different prompts: the
latent marker sat at the front of the user turn while the Text message sat
mid-prompt, and only the latent conditions received an extra "use the external
message as evidence" instruction. V3 fixes one five-block template

    [Task / Question] -> [Receiver's own prior] -> [External information]
    -> [How to integrate] -> [Output format]

with a single condition-varying slot, adds a per-dataset output contract, and
repairs five scoring defects. Read
`methods/AgentCom-StateBridge/experiments/ICR_V3_PROTOCOL_ZH.md` before
changing any of it.

Do not alter the V3 prompts, the message slot position, the output contracts,
or the parser while a sweep is in flight. Any change to them is a new protocol
version with its own artifact root, never an edit of a completed run.

Relevant files:

- `methods/AgentCom-StateBridge/icr/prompts_v3.py`
- `methods/AgentCom-StateBridge/icr/parsing_v3.py`
- `methods/AgentCom-StateBridge/icr/channels.py`
- `methods/AgentCom-StateBridge/icr/runtime.py`
- `methods/AgentCom-StateBridge/scripts/run_v3_baselines.sh`
- `methods/AgentCom-StateBridge/tests/test_icr_v3.py`

Adding a communication channel: subclass `CommunicationChannel` in
`icr/channels.py`, register it in `make_channel()`, add the condition name to
`CONDITIONS` in `icr/__init__.py`, and render its payload through
`external_block()` in `icr/prompts_v3.py` so the visible text stays matched.
Copy the alignment assertions from `tests/test_icr_v3.py` for the new
condition; they catch prompt asymmetry before any GPU time is spent.

Earlier tracks, retained as history: EGR (receiver-side evidence policy) and
TMR (training-free latent transport).

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
