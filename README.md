# AgentCom: Communication Audits and Training-Free Latent Relays

This repository contains controlled communication audits and training-free
latent communication research built around a frozen StateBridge control. The
current evaluation track is **Independent -> Communicate -> Revise (ICR)** with
**Evidence-Grounded Revision (EGR)**; TMR remains the active latent-method track.

Instead of mapping final-layer states back into token-input embedding space,
TMR exposes native hidden states as position-free external memory. The receiver
reads that memory with its own frozen Q/K/V/O attention projections. TMR adds no
trainable parameters, pseudo tokens, Procrustes alignment, vocabulary snapping,
or extra LLM calls.

Large model weights, Hugging Face caches, generated traces, and result files
are intentionally excluded from Git.

## Repository Layout

| Path | Role |
|---|---|
| `methods/AgentCom-StateBridge/` | Active TMR implementation, evaluator, tests, and protocols |
| `StateBridge-repro-qwen3-4b/` | Frozen upstream StateBridge control and reproduction protocol |
| `docs/REMOTE_REPRODUCTION.md` | Fresh-server installation and run commands |
| `AGENTS.md` | Instructions and scientific guardrails for coding agents |

The active method and frozen control both start from StateBridge release
`0.1.0`, commit `3f6bf5442c6e8848555a6132516e6d36f35444fb`.

## Quick Start

```bash
git clone git@github.com:shixuanli01/AgentCom.git
cd AgentCom
bash scripts/bootstrap_remote.sh cu128
source .venv-agentcom/bin/activate
bash scripts/verify_checkout.sh
```

Set the Hugging Face cache to a large disk before downloading Qwen3-4B:

```bash
export HF_HOME=/data/$USER/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

Run the tests and a one-item TMR-last64 smoke:

```bash
cd methods/AgentCom-StateBridge
PYTHONPATH=. python -m agentcom.tmr_eval \
  --communication-method tmr \
  --tmr-selection last64 \
  --run-dir artifacts/tmr_v1/smoke_last64_seed42 \
  --limit 1
```

## Current Controlled Comparison

### ICR and EGR

ICR generates two independent beliefs per item and evaluates both directional
handoffs under matched receiver revision semantics. Its channel controls are
no-message, full text, StateBridge, and LatentMAS, with self/other controls for
causal audits. EGR adds deterministic claim-suppressed evidence and symmetric
contrast adjudication. EGR is a receiver policy, not a latent transport, so its
system-level scores are not reported as pure channel wins.

The frozen EGR V1 definition and current MedQA/ARC/GSM8K results are documented
in:

- [EGR V1 method freeze](methods/AgentCom-StateBridge/experiments/EGR_V1_METHOD_FREEZE_ZH.md)
- [EGR cross-benchmark report](methods/AgentCom-StateBridge/experiments/EGR_CROSS_BENCHMARK_REPORT_ZH.md)
- [Cross-benchmark protocol](methods/AgentCom-StateBridge/experiments/CROSS_BENCHMARK_FIRST_WAVE_ZH.md)

Run a two-item smoke before a sampled benchmark:

```bash
cd methods/AgentCom-StateBridge
PYTHONPATH=. WORKERS_PER_GPU=2 BENCHMARK_LIMIT=2 \
  bash scripts/run_cross_benchmark_first_wave.sh gsm8k \
  artifacts/cross_benchmark/smoke_gsm8k_2

PYTHONPATH=. WORKERS_PER_GPU=2 BENCHMARK_SAMPLE_SIZE=300 \
  bash scripts/run_cross_benchmark_first_wave.sh gsm8k
```

Generated traces remain under ignored `artifacts/`; completed summary reports
should be transcribed into versioned experiment documents with exact counts and
freeze fingerprints.

### TMR

The authoritative MedQA comparison holds these variables fixed:

- model: `Qwen/Qwen3-4B`;
- dataset: bundled 300-row MedQA subset;
- seed: `42`;
- prompt topology: sequential Planner, Critic, Refiner, Judger;
- decoding: temperature `0.6`, top-p `0.95`;
- message budget: `K=64`;
- source/receiver layers: `11,23,35`;
- entropy gate: enabled;
- TMR residual norm cap: `0.25`.

The frozen StateBridge control maps post-thinking last-64 final-layer states to
input embeddings with whitened Procrustes alignment. `tmr_last64` preserves the
same position-selection semantics and changes only transport. `tmr_coverage64`
keeps TMR transport fixed and selects 48 facility-location representatives plus
16 final-tail anchors from the full trajectory.

Run both MedQA300 variants sequentially:

```bash
cd methods/AgentCom-StateBridge
tmux new-session -d -s tmr-medqa \
  'bash scripts/run_tmr_medqa300.sh artifacts/tmr_v1 \
   > artifacts/tmr_v1/server_queue.log 2>&1'
tail -f artifacts/tmr_v1/server_queue.log
```

Runs are durable at item boundaries. Re-running the command resumes completed
records. If an existing StateBridge multipath `summary.json` is available, set
`BASELINE=/path/to/summary.json` before launching to produce paired metrics;
otherwise TMR runs normally and baseline fields remain null.

## Data

MedQA and GPQA-Diamond are included because they are small and their notices
permit redistribution. GSM8K, AIME, ARC, MBPP+, and HumanEval+ are downloaded
by Hugging Face Datasets on first use. Qwen3-4B weights are never committed.
See [the remote runbook](docs/REMOTE_REPRODUCTION.md) for exact identifiers and
commands.

## Documentation

- [Coding-agent instructions](AGENTS.md)
- [Remote reproduction runbook](docs/REMOTE_REPRODUCTION.md)
- [Repository manifest](docs/REPOSITORY_MANIFEST.md)
- [Method boundary](methods/AgentCom-StateBridge/AGENTCOM_METHOD.md)
- [Current experiment index](methods/AgentCom-StateBridge/experiments/README.md)
- [TMR V1 protocol](methods/AgentCom-StateBridge/experiments/TMR_V1_PROTOCOL_ZH.md)

## License

Code is provided under Apache-2.0. StateBridge and the bundled datasets retain
their upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
