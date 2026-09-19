# AgentCom: Communication Audits and Training-Free Latent Relays

This repository contains controlled communication audits and training-free
latent communication research built around a frozen StateBridge control. The
evaluation framework is **Independent -> Communicate -> Revise (ICR)**: two
agents answer each item independently, then each revises once after receiving
the other's message through one communication channel.

The current track is **ICR-V3**, a full-test-set baseline sweep of four
channels — no-message, full text, StateBridge, and LatentMAS — under prompts
that are byte-identical across conditions apart from the message payload
itself. V2 compared channels under visibly different prompts, which confounded
transport with prompt position and wording; see the
[ICR-V3 protocol](methods/AgentCom-StateBridge/experiments/ICR_V3_PROTOCOL_ZH.md)
for the exact diff, the per-dataset output contracts, and the scoring fixes.

Earlier tracks are retained as history: **EGR** (a receiver-side evidence
policy) and **TMR** (a training-free latent transport).

Large model weights, Hugging Face caches, generated traces, and result files
are intentionally excluded from Git.

## Repository Layout

| Path | Role |
|---|---|
| `methods/AgentCom-StateBridge/icr/` | ICR framework: protocol, channels, runtime, analysis |
| `methods/AgentCom-StateBridge/` | Method implementations, evaluators, tests, and protocols |
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

Run the tests and a two-item baseline smoke:

```bash
cd methods/AgentCom-StateBridge
pytest -q tests

BENCHMARK_LIMIT=2 CUDA_DEVICES="0" \
  bash scripts/run_v3_baselines.sh medqa artifacts/icr_v3/smoke_medqa2
```

Inspect `artifacts/icr_v3/smoke_medqa2/analysis_v2/summary.md` before starting
a full set.

## Current Controlled Comparison

### ICR-V3 baselines

Four channels are evaluated on the full test set of every benchmark, one
replication, with `self`/`other` causal controls deferred to a later ablation:

| Benchmark | Items | Note |
|---|---:|---|
| MedQA | 300 | the upstream paper's fixed subset; no larger set is bundled |
| ARC-Challenge | 1,165 | 7 of 1,172 excluded by option count alone, see the protocol |
| GSM8K | 1,319 | full test split |
| GPQA-Diamond | 198 | full set |
| HumanEval+ | 164 | full set |

Conditions are `none`, `true_text`, `true_statebridge`, and `true_latentmas`.
Their revision prompts are byte-identical apart from the message payload, which
`tests/test_icr_v3.py` asserts directly. The LatentMAS KV cache remains a
front-anchored causal prefix because its positions are baked in on the sender
side; that residual asymmetry is a declared comparison boundary rather than an
implementation choice.

```bash
cd methods/AgentCom-StateBridge
bash scripts/run_v3_baselines.sh gsm8k          # all GPUs, resumable
bash scripts/run_v3_baselines.sh arc_challenge
```

Runs are durable at item boundaries; re-running resumes completed records.
Results land in `artifacts/icr_v3/<task>_full_seed42/analysis_v2/`.

### ICR-V2 and EGR (earlier track)

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

Instead of mapping final-layer states back into token-input embedding space,
TMR exposes native hidden states as position-free external memory. The receiver
reads that memory with its own frozen Q/K/V/O attention projections. TMR adds no
trainable parameters, pseudo tokens, Procrustes alignment, vocabulary snapping,
or extra LLM calls.

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
- [ICR-V3 protocol: prompts, output contracts, scoring fixes](methods/AgentCom-StateBridge/experiments/ICR_V3_PROTOCOL_ZH.md)
- [ICR protocol and prompt record (V2)](docs/ICR_PROTOCOL_AND_PROMPTS.md)
- [ICR experiment ledger](docs/ICR_EXPERIMENT_LEDGER.md)
- [Remote reproduction runbook](docs/REMOTE_REPRODUCTION.md)
- [Repository manifest](docs/REPOSITORY_MANIFEST.md)
- [Method boundary](methods/AgentCom-StateBridge/AGENTCOM_METHOD.md)
- [Current experiment index](methods/AgentCom-StateBridge/experiments/README.md)
- [TMR V1 protocol](methods/AgentCom-StateBridge/experiments/TMR_V1_PROTOCOL_ZH.md)

## License

Code is provided under Apache-2.0. StateBridge and the bundled datasets retain
their upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
