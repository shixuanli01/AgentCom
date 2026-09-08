# AgentCom-StateBridge

This repository contains the current AgentCom research built on StateBridge.
It focuses on improving which hidden states are communicated between agents
while keeping the original StateBridge alignment and receiver interface fixed.

The current experiment replaces StateBridge's fixed last-`K` selection with an
optional trajectory turning-point selector. Both methods transmit exactly
`K=64` states; only the selected token positions differ.

Large model weights, Hugging Face caches, generated traces, and result files
are intentionally excluded from Git.

## Repository Layout

| Path | Role |
|---|---|
| `methods/AgentCom-StateBridge/` | Active implementation, tests, and experiment plans |
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

Run a one-item turning-point smoke:

```bash
cd methods/AgentCom-StateBridge
python -m methods.state_bridge \
  --model Qwen/Qwen3-4B \
  --task medqa \
  --gpus 0 \
  --seed 42 \
  --limit 1 \
  --max_prefix_tokens 64 \
  --selection_method turning_point \
  --turning_point_window_size 8 \
  --selection_diagnostics
```

## Current Controlled Comparison

The authoritative MedQA comparison holds these variables fixed:

- model: `Qwen/Qwen3-4B`;
- dataset: bundled 300-row MedQA subset;
- seed: `42`;
- prompt topology: sequential Planner, Critic, Refiner, Judger;
- decoding: temperature `0.6`, top-p `0.95`;
- message budget: `K=64`;
- alignment regularization: `0.001`;
- vocabulary anchoring: `0.3`.

The control uses `--selection_method last_k`. The intervention uses
`--selection_method turning_point --turning_point_window_size 8`. No other
method, model, prompt, or decoding change belongs in this comparison.

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

## License

Code is provided under Apache-2.0. StateBridge and the bundled datasets retain
their upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
