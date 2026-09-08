# Remote Reproduction

## 1. Clone

```bash
git clone git@github.com:shixuanli01/AgentCom.git
cd AgentCom
nvidia-smi
python3 --version
```

Python 3.10 is the conservative target. Put Hugging Face data on a large disk:

```bash
export HF_HOME=/data/$USER/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME"
```

## 2. Environment

RTX 5090 / CUDA 12.8, matching the tested local environment:

```bash
bash scripts/bootstrap_remote.sh cu128
source .venv-agentcom/bin/activate
```

A100/H100 with a CUDA 12.4-compatible driver:

```bash
bash scripts/bootstrap_remote.sh cu124
source .venv-agentcom/bin/activate
```

The tested cu128 lock uses PyTorch `2.7.1+cu128`, Transformers `4.51.3`,
Datasets `3.6.0`, Accelerate `1.7.0`, NumPy `2.2.6`, and tqdm `4.67.1`.

```bash
bash scripts/verify_checkout.sh
```

## 3. Model and Data

The model is `Qwen/Qwen3-4B`. It downloads automatically, or can be prefetched:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B")
PY
```

Bundled data:

| File | Rows |
|---|---:|
| `methods/AgentCom-StateBridge/data/medqa.json` | 300 |
| `methods/AgentCom-StateBridge/data/gpqa_diamond.json` | 198 |

Runtime Hugging Face datasets:

| Task | Identifier |
|---|---|
| GSM8K | `gsm8k`, config `main` |
| AIME 2024 | `HuggingFaceH4/aime_2024` |
| AIME 2025 | `yentinglin/aime_2025` |
| ARC-Challenge | `allenai/ai2_arc`, config `ARC-Challenge` |
| MBPP+ | `evalplus/mbppplus` |
| HumanEval+ | `evalplus/humanevalplus` |

Dataset revisions are not pinned by upstream StateBridge. For an authoritative
run, record the generated dataset fingerprint, model revision, `pip freeze`,
GPU model, and NVIDIA driver.

## 4. Tests

```bash
cd methods/AgentCom-StateBridge
pytest -q tests

cd ../../StateBridge-repro-qwen3-4b
pytest -q reproduction/test_protocol.py
```

These tests do not download Qwen3-4B.

## 5. Smoke Runs

Run the frozen last-`K` control:

```bash
cd methods/AgentCom-StateBridge
python -m methods.state_bridge \
  --model Qwen/Qwen3-4B --task medqa --gpus 0 --seed 42 --limit 1 \
  --prompt sequential --temperature 0.6 --max_new_tokens 8192 \
  --max_prefix_tokens 64 --selection_method last_k \
  --adaptive_reg 0.001 --snap_ratio 0.3
```

Run the turning-point intervention:

```bash
python -m methods.state_bridge \
  --model Qwen/Qwen3-4B --task medqa --gpus 0 --seed 42 --limit 1 \
  --prompt sequential --temperature 0.6 --max_new_tokens 8192 \
  --max_prefix_tokens 64 --selection_method turning_point \
  --turning_point_window_size 8 --selection_diagnostics \
  --adaptive_reg 0.001 --snap_ratio 0.3
```

## 6. Full MedQA Runs

Turning-point intervention:

```bash
python -u -m methods.state_bridge \
  --model Qwen/Qwen3-4B \
  --task medqa \
  --gpus 0 \
  --seed 42 \
  --prompt sequential \
  --temperature 0.6 \
  --max_new_tokens 8192 \
  --max_prefix_tokens 64 \
  --selection_method turning_point \
  --turning_point_window_size 8 \
  --selection_diagnostics \
  --adaptive_reg 0.001 \
  --snap_ratio 0.3 \
  --result_prefix turning_point_w8_k64_seed42_full
```

The matched control changes only the selection method:

```bash
python -u -m methods.state_bridge \
  --model Qwen/Qwen3-4B \
  --task medqa \
  --gpus 0 \
  --seed 42 \
  --prompt sequential \
  --temperature 0.6 \
  --max_new_tokens 8192 \
  --max_prefix_tokens 64 \
  --selection_method last_k \
  --adaptive_reg 0.001 \
  --snap_ratio 0.3 \
  --result_prefix last_k_k64_seed42_full
```

Do not compare against a run with different prompts, seed, model revision,
sampling settings, or example order.

## 7. Frozen Upstream Reproduction

The separate control tree supports a one-item check and the published Qwen3-4B
task queue:

```bash
cd StateBridge-repro-qwen3-4b
python -m methods.state_bridge \
  --model Qwen/Qwen3-4B --task medqa --gpus 0 --seed 42 --limit 1

python -u reproduction/run_qwen3_4b_core.py --gpu 0 --seed 42
```

Read `StateBridge-repro-qwen3-4b/reproduction/PROTOCOL_AUDIT.md` before quoting
paper reproduction results.

## 8. Excluded Artifacts

Git does not contain model weights, Hugging Face caches, `.pt` tensors,
checkpoints, full JSON result trees, logs, or virtual environments. Regenerate
them with the commands above or transfer them through external object storage.
Always preserve the command, commit, environment lock, and hardware report with
an authoritative result.
