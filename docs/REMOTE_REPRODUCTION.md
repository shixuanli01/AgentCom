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

### ICR and EGR cross-benchmark smoke

The current EGR V1 method is frozen in
`methods/AgentCom-StateBridge/experiments/EGR_V1_METHOD_FREEZE_ZH.md`. It reuses
the ICR prebelief cache and must not be interpreted as a pure latent-channel
comparison.

```bash
cd methods/AgentCom-StateBridge
PYTHONPATH=. WORKERS_PER_GPU=2 BENCHMARK_LIMIT=2 \
  bash scripts/run_cross_benchmark_first_wave.sh arc_challenge \
  artifacts/cross_benchmark/smoke_arc2
```

Inspect `cross_benchmark_analysis/report.{md,json}` before starting 300 items.
For a fixed random sample:

```bash
PYTHONPATH=. WORKERS_PER_GPU=2 BENCHMARK_SAMPLE_SIZE=300 \
  BENCHMARK_SELECTION_SEED=42 \
  bash scripts/run_cross_benchmark_first_wave.sh gsm8k
```

The script runs prebelief generation, no-message/Text/StateBridge/LatentMAS
revisions, deterministic evidence filtering, EGR Contrast, the permutation gate
when valid for the task, and paired analysis. MBPP+/HumanEval+ intentionally
skip permutation because source-string equality is not behavioral equivalence.

### Trajectory Memory Relay

TMR is the active method. It requires no training and adds no trainable
parameters. Run both transport variants with one item before starting a full
queue:

```bash
cd methods/AgentCom-StateBridge

PYTHONPATH=. python -m agentcom.tmr_eval \
  --communication-method tmr \
  --tmr-selection last64 \
  --run-dir artifacts/tmr_v1/smoke_last64_seed42 \
  --limit 1

PYTHONPATH=. python -m agentcom.tmr_eval \
  --communication-method tmr \
  --tmr-selection coverage64 \
  --coverage-scope full \
  --run-dir artifacts/tmr_v1/smoke_coverage64_seed42 \
  --limit 1
```

Inspect `summary.json`, `diagnostics.jsonl`, and the item record. Every handoff
must report three `[1,64,2560]` memories for Qwen3-4B and startup must print
`num_new_trainable_parameters=0`.

### StateBridge controls

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

### TMR queue

Run `tmr_last64` first and `tmr_coverage64` second:

```bash
cd methods/AgentCom-StateBridge
mkdir -p artifacts/tmr_v1
tmux new-session -d -s tmr-medqa \
  'bash scripts/run_tmr_medqa300.sh artifacts/tmr_v1 \
   > artifacts/tmr_v1/server_queue.log 2>&1'

tail -f artifacts/tmr_v1/server_queue.log
```

On a four-GPU host, the item-stable role seeds allow both variants to run as
four shards each without changing generation inputs. Two Qwen3-4B workers share
each 32 GB GPU to use otherwise idle compute capacity:

```bash
cd methods/AgentCom-StateBridge
bash scripts/run_tmr_medqa300_4gpu.sh artifacts/tmr_v1
```

The eight workers share one run directory per variant and write distinct
atomic item records. A run is marked complete only after all 300 expected
records are present.

The queue writes one atomic record per item, so the same command resumes after
an interruption. Generated traces and hidden-state diagnostics remain under
ignored `artifacts/` and must not be committed.

The TMR run does not require an existing baseline result. To generate paired
StateBridge metrics during the run, provide a previously produced multipath
summary:

```bash
export BASELINE=/path/to/medqa_qwen3_4b_m5_seed42/summary.json
bash scripts/run_tmr_medqa300.sh artifacts/tmr_v1
```

Without `BASELINE`, method accuracy, latency, entropy, gate, and norm metrics
are still recorded; paired fields are null. The same per-item/per-role seed is
used regardless of sharding or resume order.

### Earlier selection experiment

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
