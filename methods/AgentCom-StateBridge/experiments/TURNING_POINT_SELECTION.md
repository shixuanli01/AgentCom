# Turning-Point Hidden-State Selection

## Question

Does selecting semantically changing positions from the sender trajectory
perform better than always transmitting the last 64 post-thinking states?

## Intervention

For each eligible sender sequence `h[0:T]` and position `t`, compute the mean
hidden state over up to eight preceding positions and up to eight following
positions. The score is:

```text
score(t) = 1 - cosine(left_mean(t), right_mean(t))
```

Select the 64 highest-scoring positions, then sort their indices so the
receiver sees them in original chronological order. Hidden states and decoded
token IDs are indexed together so StateBridge alignment still receives matched
pairs.

Boundary positions use the available context. If `T <= 64`, every position is
kept. Scores are calculated in float32.

## Controlled Variables

| Variable | Value |
|---|---|
| Model | `Qwen/Qwen3-4B` |
| Dataset | bundled MedQA, 300 rows |
| Seed | `42` |
| Prompt | `sequential` |
| Temperature | `0.6` |
| Top-p | upstream default `0.95` |
| Message budget | `K=64` |
| Alignment regularization | `0.001` |
| Vocabulary anchoring | `0.3` |
| Turning-point window | `8` |

The control and intervention must differ only in `selection_method` and
turning-point diagnostics.

## Commands

Intervention:

```bash
python -u -m methods.state_bridge \
  --model Qwen/Qwen3-4B --task medqa --gpus 0 --seed 42 \
  --prompt sequential --temperature 0.6 --max_new_tokens 8192 \
  --max_prefix_tokens 64 --selection_method turning_point \
  --turning_point_window_size 8 --selection_diagnostics \
  --adaptive_reg 0.001 --snap_ratio 0.3 \
  --result_prefix turning_point_w8_k64_seed42_full
```

Control:

```bash
python -u -m methods.state_bridge \
  --model Qwen/Qwen3-4B --task medqa --gpus 0 --seed 42 \
  --prompt sequential --temperature 0.6 --max_new_tokens 8192 \
  --max_prefix_tokens 64 --selection_method last_k \
  --adaptive_reg 0.001 --snap_ratio 0.3 \
  --result_prefix last_k_k64_seed42_full
```

## Evaluation

Report exact accuracy counts for both conditions and a paired transition table:

- both correct;
- control only correct;
- turning-point only correct;
- both wrong.

Also report parse failures, generation/token costs, selected-position
distribution, hop-specific overlap with last-64, and paired significance. A
gain does not by itself prove better communication; interpretation must remain
limited to this fixed-budget selection intervention.

Generated logs, result JSON, and per-sample selection diagnostics are ignored
by Git and must be stored externally with the commit hash and environment.
