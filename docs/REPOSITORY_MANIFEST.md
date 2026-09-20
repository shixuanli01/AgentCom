# Repository Manifest

## Included

- ICR-V3 baseline pipeline: protocol, channels, runtime, repair, and analysis.
- EGR evidence-adjudication track.
- StateBridge-compatible active-method workspace and frozen Qwen3-4B control.
- Current experiment plans and unit tests.
- Frozen StateBridge Qwen3-4B reproduction audit.
- Small redistributable MedQA and GPQA-Diamond files.
- Remote bootstrap and verification scripts.

## Excluded

- Historical Task 5-14 / RuleMAS code and reports.
- Trajectory Memory Relay, the multi-path evaluator, the LLM judge, and the
  Refiner causal runner: superseded tracks, unreachable from the current
  pipeline, kept in Git history rather than the working tree.
- LatentMAS, KVComm, C2C, Single, and TextMAS baseline workspaces.
- Model weights and Hugging Face caches.
- Checkpoints, hidden-state dumps, logs, and full result trees.
- Virtual environments and nested Git metadata.

The original workstation keeps historical code and generated results outside
this focused GitHub repository. Their removal from Git does not delete those
local files.
