# Repository Manifest

## Included

- Current training-free Trajectory Memory Relay (TMR) implementation.
- TMR last-64 and coverage-64 selectors, diagnostics, evaluator, and tests.
- StateBridge-compatible active-method workspace and frozen Qwen3-4B control.
- Earlier selector, multi-path, and Refiner causal code retained as supporting
  development history.
- Current experiment plans and unit tests.
- Frozen StateBridge Qwen3-4B reproduction audit.
- Small redistributable MedQA and GPQA-Diamond files.
- Remote bootstrap and verification scripts.

## Excluded

- Historical Task 5-14 / RuleMAS code and reports.
- LatentMAS, KVComm, C2C, Single, and TextMAS baseline workspaces.
- Model weights and Hugging Face caches.
- Checkpoints, hidden-state dumps, logs, and full result trees.
- Virtual environments and nested Git metadata.

The original workstation keeps historical code and generated results outside
this focused GitHub repository. Their removal from Git does not delete those
local files.
