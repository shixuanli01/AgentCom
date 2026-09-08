# Repository Manifest

## Included

- Current AgentCom-StateBridge implementation.
- Turning-point hidden-state selection and diagnostics.
- Multi-path and Refiner causal experiment code that directly precedes the
  current selector experiment.
- Current experiment plans and unit tests.
- Frozen StateBridge Qwen3-4B control and reproduction audit.
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
