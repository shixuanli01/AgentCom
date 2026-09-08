#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

required=(
  methods/AgentCom-StateBridge/methods/state_bridge.py
  methods/AgentCom-StateBridge/tests/test_state_selection.py
  StateBridge-repro-qwen3-4b/reproduction/run_qwen3_4b_core.py
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -m compileall -q \
  methods/AgentCom-StateBridge/agentcom \
  methods/AgentCom-StateBridge/methods \
  StateBridge-repro-qwen3-4b/reproduction

(
  cd methods/AgentCom-StateBridge
  "$PYTHON_BIN" -m pytest -q tests
)

(
  cd StateBridge-repro-qwen3-4b
  "$PYTHON_BIN" -m pytest -q reproduction/test_protocol.py
)

echo "Checkout verification passed."
