#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT_DIR/external/upstreams"
mkdir -p "$TARGET"

fetch_at_commit() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local directory="$TARGET/$name"

  if [[ ! -d "$directory/.git" ]]; then
    git clone "$url" "$directory"
  fi
  git -C "$directory" fetch origin "$commit"
  git -C "$directory" checkout --detach "$commit"
}

fetch_at_commit StateBridge https://github.com/YanwenPneg/StateBridge.git \
  3f6bf5442c6e8848555a6132516e6d36f35444fb

echo "The pinned StateBridge checkout is available under $TARGET"
