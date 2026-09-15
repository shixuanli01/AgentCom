#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/AgentCom}"
ARTIFACTS_REL="${2:-methods/AgentCom-StateBridge/artifacts}"
RELEASE_ID="${3:-icr-artifacts-20260915}"
OUTPUT_DIR="$REPO_ROOT/.release-assets/$RELEASE_ID"
ARTIFACTS_DIR="$REPO_ROOT/$ARTIFACTS_REL"

if [[ ! -d "$ARTIFACTS_DIR" ]]; then
  echo "Artifacts directory does not exist: $ARTIFACTS_DIR" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing release directory: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

manifest="$OUTPUT_DIR/files.sha256"
link_manifest="$OUTPUT_DIR/symlinks.txt"
archive="$OUTPUT_DIR/${RELEASE_ID}.tar.zst"

(
  cd "$REPO_ROOT"
  find "$ARTIFACTS_REL" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$manifest"
(
  cd "$REPO_ROOT"
  find "$ARTIFACTS_REL" -type l -printf '%p -> %l\n' | sort
) > "$link_manifest"

manifest_rel="${manifest#"$REPO_ROOT/"}"
link_manifest_rel="${link_manifest#"$REPO_ROOT/"}"
(
  cd "$REPO_ROOT"
  tar --sort=name --owner=0 --group=0 --numeric-owner \
    -I 'zstd -T0 -10' \
    -cf "$archive" "$ARTIFACTS_REL" "$manifest_rel" "$link_manifest_rel"
)
(
  cd "$OUTPUT_DIR"
  sha256sum "$(basename "$archive")"
) > "$archive.sha256"

echo "archive=$archive"
echo "archive_bytes=$(stat -c %s "$archive")"
echo "file_manifest=$manifest"
echo "archive_sha256=$archive.sha256"
