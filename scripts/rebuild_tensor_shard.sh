#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PART_DIR="$ROOT_DIR/tensors/time_series_full_v1"
OUT_PATH="$PART_DIR/shard_0001.h5"
TMP_PATH="$OUT_PATH.tmp"
CHECKSUM_FILE="$PART_DIR/SHA256SUMS.original"

if [[ ! -f "$CHECKSUM_FILE" ]]; then
  echo "Missing checksum file: $CHECKSUM_FILE" >&2
  exit 1
fi

mapfile -t PARTS < <(find "$PART_DIR" -maxdepth 1 -type f -name 'shard_0001.h5.part-*' | sort)
if [[ "${#PARTS[@]}" -eq 0 ]]; then
  echo "No split tensor parts were found in $PART_DIR" >&2
  exit 1
fi

EXPECTED_SHA="$(awk '$2 == "shard_0001.h5" {print $1}' "$CHECKSUM_FILE")"
if [[ -z "$EXPECTED_SHA" ]]; then
  echo "Could not read the expected SHA-256 checksum from $CHECKSUM_FILE" >&2
  exit 1
fi

if [[ -f "$OUT_PATH" ]]; then
  CURRENT_SHA="$(sha256sum "$OUT_PATH" | awk '{print $1}')"
  if [[ "$CURRENT_SHA" == "$EXPECTED_SHA" ]]; then
    echo "Tensor shard already exists and passed checksum verification:"
    echo "  $OUT_PATH"
    exit 0
  fi
  rm -f "$OUT_PATH"
fi

cat "${PARTS[@]}" > "$TMP_PATH"
ACTUAL_SHA="$(sha256sum "$TMP_PATH" | awk '{print $1}')"

if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  rm -f "$TMP_PATH"
  echo "Checksum mismatch after rebuilding shard_0001.h5" >&2
  echo "Expected: $EXPECTED_SHA" >&2
  echo "Actual:   $ACTUAL_SHA" >&2
  exit 1
fi

mv "$TMP_PATH" "$OUT_PATH"
echo "Rebuilt tensor shard successfully:"
echo "  $OUT_PATH"
