#!/usr/bin/env bash
# Build the Apptainer SIF locally
# Usage: bash build_container.sh

set -euo pipefail

DEF="${DEF:-./container.def}"
SIF="${SIF:-./container.sif}"

command -v apptainer >/dev/null || { echo "ERROR: apptainer not found"; exit 1; }
[[ -f "$DEF" ]] || { echo "ERROR: $DEF not found"; exit 1; }

echo "[BUILD] apptainer build $SIF $DEF"
apptainer build "$SIF" "$DEF"

echo "[TEST] apptainer test $SIF"
apptainer test "$SIF"

echo "[DONE] Built -> $SIF"
