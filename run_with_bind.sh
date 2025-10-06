#!/usr/bin/env bash
# Run a command inside the container with host repo bind-mounted at /opendilab/LightZero.
# The image already has `pip install -e .` pointing to /opendilab/LightZero,
# so binding to the same path makes Python import your live sources immediately.
#
# Usage:
#   bash run_with_bind.sh /abs/path/to/LightZero ./container.sif --nv "python -u zoo/atari/config/experiment_pong_muzero_config.py"

set -euo pipefail

HOST_LZ="${1:?Usage: run_with_bind.sh /abs/path/to/LightZero ./container.sif [--nv] \"CMD...\"}"
SIF="${2:?Usage: run_with_bind.sh /abs/path/to/LightZero ./container.sif [--nv] \"CMD...\"}"
NV_FLAG="${3:-}"
CMD="${4:-}"

CTR_LZ="/opendilab/LightZero"
[[ -d "$HOST_LZ" ]] || { echo "ERROR: host repo not found: $HOST_LZ"; exit 1; }
[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[[ -n "$CMD" ]] || { echo "ERROR: missing command to run"; exit 1; }

echo "[RUN] bind $HOST_LZ -> $CTR_LZ"
echo "[RUN] cmd : $CMD"
apptainer exec $NV_FLAG --bind "$HOST_LZ":"$CTR_LZ" "$SIF" \
  bash -lc 'cd "$LZ_HOME" && python -m pip install -e . --no-build-isolation && '"$CMD"
