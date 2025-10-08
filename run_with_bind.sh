#!/usr/bin/env bash
# run_with_bind.sh
# Minimal local runner: bind host LightZero repo into container and run CMD.
# Usage:
#   bash run_with_bind.sh /abs/path/LightZero ./container.sif [--nv|--nvccli] "python -u zoo/atari/config/ms_pacman_v5_muzero.py --seed 0"

set -euo pipefail

HOST_LZ="${1:?Usage: run_with_bind.sh /path/LightZero ./container.sif [--nv|--nvccli] \"CMD...\"}"
SIF="${2:?Usage: run_with_bind.sh /path/LightZero ./container.sif [--nv|--nvccli] \"CMD...\"}"
GPU_OPT="${3:-}"     # --nv | --nvccli | empty
CMD="${4:-}"

# Normalize GPU flag
GPU_FLAG=""
case "${GPU_OPT}" in
  --nvccli|nvccli) GPU_FLAG="--nvccli" ;;
  --nv|nv)         GPU_FLAG="--nv" ;;
  "" )             GPU_FLAG="" ;;
  * )              GPU_FLAG="${GPU_OPT}" ;;  # passthrough
esac

CTR_LZ="/opendilab/LightZero"
[[ -d "$HOST_LZ" ]] || { echo "ERROR: host repo not found: $HOST_LZ"; exit 1; }
[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
[[ -n "$CMD" ]] || { echo "ERROR: missing command to run"; exit 1; }

echo "[RUN] bind $HOST_LZ -> $CTR_LZ"
echo "[RUN] cmd : $CMD"

# Bind repo; rely on container's LZ_HOME=/opendilab/LightZero
apptainer exec ${GPU_FLAG:+$GPU_FLAG} --bind "$HOST_LZ":"$CTR_LZ" "$SIF" \
  bash -lc 'export PYTHONPATH="$LZ_HOME:$PYTHONPATH"; cd "$LZ_HOME"; '"$CMD"
