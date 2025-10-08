#!/usr/bin/env bash
# run_experiments.sh — dispatcher for Atari experiments (MuZero / Stochastic MuZero)
#
# Run this **from the LightZero repo root**.
#
# Layout & assumptions:
#   - this script:                         ./run_experiments.sh
#   - container image (default):           ./container.sif
#   - per-env training configs (python) live under ./zoo/atari/config/
#     * for --env pacman:
#         ./zoo/atari/config/experiment_pacman_muzero_config.py
#         ./zoo/atari/config/experiment_pacman_stochastic_muzero_config.py
#
# What this does:
#   - Interleaves jobs by algo/seed:  muzero_s0, stochastic_s0, muzero_s1, ...
#   - Local mode uses **Apptainer**; SLURM mode uses **Singularity**.
#   - GPU flag is normalized: pass **--nv** or **--nvccli** (required by your site).
#   - Output roots:
#       * Local  ->  ./lz_out/<env_subdir>/<algo>/seed<k>/
#       * SLURM  ->  /scratch/users/$USER/lz_out/<env_subdir>/<algo>/seed<k>/
#     W&B dir is placed at <outroot>/wandb_runs in both modes.
#   - Seeds: defaults to 0..4; you can supply a CSV or a count.
#   - No train-step/iteration knobs are passed here; **configs use their internal defaults**.
#
# Quick examples
# ------------------------------------------------------------------------------
# 0) Local (Apptainer), default container ./container.sif, GPU via --nv,
#    run **both** algos, seeds 0..4, on pacman:
#       bash run_experiments.sh --env pacman --nv
#
# 1) Local with nvccli runtime:
#       bash run_experiments.sh --env pacman --nvccli
#
# 2) Local, only MuZero, specific seeds:
#       bash run_experiments.sh --env pacman --nv --algo muzero --seeds 0,2,4
#
# 3) Local, generate first N seeds (0..N-1), here N=3:
#       bash run_experiments.sh --env pacman --nv --num-seeds 3
#
# 4) Local, custom container path:
#       bash run_experiments.sh --env pacman --nv --sif /abs/path/to/container.sif
#
# 5) SLURM (Singularity), default partitions (gpu,nmes_gpu), request 1 GPU:
#       bash run_experiments.sh --env pacman --use-slurm --nv
#
# 6) SLURM with custom resources:
#       bash run_experiments.sh --env pacman --use-slurm --nv \
#         --partition gpu --gres gpu:1 --mem 48G --cpus 12 --days 1.5
#
# 7) SLURM, only Stochastic MuZero, exclude some nodes:
#       bash run_experiments.sh --env pacman --use-slurm --nv --algo stochastic \
#         --exclude node[01-03]
#
# 8) Override output roots:
#       # Local:
#       bash run_experiments.sh --env pacman --nv --outroot-local /data/lz_out
#       # SLURM:
#       bash run_experiments.sh --env pacman --use-slurm --nv --outroot-slurm /scratch/users/$USER/lz_out
#
# 9) Change SLURM stdout/err directory:
#       bash run_experiments.sh --env pacman --use-slurm --nv --slurm-stdout-dir /scratch/users/$USER/slurm_out
#
# Notes:
#   - The script bind-mounts repo root -> /opendilab/LightZero inside the container.
#   - PYTHONPATH is set so Python imports your live sources.
#   - W&B logging is handled by the Python configs themselves.
#   - Keep using **--nv** or **--nvccli** consistently with your site config.

set -euo pipefail

# ---- Repo-root aware defaults ----
LZ_ROOT="$(pwd)"
SIF_DEFAULT="${LZ_ROOT}/container.sif"

# ---- Defaults ----
USE_SLURM=false
SIF="${SIF_DEFAULT}"
GPU_FLAG="--nv"             # choose via --nv or --nvccli
ALGO="all"                  # all | muzero | stochastic
SEEDS_CSV=""                # e.g. "0,1,2,3,4"
NUM_SEEDS=5                 # used only if SEEDS_CSV empty
ENV_NAME=""                 # required; e.g., pacman

# Paths (to be set by --env)
MUZERO_PY=""
STOCH_PY=""
ENV_SUBDIR=""

# Output roots (updated per your request)
OUTROOT_LOCAL_DEFAULT="${LZ_ROOT}/lz_out"
OUTROOT_SLURM_DEFAULT="/scratch/users/${USER}/lz_out"
OUTROOT_LOCAL="${OUTROOT_LOCAL_DEFAULT}"
OUTROOT_SLURM="${OUTROOT_SLURM_DEFAULT}"

# SLURM defaults (GPU by default)
SLURM_PARTITION="gpu,nmes_gpu"
SLURM_GRES="gpu:1"
SLURM_MEM="32G"
SLURM_CPUS="18"
SLURM_DAYS="2.0"
SLURM_EXCLUDE=""
SLURM_STDOUT_DIR="/scratch/users/${USER}/slurm_out"

# ---- CLI ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)              ENV_NAME="${2:-}"; shift ;;          # e.g., pacman (required)
    --use-slurm)        USE_SLURM=true ;;
    --sif)              SIF="${2:-}"; shift ;;
    --nv)               GPU_FLAG="--nv" ;;
    --nvccli)           GPU_FLAG="--nvccli" ;;
    --algo)             ALGO="${2:-}"; shift ;;              # all|muzero|stochastic
    --seeds)            SEEDS_CSV="${2:-}"; shift ;;         # CSV list
    --num-seeds)        NUM_SEEDS="${2:-}"; shift ;;
    --outroot-local)    OUTROOT_LOCAL="${2:-}"; shift ;;
    --outroot-slurm)    OUTROOT_SLURM="${2:-}"; shift ;;
    --partition)        SLURM_PARTITION="${2:-}"; shift ;;
    --gres)             SLURM_GRES="${2:-}"; shift ;;
    --mem)              SLURM_MEM="${2:-}"; shift ;;
    --cpus)             SLURM_CPUS="${2:-}"; shift ;;
    --days)             SLURM_DAYS="${2:-}"; shift ;;
    --exclude)          SLURM_EXCLUDE="${2:-}"; shift ;;
    --slurm-stdout-dir) SLURM_STDOUT_DIR="${2:-}"; shift ;;
    -h|--help)
      echo "Usage: $0 --env <name> [--nv|--nvccli] [--use-slurm] [--sif PATH] [--algo all|muzero|stochastic] [--seeds CSV|--num-seeds N]"
      exit 0;;
    *) echo "Unknown: $1"; exit 2 ;;
  esac
  shift
done

# Resolve per-env script paths
case "${ENV_NAME}" in
  pacman)
    MUZERO_PY="zoo/atari/config/experiment_pacman_muzero_config.py"
    STOCH_PY="zoo/atari/config/experiment_pacman_stochastic_muzero_config.py"
    ENV_SUBDIR="ms_pacman"
    ;;
  "" )
    echo "Error: --env is required (currently supported: pacman)"; exit 2 ;;
  * )
    echo "Error: --env '${ENV_NAME}' not supported yet (currently: pacman)"; exit 2 ;;
esac

# Sanity
[[ -d "${LZ_ROOT}" ]] || { echo "Error: repo root not found: ${LZ_ROOT}"; exit 2; }
[[ -f "${SIF}" ]] || { echo "Error: SIF not found: ${SIF}"; exit 2; }
[[ -f "${MUZERO_PY}" && -f "${STOCH_PY}" ]] || { echo "Error: config py not found for env '${ENV_NAME}'"; exit 2; }

# Seeds array
declare -a SEEDS=()
if [[ -n "${SEEDS_CSV}" ]]; then
  IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
else
  for ((i=0;i<NUM_SEEDS;i++)); do SEEDS+=("$i"); done
fi
(( ${#SEEDS[@]} > 0 )) || { echo "No seeds parsed."; exit 2; }

# Algo list (for interleaving)
declare -a ALGOS=()
case "${ALGO}" in
  all)        ALGOS=(muzero stochastic) ;;
  muzero)     ALGOS=(muzero) ;;
  stochastic) ALGOS=(stochastic) ;;
  *) echo "Invalid --algo: ${ALGO}"; exit 2 ;;
esac

# Engine + paths
if ${USE_SLURM}; then
  ENGINE="singularity"
  EXP_OUTROOT="${OUTROOT_SLURM}"
else
  ENGINE="apptainer"
  EXP_OUTROOT="${OUTROOT_LOCAL}"
fi
if ${USE_SLURM}; then
  mkdir -p "${EXP_OUTROOT}" "${SLURM_STDOUT_DIR}"
else
  mkdir -p "${EXP_OUTROOT}"
fi

# GPU flag (SLURM: only add if GPU requested; Local: as provided)
if ${USE_SLURM}; then
  if [[ "${SLURM_GRES}" =~ gpu ]] || [[ "${SLURM_PARTITION}" =~ gpu ]]; then
    [[ -n "${GPU_FLAG}" ]] || GPU_FLAG="--nv"
  else
    GPU_FLAG=""
  fi
else
  [[ -n "${GPU_FLAG}" ]] || GPU_FLAG="--nv"
fi

# Container binds (repo + outroot)
CTR_LZ="/opendilab/LightZero"
BINDS="--bind ${LZ_ROOT}:${CTR_LZ},${EXP_OUTROOT}:${EXP_OUTROOT}"

# Build interleaved jobs: muzero0, stochastic0, muzero1, ...
declare -a JOBS_ALGO=()
declare -a JOBS_SEED=()
for s in "${SEEDS[@]}"; do
  for a in "${ALGOS[@]}"; do
    JOBS_ALGO+=("$a"); JOBS_SEED+=("$s")
  done
done

# Helper: print command nicely
print_cmd() { printf "%q " "$@"; }

# Submit or run
if ! ${USE_SLURM}; then
  echo "[Local/${ENGINE}] jobs: ${#JOBS_ALGO[@]}  outroot=${EXP_OUTROOT}"
  for ((i=0;i<${#JOBS_ALGO[@]};i++)); do
    a="${JOBS_ALGO[$i]}"; s="${JOBS_SEED[$i]}"
    outdir="${EXP_OUTROOT%/}/${ENV_SUBDIR}/${a}/seed${s}"
    mkdir -p "${outdir}"

    if [[ "${a}" == "muzero" ]]; then
      PY_REL="${MUZERO_PY}"
    else
      PY_REL="${STOCH_PY}"
    fi

    CMD=( ${ENGINE} exec ${GPU_FLAG:+$GPU_FLAG} --pwd "${CTR_LZ}" ${BINDS} "${SIF}"
          bash -lc "export PYTHONPATH='${CTR_LZ}:\$PYTHONPATH'; \
                    export WANDB_DIR='${EXP_OUTROOT%/}/wandb_runs'; mkdir -p \"\$WANDB_DIR\"; \
                    cd '${CTR_LZ}'; \
                    python -u '${PY_REL}' --seed ${s} --out_dir '${outdir}'" )
    echo "[CMD] $(print_cmd "${CMD[@]}")"
    "${CMD[@]}"
    echo
  done
  echo "[Local] Done."
  exit 0
fi

# ---- SLURM path ----
# walltime from days (supports decimals)
total_min=$(awk -v d="${SLURM_DAYS}" 'BEGIN{printf("%d", d*24*60 + 0.5)}')
d=$(( total_min / (24*60) )); rem=$(( total_min % (24*60) )); h=$(( rem / 60 )); m=$(( rem % 60 ))
SLURM_TIME=$(printf "%d-%02d:%02d:00" "$d" "$h" "$m")

echo "[SLURM] jobs: ${#JOBS_ALGO[@]}  outroot=${EXP_OUTROOT}  stdout=${SLURM_STDOUT_DIR}"
for ((i=0;i<${#JOBS_ALGO[@]};i++)); do
  a="${JOBS_ALGO[$i]}"; s="${JOBS_SEED[$i]}"
  job_name="mz_${ENV_NAME}_${a}_s${s}"
  outdir="${EXP_OUTROOT%/}/${ENV_SUBDIR}/${a}/seed${s}"
  mkdir -p "${outdir}"

  if [[ "${a}" == "muzero" ]]; then
    PY_REL="${MUZERO_PY}"
  else
    PY_REL="${STOCH_PY}"
  fi

  tmp="$(mktemp)"
  {
    echo "#!/bin/bash -l"
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --partition=${SLURM_PARTITION}"
    echo "#SBATCH --cpus-per-task=${SLURM_CPUS}"
    echo "#SBATCH --mem=${SLURM_MEM}"
    echo "#SBATCH --time=${SLURM_TIME}"
    echo "#SBATCH --output=${SLURM_STDOUT_DIR%/}/${job_name}_%j.out"
    echo "#SBATCH --error=${SLURM_STDOUT_DIR%/}/${job_name}_%j.err"
    [[ -n "${SLURM_GRES}" ]]    && echo "#SBATCH --gres=${SLURM_GRES}"
    [[ -n "${SLURM_EXCLUDE}" ]] && echo "#SBATCH --exclude=${SLURM_EXCLUDE}"
    echo "set -euo pipefail"
    echo "module purge >/dev/null 2>&1 || true"
    echo "export SINGULARITY_CACHEDIR='/scratch/users/${USER}/singularity/cache'"
    echo "export SINGULARITY_TMPDIR='/scratch/users/${USER}/\${SLURM_JOB_ID}/tmp'"
    echo "mkdir -p \"\${SINGULARITY_CACHEDIR}\" \"\${SINGULARITY_TMPDIR}\""
    echo "export WANDB_DIR='${EXP_OUTROOT%/}/wandb_runs'"
    echo "mkdir -p \"\$WANDB_DIR\""
    echo "singularity exec ${GPU_FLAG:+$GPU_FLAG} --pwd '${CTR_LZ}' ${BINDS} '${SIF}' bash -lc \"\
export PYTHONPATH='${CTR_LZ}:\$PYTHONPATH'; cd '${CTR_LZ}'; \
python -u '${PY_REL}' --seed ${s} --out_dir '${outdir}'\""
  } > "${tmp}"

  if sb_out=$(sbatch "${tmp}"); then
    echo "Submitted: ${sb_out}"
  else
    echo "Failed to submit ${job_name}" >&2
  fi
  rm -f "${tmp}"
  sleep 0.2
done

echo "[SLURM] All jobs submitted."
