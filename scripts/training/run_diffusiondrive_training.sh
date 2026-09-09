#!/usr/bin/env bash
# DiffusionDrive agent training.
#
# Usage:
#   bash scripts/training/run_diffusiondrive_training.sh              [MODE] [CKPT]
#
#   MODE:
#     scratch  (default) train from scratch
#     resume   seamlessly resume from a ckpt (requires CKPT, explicit or RESUME_CKPT)
#     init     weight-only init from a ckpt (requires CKPT)
#
#   CKPT: checkpoint path (positional $2, or RESUME_CKPT / CKPT env).
#         REQUIRED for resume/init — never auto-discovered.
#
# Examples:
#   bash scripts/training/run_diffusiondrive_training.sh                     # from scratch
#   bash scripts/training/run_diffusiondrive_training.sh scratch             # from scratch
#   bash scripts/training/run_diffusiondrive_training.sh scratch "" 100      # scratch, 100 epochs
#   RESUME_CKPT=/path/last.ckpt bash scripts/training/run_diffusiondrive_training.sh   # resume
#   bash scripts/training/run_diffusiondrive_training.sh resume /path/last.ckpt         # resume
#   bash scripts/training/run_diffusiondrive_training.sh init /path/epoch.ckpt          # init
#
# Resume ALWAYS requires an explicit checkpoint (positional arg or RESUME_CKPT
# env on the SAME command line). It is never auto-discovered, so an accidental
# scratch-vs-resume mixup cannot happen silently.

set -euo pipefail

# Repo root derived from this script's location (scripts/training/ -> repo).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# ---- argument resolution: positional > env vars ----
# MODE:   scratch (default) | resume | init
#         If no positional MODE is given but RESUME_CKPT or CKPT env is set
#         (on the same command line), MODE is inferred automatically:
#           RESUME_CKPT set  -> resume
#           CKPT set         -> init
#         This keeps "RESUME_CKPT=... bash script" working as expected.
MODE="${1:-}"
CKPT="${2:-${RESUME_CKPT:-${CKPT:-}}}"
if [[ -z "${MODE}" ]]; then
    if [[ -n "${RESUME_CKPT:-}" ]]; then
        MODE="resume"
    elif [[ -n "${CKPT:-}" && -z "${2:-}" ]]; then
        MODE="init"
    else
        MODE="${RESUME_MODE:-scratch}"
    fi
fi
if [[ -n "${3:-}" ]]; then MAX_EPOCHS="$3"; else MAX_EPOCHS="${MAX_EPOCHS:-100}"; fi
if [[ -n "${4:-}" ]]; then NUM_WORKERS="$4"; fi

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"

# Auto-size dataloader workers to the machine's CPU cores, leaving 2 cores
# for the main process / system. Floor at 8, cap at 16.
if [[ -z "${NUM_WORKERS:-}" ]]; then
    NPROC="$(nproc)"
    NUM_WORKERS=$(( NPROC - 2 ))
    [[ "${NUM_WORKERS}" -lt 8 ]] && NUM_WORKERS=8
    [[ "${NUM_WORKERS}" -gt 16 ]] && NUM_WORKERS=16
fi

# NAVSIM_EXP_ROOT must be set (see env.sh); checkpoints and cache live under it.
if [[ -z "${NAVSIM_EXP_ROOT:-}" ]]; then
    echo "ERROR: NAVSIM_EXP_ROOT is not set. Source env.sh first." >&2
    exit 1
fi
CACHE_PATH="${NAVSIM_EXP_ROOT}/training_cache/"
EXP_ROOT="${NAVSIM_EXP_ROOT}/training_diffusiondrive_agent"
RESUME_CKPT=""

# ---- validate mode / checkpoint requirements ----
if [[ "${MODE}" != "scratch" && "${MODE}" != "resume" && "${MODE}" != "init" ]]; then
    echo "ERROR: unknown MODE '${MODE}' (expected: scratch | resume | init)" >&2
    exit 1
fi

if [[ "${MODE}" == "resume" || "${MODE}" == "init" ]]; then
    if [[ -z "${CKPT}" ]]; then
        echo "ERROR: MODE '${MODE}' requires an explicit checkpoint path." >&2
        echo "       Either pass it as arg 2 or set RESUME_CKPT on the SAME command line, e.g.:" >&2
        echo "       RESUME_CKPT=/path/last.ckpt bash $0 ${MODE}" >&2
        exit 1
    fi
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: checkpoint file not found: ${CKPT}" >&2
        exit 1
    fi
fi

# ---- build hydra extra args ----
EXTRA_ARGS=()
if [[ "${MODE}" == "init" ]]; then
    EXTRA_ARGS+=(agent.checkpoint_path="${CKPT}")
    echo "MODE: init (weight-only) from ${CKPT}"
elif [[ "${MODE}" == "resume" ]]; then
    RESUME_CKPT="${CKPT}"
    # Quote the whole override: ckpt filenames contain '=' which breaks hydra grammar
    EXTRA_ARGS+=("+trainer.resume_from_checkpoint='${RESUME_CKPT}'")
    echo "MODE: seamless resume from ${RESUME_CKPT}"
    echo "      (model + optimizer + scheduler + epoch counter all restored)"
else
    echo "MODE: scratch (training from scratch)"
fi

echo "NUM_WORKERS=${NUM_WORKERS}  MAX_EPOCHS=${MAX_EPOCHS}"

# ---------------------------------------------------------------------------
# Full log capture: stdout + stderr + Python crash tracebacks (which bypass
# logging and previously only lived in the tmux scrollback, lost on pane close)
# land in one file next to the experiment dirs, named with the launch time.
# HYDRA_FULL_ERROR=1 gives complete (chained) tracebacks instead of the
# abbreviated one. `tee` keeps the tmux/console view intact.
# ---------------------------------------------------------------------------
LOG_DIR="${TRAIN_LOG_DIR:-${EXP_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
export HYDRA_FULL_ERROR=1

echo "FULL LOG: ${LOG_FILE}"

python "${ROOT}/navsim/planning/script/run_training.py" \
    agent=diffusiondrive_agent \
    +agent.config.bkb_path="${ROOT}/checkpoints/pytorch_model.bin" \
    +agent.config.plan_anchor_path="${ROOT}/checkpoints/kmeans_navsim_traj_20.npy" \
    experiment_name=training_diffusiondrive_agent \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    split="${SPLIT}" \
    trainer.params.max_epochs="${MAX_EPOCHS}" \
    cache_path="${CACHE_PATH}" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    dataloader.params.num_workers="${NUM_WORKERS}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"