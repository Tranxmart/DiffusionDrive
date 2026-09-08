#!/usr/bin/env bash
# DiffusionDrive agent training.
# Usage:
#   bash scripts/training/run_diffusiondrive_training.sh              # train from scratch
#   RESUME=1 bash scripts/training/...                                # auto-resume from newest ckpt
#   RESUME_CKPT=/path/to/last.ckpt bash scripts/training/...          # seamless Lightning resume
#   CKPT=/path/to/epoch.ckpt bash scripts/training/...                # weight-init only
#
# Env overrides (all optional):
#   TRAIN_TEST_SPLIT  train/test split name     (default: navtrain)
#   SPLIT             data_split for logs/blobs (default: trainval)
#   MAX_EPOCHS        max training epochs       (default: 100)
#   NUM_WORKERS       dataloader workers        (default: auto = nproc-2)
#   RESUME            set to 1 to auto-find newest ckpt under the experiment root
#   RESUME_CKPT       Lightning .ckpt for seamless resume (restores optimizer/scheduler/epoch)
#   CKPT              checkpoint for weight-only init via agent.checkpoint_path

set -euo pipefail

ROOT="${NAVSIM_DEVKIT_ROOT:-/home/guorun/e2e/DiffusionDrive}"
cd "${ROOT}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

# Auto-size dataloader workers to the machine's CPU cores, leaving 2 cores
# for the main process / system. Floor at 8, cap at 16. Override via env.
if [[ -z "${NUM_WORKERS:-}" ]]; then
    NPROC="$(nproc)"
    NUM_WORKERS=$(( NPROC - 2 ))
    [[ "${NUM_WORKERS}" -lt 8 ]] && NUM_WORKERS=8
    [[ "${NUM_WORKERS}" -gt 16 ]] && NUM_WORKERS=16
fi

CACHE_PATH="${NAVSIM_EXP_ROOT:-/disk1/e2e/exp}/training_cache/"
CKPT="${CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
RESUME="${RESUME:-}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/disk1/e2e/exp}/training_diffusiondrive_agent"

echo "NUM_WORKERS=${NUM_WORKERS}"

# RESUME=1: auto-discover the newest checkpoint under the experiment root.
if [[ -z "${RESUME_CKPT}" && "${RESUME}" == "1" ]]; then
    RESUME_CKPT="$(find "${EXP_ROOT}" -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)"
    if [[ -z "${RESUME_CKPT}" ]]; then
        echo "ERROR: RESUME=1 but no *.ckpt found under ${EXP_ROOT}" >&2
        exit 1
    fi
    echo "AUTO-RESUME: using newest ckpt ${RESUME_CKPT}"
fi

if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        echo "ERROR: RESUME_CKPT file not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    echo "RESUMING from: ${RESUME_CKPT}"
elif [[ -z "${CKPT}" ]]; then
    echo "MODE: training from scratch (no RESUME_CKPT / CKPT given)"
fi

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
    EXTRA_ARGS+=(agent.checkpoint_path="${CKPT}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
    # Quote the whole override: ckpt filenames contain '=' which breaks hydra grammar
    EXTRA_ARGS+=("+trainer.resume_from_checkpoint='${RESUME_CKPT}'")
fi

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
    "${EXTRA_ARGS[@]}"