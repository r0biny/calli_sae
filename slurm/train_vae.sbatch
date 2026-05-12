#!/bin/bash
#SBATCH --job-name=calli_vae
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ruohan.yu/Code/calli_sae_results/slurm_out/%x-%j.out
#SBATCH --error=/home/ruohan.yu/Code/calli_sae_results/slurm_out/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Code/calli_sae}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/Code/calligraphy_project}"
RES_REPO="${RES_REPO:-$HOME/Code/calli_sae_results}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/vae_finetune.yaml}"
VAE_MODEL="${VAE_MODEL:-$HOME/Code/pretrained/sd-vae-ft-mse}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

EPOCHS="${EPOCHS:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${RES_REPO}" "${RES_REPO}/slurm_out"

source /apps/local/conda_init.sh
conda activate tf

cd "${PROJECT_ROOT}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

python -u scripts/train_vae.py \
  --config "${CONFIG}" \
  --dataset_root "${DATASET_ROOT}" \
  --train_csv "${DATASET_ROOT}/dataset/train_260402.csv" \
  --val_csv "${DATASET_ROOT}/dataset/validation.csv" \
  --image_column path \
  --pretrained_model_name_or_path "${VAE_MODEL}" \
  --image_size "${IMAGE_SIZE}" \
  --epochs "${EPOCHS}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --num_workers "${NUM_WORKERS}" \
  --output_root "${RES_REPO}" \
  "${RESUME_ARGS[@]}"
