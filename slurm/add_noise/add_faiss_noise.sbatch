#!/bin/bash
#SBATCH --job-name=50-noise_faiss
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=logs/noise_faiss-%j.out
#SBATCH --error=logs/noise_faiss-%j.err

set -euo pipefail

mkdir -p logs

NOISE_TYPES="${NOISE_TYPES:-delete word_swap char_noise duplicate truncate}"
DELETE_RATIO="${DELETE_RATIO:-0.1}"
DUPLICATE_RATIO="${DUPLICATE_RATIO:-0.2}"
CHAR_NOISE_RATIO="${CHAR_NOISE_RATIO:-0.1}"
SWAP_RATIO="${SWAP_RATIO:-0.1}"
TRUNCATE_RATIO="${TRUNCATE_RATIO:-0.8}"
APPLY="${APPLY:-0.5}"
OUT_CHUNKS="${OUT_CHUNKS:-results/50_noise/50_chunks_noisy.jsonl}"
OUT_INDEX="${OUT_INDEX:-results/50_noise}"
SEED="${SEED:-42}"
SKIP_INDEX="${SKIP_INDEX:-0}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.9

EXTRA_ARGS=()
if [[ "${SKIP_INDEX}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-index)
fi

python -m graph.add_faiss_noise \
  --config configs/config.yml \
  --out-chunks "${OUT_CHUNKS}" \
  --out-index "${OUT_INDEX}" \
  --noise-types ${NOISE_TYPES} \
  --delete-ratio "${DELETE_RATIO}" \
  --seed "${SEED}" \
  --duplicate-ratio "${DUPLICATE_RATIO}" \
  --char-noise-ratio "${CHAR_NOISE_RATIO}" \
  --word-swap-ratio "${SWAP_RATIO}" \
  --truncate-keep-ratio "${TRUNCATE_RATIO}" \
  --apply-prob "${APPLY}" \
  --chunks "data/processed/chunks.jsonl"