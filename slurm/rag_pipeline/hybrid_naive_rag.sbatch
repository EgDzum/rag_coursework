#!/bin/bash
#SBATCH --job-name=30-naive-rag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --output=logs/rag-naive-30-%j.out
#SBATCH --error=logs/rag-naive-30-%j.err

set -euo pipefail

module load cuda/12.9

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p results


#   naive  = FAISS only
#   hybrid = FAISS + BM25 + RRF
MODE="${MODE:-naive}"

OUTPUT="${OUTPUT:-results/30_noise/${MODE}_rag.jsonl}"

TOP_K_FAISS="${TOP_K_FAISS:-5}"
TOP_K_BM25="${TOP_K_BM25:-5}"
TOP_K="${TOP_K:-3}"
RRF_K="${RRF_K:-60}"


if [[ "${MODE}" != "naive" && "${MODE}" != "hybrid" ]]; then
    echo "ERROR: MODE must be 'naive' or 'hybrid'."
    echo "Got: ${MODE}"
    exit 1
fi

echo "=========================================="
echo "RAG Retrieval"
echo "=========================================="
echo "SLURM job:       ${SLURM_JOB_ID}"
echo "Compute node:    $(hostname)"
echo "Project:         ${PROJECT_DIR}"
echo "Mode:            ${MODE}"
echo "FAISS top-k:     ${TOP_K_FAISS}"
echo "BM25 top-k:      ${TOP_K_BM25}"
echo "Final top-k:     ${TOP_K}"
echo "RRF k:           ${RRF_K}"
echo "Output:          ${OUTPUT}"
echo "=========================================="
echo ""

CMD=(
    python -m
    rag.hybrid_naive_rag
    --config configs/config.yml
    --mode "${MODE}"
    --output "${OUTPUT}"
    --top-k-faiss "${TOP_K_FAISS}"
    --top-k-bm25 "${TOP_K_BM25}"
    --top-k "${TOP_K}"
    --rrf-k "${RRF_K}"
)

echo "Running:"
printf '  %q ' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

echo ""
echo "=========================================="
echo "RAG retrieval finished"
echo "=========================================="
echo "Mode:    ${MODE}"
echo "End time: $(date)"
echo "Results: ${OUTPUT}"
echo ""