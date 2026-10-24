#!/bin/bash
#SBATCH --job-name=offline_retrieval_10_naive
#SBATCH --output=logs/retrieval_10_%j.out
#SBATCH --error=logs/retrieval_10_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

mkdir -p logs data

DATASET_PATH="/scratch/ws/emdzyumenko-emdzyu_neo4j_space/dataset/full_dataset.jsonl"
FAISS_PATH="/home/emdzyumenko/coursework/results/10_noise"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"

python evaluate_retrieval.py \
    --dataset $DATASET_PATH \
    --index $FAISS_PATH/index.faiss\
    --metadata $FAISS_PATH/metadata.json \
    --embedding-model $EMBEDDING_MODEL \
    --output $FAISS_PATH/10_retrieval_results.jsonl \
    --question-field input \
    --top-k 8

echo "Retrieval finished at $(date)"