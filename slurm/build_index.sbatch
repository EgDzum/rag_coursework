#!/bin/bash

#SBATCH --job-name=agri-faiss
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=logs/faiss-%j.out
#SBATCH --error=logs/faiss-%j.err

set -e

module load cuda/12.9

export TOKENIZERS_PARALLELISM=false

python -m retrieval.build_index