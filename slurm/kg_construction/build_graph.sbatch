#!/bin/bash

#SBATCH --job-name=cs-faiss
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/faiss-%j.out
#SBATCH --error=logs/faiss-%j.err

set -e

module load cuda/12.9

export TOKENIZERS_PARALLELISM=false

python -m graph.build_graph