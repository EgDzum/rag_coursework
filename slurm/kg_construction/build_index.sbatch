#!/bin/bash
#SBATCH --job-name=legal_indexing
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --error=logs/index_err-%j.txt
#SBATCH --output=logs/index_out-%j.txt
#SBATCH --nodes=1
#SBATCH --time=00:50:00

export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

# Set project root
PROJECT_ROOT=/home/emdzyumenko/coursework

# Set CUDA path
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Change to project directory
cd $PROJECT_ROOT || { echo "Failed to cd to $PROJECT_ROOT"; exit 1; }

# Add to Python path
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "=== NVIDIA-SMI ==="
nvidia-smi
echo "=================="

python -m retrieval.build_index --mode append --data-path "/scratch/ws/emdzyumenko-emdzyu_neo4j_space/index_data/processed/cs_chunks.jsonl"