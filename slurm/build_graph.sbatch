#!/bin/bash

#SBATCH --job-name=agri-kg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:15:00
#SBATCH --output=logs/kg-%j.out
#SBATCH --error=logs/kg-%j.err

set -e

module purge
module load cuda/12.9

export TOKENIZERS_PARALLELISM=false

neo4j start

sleep 10

neo4j status

python -m graph.build_kg