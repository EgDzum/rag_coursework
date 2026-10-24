#!/bin/bash

#SBATCH --job-name=legal_kg_const
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/kg-%j.out
#SBATCH --error=logs/kg-%j.err

set -euo pipefail

module load cuda/12.9
module load apptainer/1.5.3

export TOKENIZERS_PARALLELISM=false

NEO4J_HOME="/scratch/ws/emdzyumenko-emdzyu_neo4j_space"
NEO4J_IMAGE="$NEO4J_HOME/containers/neo4j.sif"

NEO4J_DATA="$NEO4J_HOME/data"
NEO4J_LOGS="$NEO4J_HOME/logs"
NEO4J_IMPORT="$NEO4J_HOME/import"
NEO4J_PLUGINS="$NEO4J_HOME/plugins"
NEO4J_CONF="$NEO4J_HOME/conf"

echo "=========================================="
echo "SLURM job:      $SLURM_JOB_ID"
echo "Compute node:   $(hostname)"
echo "Neo4j image:    $NEO4J_IMAGE"
echo "Neo4j home:     $NEO4J_HOME"
echo "=========================================="

if [ ! -f "$NEO4J_IMAGE" ]; then
    echo "ERROR: Neo4j image not found:"
    echo "$NEO4J_IMAGE"
    exit 1
fi

if [ ! -f "$NEO4J_CONF/neo4j.conf" ]; then
    echo "ERROR: Neo4j configuration not found:"
    echo "$NEO4J_CONF/neo4j.conf"
    exit 1
fi

echo "Starting Neo4j..."


apptainer exec \
    --bind "$NEO4J_DATA:/var/lib/neo4j/data" \
    --bind "$NEO4J_LOGS:/var/lib/neo4j/logs" \
    --bind "$NEO4J_IMPORT:/var/lib/neo4j/import" \
    --bind "$NEO4J_PLUGINS:/var/lib/neo4j/plugins" \
    --bind "$NEO4J_CONF:/var/lib/neo4j/conf" \
    --bind "$NEO4J_HOME/run:/var/lib/neo4j/run" \
    "$NEO4J_IMAGE" \
    /var/lib/neo4j/bin/neo4j console \
    > "$NEO4J_LOGS/neo4j-$SLURM_JOB_ID.log" 2>&1 &

NEO4J_PID=$!

echo "Neo4j PID: $NEO4J_PID"

cleanup() {
    echo "Stopping Neo4j..."

    if kill -0 "$NEO4J_PID" 2>/dev/null; then
        kill "$NEO4J_PID" || true
        wait "$NEO4J_PID" || true
    fi
}

trap cleanup EXIT INT TERM

echo "Waiting for Neo4j Bolt on localhost:7687..."

for i in $(seq 1 60); do

    if (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1; then
        echo "Neo4j Bolt is available."
        break
    fi

    if ! kill -0 "$NEO4J_PID" 2>/dev/null; then
        echo "ERROR: Neo4j process died."

        echo "===== Neo4j log ====="
        cat "$NEO4J_LOGS/neo4j-$SLURM_JOB_ID.log" || true

        exit 1
    fi

    echo "Waiting... ($i/60)"
    sleep 2

done

if ! (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1; then
    echo "ERROR: Neo4j did not start Bolt within 120 seconds."

    echo "===== Neo4j log ====="
    cat "$NEO4J_LOGS/neo4j-$SLURM_JOB_ID.log" || true

    exit 1
fi

echo "Starting knowledge graph construction..."

python -u -m graph.build_kg --input "$NEO4J_HOME/index_data/processed/legal_chunks.jsonl"

echo "Knowledge graph construction finished."