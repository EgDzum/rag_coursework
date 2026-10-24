#!/bin/bash
#SBATCH --job-name=graph_retrieval_eval
#SBATCH --output=logs/graph_retrieval_%j.out
#SBATCH --error=logs/graph_retrieval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

module load cuda/12.9
module load apptainer/1.5.3

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs evaluation

echo "=========================================="
echo "Graph Retrieval Evaluation"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "Start time: $(date)"

DATASET="/scratch/ws/emdzyumenko-emdzyu_neo4j_space/dataset/full_dataset.jsonl"

FAISS_INDEX="/scratch/ws/emdzyumenko-emdzyu_neo4j_space/indices/faiss/index.faiss"
FAISS_METADATA="/scratch/ws/emdzyumenko-emdzyu_neo4j_space/indices/faiss/metadata.json"

EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"

OUTPUT="data/graph_retrieval.jsonl"

TOP_K=8
GRAPH_LIMIT=20
MAX_HOPS=2

MAX_ITEMS=20

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

python export_graph_retrieval.py \
    --dataset "$DATASET" \
    --index "$FAISS_INDEX" \
    --metadata "$FAISS_METADATA" \
    --embedding-model "$EMBEDDING_MODEL" \
    --output "$OUTPUT" \
    --top-k "$TOP_K" \
    --graph-limit "$GRAPH_LIMIT" \
    --max-hops "$MAX_HOPS" \
    --max-items "$MAX_ITEMS"

echo "=========================================="
echo "Graph retrieval completed"
echo "Output: $OUTPUT"
echo "End time: $(date)"
echo "=========================================="
