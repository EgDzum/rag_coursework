#!/bin/bash
#SBATCH --job-name=graph_noise
#SBATCH --output=logs/graph_noise_%j.out
#SBATCH --error=logs/graph_noise_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

set -euo pipefail

module load apptainer/1.5.3

echo "======================================================================"
echo "Graph Noise Injection"
echo "======================================================================"

echo "Job ID:       ${SLURM_JOB_ID:-unknown}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "Start time:   $(date)"
echo ""

NOISE_PERCENT="${NOISE_PERCENT:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-100}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_NOISE="${ALLOW_EXISTING_NOISE:-0}"

echo "Configuration:"
echo "  Noise percentage:      ${NOISE_PERCENT}%"
echo "  Random seed:           ${SEED}"
echo "  Batch size:            ${BATCH_SIZE}"
echo "  Dry run:               ${DRY_RUN}"
echo "  Allow existing noise:  ${ALLOW_EXISTING_NOISE}"
echo ""

if [[ ! -f "graph/add_graph_noise.py" ]]; then
    echo "ERROR: graph/add_graph_noise.py not found."
    echo "Current directory: $(pwd)"
    exit 1
fi

if [[ ! -f "configs/config.yml" ]]; then
    echo "ERROR: configs/config.yml not found."
    echo "Current directory: $(pwd)"
    exit 1
fi

mkdir -p logs

NEO4J_HOME="/scratch/ws/emdzyumenko-emdzyu_neo4j_space"
NEO4J_IMAGE="$NEO4J_HOME/containers/neo4j.sif"

NEO4J_DATA="$NEO4J_HOME/data"
NEO4J_LOGS="$NEO4J_HOME/logs"
NEO4J_IMPORT="$NEO4J_HOME/import"
NEO4J_PLUGINS="$NEO4J_HOME/plugins"
NEO4J_CONF="$NEO4J_HOME/conf"
NEO4J_RUN="$NEO4J_HOME/run"

if [[ ! -f "$NEO4J_IMAGE" ]]; then
    echo "ERROR: Neo4j Apptainer image not found: $NEO4J_IMAGE"
    exit 1
fi

for dir in \
    "$NEO4J_DATA" \
    "$NEO4J_LOGS" \
    "$NEO4J_IMPORT" \
    "$NEO4J_PLUGINS" \
    "$NEO4J_CONF" \
    "$NEO4J_RUN"
do
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: Required Neo4j directory not found: $dir"
        exit 1
    fi
done

echo "Starting Neo4j..."

apptainer exec \
    --bind "$NEO4J_DATA:/var/lib/neo4j/data" \
    --bind "$NEO4J_LOGS:/var/lib/neo4j/logs" \
    --bind "$NEO4J_IMPORT:/var/lib/neo4j/import" \
    --bind "$NEO4J_PLUGINS:/var/lib/neo4j/plugins" \
    --bind "$NEO4J_CONF:/var/lib/neo4j/conf" \
    --bind "$NEO4J_RUN:/var/lib/neo4j/run" \
    "$NEO4J_IMAGE" \
    /var/lib/neo4j/bin/neo4j console \
    > "$NEO4J_LOGS/neo4j-${SLURM_JOB_ID}.log" 2>&1 &

NEO4J_PID=$!

echo "Neo4j PID: ${NEO4J_PID}"
echo "Neo4j log: $NEO4J_LOGS/neo4j-${SLURM_JOB_ID}.log"
echo ""

cleanup() {
    echo ""
    echo "Stopping Neo4j..."

    if kill -0 "$NEO4J_PID" 2>/dev/null; then
        kill "$NEO4J_PID" 2>/dev/null || true
        wait "$NEO4J_PID" 2>/dev/null || true
        echo "Neo4j stopped."
    else
        echo "Neo4j process is already stopped."
    fi
}

trap cleanup EXIT INT TERM

echo "Waiting for Neo4j Bolt on localhost:7687..."

NEO4J_READY=0

for i in $(seq 1 60); do
    if (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1; then
        echo "Neo4j Bolt is available."
        NEO4J_READY=1
        break
    fi

    if ! kill -0 "$NEO4J_PID" 2>/dev/null; then
        echo "ERROR: Neo4j process died."
        echo "===== Neo4j log ====="
        cat "$NEO4J_LOGS/neo4j-${SLURM_JOB_ID}.log" || true
        exit 1
    fi

    echo "Waiting... ($i/60)"
    sleep 3
done

if [[ "$NEO4J_READY" -ne 1 ]]; then
    echo "ERROR: Neo4j Bolt did not become available in time."
    echo "===== Neo4j log ====="
    cat "$NEO4J_LOGS/neo4j-${SLURM_JOB_ID}.log" || true
    exit 1
fi

CMD=(
    python
    graph/add_graph_noise.py
    --config configs/config.yml
    --noise-percent "$NOISE_PERCENT"
    --seed "$SEED"
    --batch-size "$BATCH_SIZE"
)

if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry-run)
fi

if [[ "$ALLOW_EXISTING_NOISE" == "1" ]]; then
    CMD+=(--allow-existing-noise)
fi

echo ""
echo "Running graph noise injection..."
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

echo "======================================================================"
echo "Graph noise injection finished successfully"
echo "======================================================================"

echo "Job ID:   ${SLURM_JOB_ID}"
echo "End time: $(date)"
echo ""
echo "Neo4j will be stopped by cleanup."