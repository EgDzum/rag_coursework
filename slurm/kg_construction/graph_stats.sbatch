#!/bin/bash
#SBATCH --job-name=graph-stats
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/graph-stats-%j.out
#SBATCH --error=logs/graph-stats-%j.err

set -euo pipefail

mkdir -p logs

module load apptainer/1.5.3

NEO4J_HOME="/scratch/ws/emdzyumenko-emdzyu_neo4j_space"
NEO4J_IMAGE="$NEO4J_HOME/containers/neo4j.sif"

NEO4J_DATA="$NEO4J_HOME/data"
NEO4J_LOGS="$NEO4J_HOME/logs"
NEO4J_IMPORT="$NEO4J_HOME/import"
NEO4J_PLUGINS="$NEO4J_HOME/plugins"
NEO4J_CONF="$NEO4J_HOME/conf"
NEO4J_RUN="$NEO4J_HOME/run"

NEO4J_BOLT_PORT=7687

NEO4J_PASSWORD="password"


echo "=========================================="
echo "Graph statistics"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Bolt port:    ${NEO4J_BOLT_PORT}"
echo "=========================================="

mkdir -p \
    "$NEO4J_DATA" \
    "$NEO4J_LOGS" \
    "$NEO4J_IMPORT" \
    "$NEO4J_PLUGINS" \
    "$NEO4J_CONF" \
    "$NEO4J_RUN"

if [[ ! -f "$NEO4J_IMAGE" ]]; then
    echo "ERROR: Neo4j image not found:"
    echo "$NEO4J_IMAGE"
    exit 1
fi

if [[ ! -f "$NEO4J_CONF/neo4j.conf" ]]; then
    echo "ERROR: Neo4j config not found:"
    echo "$NEO4J_CONF/neo4j.conf"
    exit 1
fi

cleanup() {
    echo
    echo "Stopping Neo4j..."

    if [[ -n "${NEO4J_PID:-}" ]] && kill -0 "$NEO4J_PID" 2>/dev/null; then
        kill "$NEO4J_PID" 2>/dev/null || true
        wait "$NEO4J_PID" 2>/dev/null || true
    fi

    echo "Neo4j stopped."
}

trap cleanup EXIT INT TERM

echo
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
    > "$NEO4J_LOGS/graph-stats-neo4j-${SLURM_JOB_ID}.log" 2>&1 &

NEO4J_PID=$!

echo "Neo4j PID: $NEO4J_PID"

echo
echo "Waiting for Neo4j on port ${NEO4J_BOLT_PORT}..."

MAX_WAIT=120
WAITED=0

while ! (echo > /dev/tcp/127.0.0.1/${NEO4J_BOLT_PORT}) 2>/dev/null; do

    if ! kill -0 "$NEO4J_PID" 2>/dev/null; then
        echo "ERROR: Neo4j process exited unexpectedly."
        echo
        echo "Neo4j log:"
        cat "$NEO4J_LOGS/graph-stats-neo4j-${SLURM_JOB_ID}.log"
        exit 1
    fi

    if [[ "$WAITED" -ge "$MAX_WAIT" ]]; then
        echo "ERROR: Neo4j did not start within ${MAX_WAIT} seconds."
        echo
        echo "Neo4j log:"
        cat "$NEO4J_LOGS/graph-stats-neo4j-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 2
    WAITED=$((WAITED + 2))
done

echo "Neo4j Bolt port is available."

echo "=========================================="
echo "GRAPH STATISTICS"
echo "=========================================="

export NEO4J_URI="bolt://localhost:${NEO4J_BOLT_PORT}"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="$NEO4J_PASSWORD"

python -u -m graph.graph_stats

echo "=========================================="
echo "Statistics completed."
echo "=========================================="