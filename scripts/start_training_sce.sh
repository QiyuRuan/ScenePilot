#!/usr/bin/env bash
# Start N training jobs in parallel with a fixed step, default 4; CARLA and TM ports increase together.
# Examples:
#   bash scripts/start_training_sce.sh -S 6 -r 4 -k 2 -p 2000
#   bash scripts/start_training_sce.sh -S 6 -r 5 -k 3 -p 2004
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"   # Avoid importing a different safebench package.


set -euo pipefail

PYTHON="${PYTHON:-python}"
RUN_PY="$REPO_ROOT/scripts/run.py"

SCENARIO_ID=6    # -S / --scenario
ROUTE_START=4     # -r / --route_start
COUNT=10          # -k / --count
BASE_PORT=3000    # -p / --port  (CARLA base port)
STEP=4            # -s / --step  (Port step)
TM_BASE=9000      # --tm_base    (TM base port)
DEVICE="cuda:0"   # --device     (PyTorch device)
LOG_DIR="./train_logs"

usage() {
  echo "Usage: $0 -S SCENARIO_ID -r ROUTE_START [-k COUNT] [-p BASE_PORT] [-s STEP] [--tm_base TM_BASE] [--device cuda:N]"
  exit 2
}

# Parse arguments
while (( "$#" )); do
  case "$1" in
    -S|--scenario)    SCENARIO_ID="$2"; shift 2 ;;
    -r|--route_start) ROUTE_START="$2"; shift 2 ;;
    -k|--count)       COUNT="$2";       shift 2 ;;
    -p|--port)        BASE_PORT="$2";   shift 2 ;;
    -s|--step)        STEP="$2";        shift 2 ;;
    --tm_base)        TM_BASE="$2";     shift 2 ;;
    --device)         DEVICE="$2";      shift 2 ;;
    --) shift; break ;;
    *)  echo "Unknown argument: $1"; usage ;;
  esac
done

mkdir -p "$LOG_DIR"

for ((i=0;i<COUNT;i++)); do
  PORT=$((BASE_PORT + i*STEP))
  TM_PORT=$((TM_BASE + i*STEP))
  ROUTE_ID=$((ROUTE_START + i))
  TAG="${SCENARIO_ID}-${ROUTE_ID}"
  OUT="$LOG_DIR/train_${TAG}_p${PORT}.out"

  CMD=( "$PYTHON" "$RUN_PY"
        --tag "$TAG"
        --port "$PORT"
        --tm_port "$TM_PORT"
        --scenario_id "$SCENARIO_ID"
        --route_id "$ROUTE_ID"
        --device "$DEVICE" )

  echo ">> Starting training: tag=$TAG  port=$PORT  tm=$TM_PORT  device=$DEVICE  (log: $OUT)"
  nohup "${CMD[@]}" > "$OUT" 2>&1 &
  echo "   PID: $!"
done

echo "== All training jobs started in parallel. Use 'tail -f ${LOG_DIR}/train_*.out' to inspect logs."
