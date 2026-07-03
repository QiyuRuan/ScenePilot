#!/usr/bin/env bash
# Start arbitrary SCENARIO-ROUTE pairs in parallel; CARLA and TM ports increase together.
# Examples:
#   ./run_train_pairs.sh --pairs "1-5 7-5 6-7 6-8" -p 2000 --tm_base 8000 --step 4
#   ./run_train_pairs.sh --pairs_file pairs.txt -p 2000 --tm_base 8000 --devices "cuda:0,cuda:1"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

PYTHON="${PYTHON:-python}"
RUN_PY="$REPO_ROOT/scripts/run.py"

BASE_PORT=2000      # -p / --port
TM_BASE=8000       # --tm_base
STEP=4               # -s / --step
DEVICES="cuda:0"     # --devices  comma-separated; assigned round-robin
LOG_DIR="./train_logs"

PAIRS="7-5"
PAIRS_FILE=""        # --pairs_file one S-R pair per line

usage() {
  cat <<EOF
Usage:
  $0 [--pairs "S-R S-R ..."] [--pairs_file FILE] [-p BASE_PORT] [--tm_base TM_BASE] [-s STEP] [--devices "cuda:0,cuda:1"]

Examples:
  $0 --pairs "1-5 7-5 6-7 6-8" -p 2000 --tm_base 8000 -s 4 --devices "cuda:0,cuda:1"
  $0 --pairs_file pairs.txt -p 2004 --tm_base 8004 -s 4

Notes:
  - Ports start at BASE_PORT / TM_BASE and increase by STEP for each job.
  - devices are assigned to jobs round-robin.
EOF
  exit 2
}

# Parse arguments
while (( "$#" )); do
  case "$1" in
    --pairs)       PAIRS="$2"; shift 2 ;;
    --pairs_file)  PAIRS_FILE="$2"; shift 2 ;;
    -p|--port)     BASE_PORT="$2"; shift 2 ;;
    --tm_base)     TM_BASE="$2"; shift 2 ;;
    -s|--step)     STEP="$2"; shift 2 ;;
    --devices)     DEVICES="$2"; shift 2 ;;
    -h|--help)     usage ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

# Collect pairs
declare -a PAIR_LIST=()
if [[ -n "$PAIRS" ]]; then
  # shellcheck disable=SC2206
  PAIR_LIST=($PAIRS)   # Split by whitespace
fi

if [[ -n "$PAIRS_FILE" ]]; then
  if [[ ! -f "$PAIRS_FILE" ]]; then
    echo "File not found: $PAIRS_FILE"; exit 1
  fi
  # Append pairs from the file, ignoring empty lines and comments.
  while IFS= read -r line; do
    line="${line%%#*}"         # Remove comments
    line="$(echo "$line" | xargs)" # trim
    [[ -z "$line" ]] && continue
    PAIR_LIST+=("$line")
  done < "$PAIRS_FILE"
fi

if [[ ${#PAIR_LIST[@]} -eq 0 ]]; then
  echo "No pairs specified. Use --pairs or --pairs_file."
  usage
fi

mkdir -p "$LOG_DIR"

# Prepare device array
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
DEV_CNT=${#DEV_ARR[@]}

# Launch jobs
job_idx=0
for pair in "${PAIR_LIST[@]}"; do
  # Parse S-R
  if [[ "$pair" != *"-"* ]]; then
    echo "Invalid pair: '$pair'. Expected S-R format, for example 6-7"; exit 1
  fi
  SCENARIO_ID="${pair%-*}"
  ROUTE_ID="${pair#*-}"

  # Port assignment
  PORT=$((BASE_PORT + job_idx*STEP))
  TM_PORT=$((TM_BASE  + job_idx*STEP))

  # Device round-robin
  DEV="${DEV_ARR[$((job_idx % DEV_CNT))]}"

  TAG="${SCENARIO_ID}-${ROUTE_ID}"
  OUT="$LOG_DIR/train_${TAG}_p${PORT}.out"

  CMD=( "$PYTHON" "$RUN_PY"
        --tag "$TAG"
        --port "$PORT"
        --tm_port "$TM_PORT"
        --scenario_id "$SCENARIO_ID"
        --route_id "$ROUTE_ID"
        --device "$DEV" )

  echo ">> Starting: tag=$TAG  S=$SCENARIO_ID R=$ROUTE_ID  port=$PORT  tm=$TM_PORT  dev=$DEV  (log: $OUT)"
  nohup "${CMD[@]}" > "$OUT" 2>&1 &
  echo "   PID: $!"

  job_idx=$((job_idx+1))
done

echo "== Started ${#PAIR_LIST[@]} jobs in parallel. Use 'tail -f ${LOG_DIR}/train_*.out' to inspect logs."
