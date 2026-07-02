#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

PYTHON="${PYTHON:-python}"
RUN_PY="$REPO_ROOT/scripts/run_train_av.py"

LOG_DIR="$REPO_ROOT/train_logs_av"

mkdir -p "$LOG_DIR"

i=1
for sid in 2 5 6 7 8; do
  PORT=$((2000 + i*4))
  TM=$((8000 + i*4))
  TAG="av-${sid}"

  nohup "$PYTHON" "$RUN_PY" \
    --tag "$TAG" \
    --mode train_agent \
    --agent_cfg chatscene.yaml \
    --scenario_cfg scenepilot.yaml \
    --scenario_id "$sid" \
    --port "$PORT" --tm_port "$TM" \
    > "$LOG_DIR/${TAG}_p${PORT}.out" 2>&1 &

  echo "PID $!  TAG=$TAG  port=$PORT  tm=$TM"
  i=$((i+1))
done
