#!/usr/bin/env bash
# Start N CARLA instances with ports base_port, base_port+step, base_port+2*step, ...
# Examples:
#   ./start_carla_step.sh -n 1                 # start 1 instance: 2000
#   ./start_carla_step.sh -n 3                 # start 3 instances: 2000, 2004, 2008
#   ./start_carla_step.sh -b 2012 -n 2         # 2012,2016
#   ./start_carla_step.sh -C /path/to/CARLA    # set CARLA directory
#   ./start_carla_step.sh -n 2 -x              # wrap with xvfb-run if needed

set -euo pipefail

N=2
BASE_PORT=3000
STEP=4
# Prefer CARLA_DIR, falling back to legacy CARLA_ROOT.
CARLA_DIR="${CARLA_DIR:-${CARLA_ROOT:-}}"
LOG_DIR="./carla_logs"
USE_XVFB=0
SCREEN_SPEC="-screen 0 1024x768x24"   # Used only with -x.

usage() {
  echo "Usage: $0 [-n NUM] [-b BASE_PORT] [-s STEP] [-C CARLA_DIR] [-l LOG_DIR] [-x]"
  echo "  -n NUM         Number of instances (default: $N)"
  echo "  -b BASE_PORT   Base port (default: $BASE_PORT)"
  echo "  -s STEP        Port step (default: $STEP)"
  echo "  -C CARLA_DIR   CARLA root directory containing CarlaUE4.sh"
  echo "  -l LOG_DIR     Log directory (default: $LOG_DIR)"
  echo "  -x             Wrap with xvfb-run (disabled by default)"
  exit 2
}

while getopts ":n:b:s:C:l:x" opt; do
  case "$opt" in
    n) N="$OPTARG" ;;
    b) BASE_PORT="$OPTARG" ;;
    s) STEP="$OPTARG" ;;
    C) CARLA_DIR="$OPTARG" ;;
    l) LOG_DIR="$OPTARG" ;;
    x) USE_XVFB=1 ;;
    *) usage ;;
  esac
done

[[ -n "${CARLA_DIR}" ]] || { echo "Specify CARLA directory with -C, or export CARLA_DIR/CARLA_ROOT first."; exit 1; }

CARLA_BIN="$CARLA_DIR/CarlaUE4.sh"
[[ -x "$CARLA_BIN" ]] || { echo "Executable not found: $CARLA_BIN"; exit 1; }

mkdir -p "$LOG_DIR"

for ((i=0;i<N;i++)); do
  PORT=$((BASE_PORT + i*STEP))
  LOG_FILE="$LOG_DIR/carla_${PORT}.out"
  echo ">> Starting CARLA: port=$PORT  (log: $LOG_FILE)"

  if [[ "$USE_XVFB" -eq 1 ]]; then
    # Use when a virtual display is required.
    nohup xvfb-run --auto-servernum --server-args="$SCREEN_SPEC" \
      "$CARLA_BIN" -prefernvidia -RenderOffScreen -carla-port="$PORT" \
      > "$LOG_FILE" 2>&1 &
  else
    # Use direct headless rendering.
    nohup "$CARLA_BIN" -prefernvidia -RenderOffScreen -carla-port="$PORT" \
      > "$LOG_FILE" 2>&1 &
  fi

  echo "   PID: $!"
done

echo "== Done. Suggested TM ports: 8000, $((8000+STEP)), $((8000+2*STEP)) ..."
