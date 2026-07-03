#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

PYTHON="${PYTHON:-python}"
RUN_PY="$REPO_ROOT/scripts/run.py"

AGENT_CFG="chatscene.yaml"
SCENARIO_CFG="ppo.yaml"
DEVICE="cuda:0"
PORT=3000
TM_PORT=9000
MAX_EPISODE_STEP=300
AVSAFE_STEPS_PER_ROUTE=5000
MAX_WAIT_COLLISION_EPISODES=100
ROUTES=(4 5 6 7 8 9 10 11 12 13)
SCENARIOS=(2 5 6 7 8)
EXP_NAME="avsafe_ppo"
LOG_DIR="$REPO_ROOT/train_logs_avsafe_ppo"

USE_WANDB=1
WANDB_PROJECT="ScenePilot-AVSafe"
WANDB_ENTITY=""
WANDB_GROUP="avsafe-ppo-8x50k"
WANDB_MODE="online"

usage() {
  cat <<EOF
Usage:
  $0 [--device cuda:0] [--port 3000] [--tm_port 9000]
     [--avsafe_steps_per_route 5000] [--max_wait_collision_episodes 100] [--agent_cfg autopilot.yaml]
     [--scenario_cfg ppo.yaml] [--exp_name avsafe_ppo]
     [--use_wandb] [--wandb_project NAME] [--wandb_entity ENTITY]
     [--wandb_group NAME] [--wandb_mode online|offline|disabled]

This script runs AV-safe data generation/training sequentially over:
  scenarios: 1..8
  routes:    4..13
  av-safe update steps: 5000 per route
  skip route after 100 collision-free episodes

Total planned av-safe update steps: 8 * 10 * 5000 = 400000
EOF
  exit 2
}

while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tm_port) TM_PORT="$2"; shift 2 ;;
    --avsafe_steps_per_route) AVSAFE_STEPS_PER_ROUTE="$2"; shift 2 ;;
    --max_wait_collision_episodes) MAX_WAIT_COLLISION_EPISODES="$2"; shift 2 ;;
    --agent_cfg) AGENT_CFG="$2"; shift 2 ;;
    --scenario_cfg) SCENARIO_CFG="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --use_wandb) USE_WANDB=1; shift ;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb_group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb_mode) WANDB_MODE="$2"; shift 2 ;;
    -h|--help) usage ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

mkdir -p "$LOG_DIR"

job_idx=0
for sid in "${SCENARIOS[@]}"; do
  for rid in "${ROUTES[@]}"; do
    step_offset=$((job_idx * AVSAFE_STEPS_PER_ROUTE))
    tag="avsafe-ppo-s${sid}-r${rid}"
    out="$LOG_DIR/${tag}.out"

    cmd=(
      "$PYTHON" "$RUN_PY"
      --mode train_scenario
      --agent_cfg "$AGENT_CFG"
      --scenario_cfg "$SCENARIO_CFG"
      --exp_name "$EXP_NAME"
      --tag "$tag"
      --device "$DEVICE"
      --port "$PORT"
      --tm_port "$TM_PORT"
      --max_episode_step "$MAX_EPISODE_STEP"
      --scenario_id "$sid"
      --route_id "$rid"
      --avsafe_training 1
      --train_avsafe_steps "$AVSAFE_STEPS_PER_ROUTE"
      --train_avsafe_step_offset "$step_offset"
      --max_wait_collision_episodes "$MAX_WAIT_COLLISION_EPISODES"
      --route_level_avsafe_training
    )

    if [[ "$USE_WANDB" -eq 1 ]]; then
      cmd+=(--use_wandb)
      cmd+=(--wandb_project "$WANDB_PROJECT")
      cmd+=(--wandb_group "$WANDB_GROUP")
      cmd+=(--wandb_mode "$WANDB_MODE")
      cmd+=(--wandb_name "$tag")
      if [[ -n "$WANDB_ENTITY" ]]; then
        cmd+=(--wandb_entity "$WANDB_ENTITY")
      fi
    fi

    echo ">> [$((job_idx + 1))/80] scenario=$sid route=$rid step_offset=$step_offset"
    echo "   log: $out"
    "${cmd[@]}" 2>&1 | tee "$out"

    job_idx=$((job_idx + 1))
  done
done

echo ">> Finished all ${job_idx} runs. Total planned av-safe update steps: $((job_idx * AVSAFE_STEPS_PER_ROUTE))"
