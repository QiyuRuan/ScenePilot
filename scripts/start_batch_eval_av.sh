#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_DIR="${1:-$ROOT_DIR/safebench/agent/model_ckpt/sac_chatscene/finetune}"
AGENT_CFG="${2:-chatscene.yaml}"
SCENARIO_CFG="${3:-scenepilot.yaml}"

PARALLEL="${PARALLEL:-2}"
NUM_SCENARIO="${NUM_SCENARIO:-1}"
BASE_PORT="${BASE_PORT:-3000}"
BASE_TM_PORT="${BASE_TM_PORT:-9000}"
PORT_STEP="${PORT_STEP:-4}"
SCENARIO_IDS="${SCENARIO_IDS:-"7 8"}"

jobs=()
idx=0
if [[ -n "$SCENARIO_IDS" ]]; then
  for sid in $SCENARIO_IDS; do
    jobs+=("${sid}:${idx}")
    idx=$((idx + 1))
  done
else
  for dir in "$FINETUNE_DIR"/*; do
    [[ -d "$dir" ]] || continue
    sid="$(basename "$dir")"
    jobs+=("${sid}:${idx}")
    idx=$((idx + 1))
  done
fi

if [[ "${#jobs[@]}" -eq 0 ]]; then
  echo "No scenario_id directories found under $FINETUNE_DIR" >&2
  exit 1
fi

printf "%s\n" "${jobs[@]}" | xargs -I{} -P "$PARALLEL" bash -c '
  set -euo pipefail
  IFS=":" read -r sid idx <<< "$1"
  root_dir="$2"
  finetune_dir="$3"
  agent_cfg="$4"
  scenario_cfg="$5"
  num_scenario="$6"
  base_port="$7"
  base_tm_port="$8"
  port_step="$9"

  port=$((base_port + idx * port_step))
  tm_port=$((base_tm_port + idx * port_step))

  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/eval_av_XXXXXX")"
  trap "rm -rf \"$tmp_dir\"" EXIT
  if [[ "$agent_cfg" = /* ]]; then
    cp "$agent_cfg" "$tmp_dir/agent_cfg.yaml"
  else
    cp "$root_dir/safebench/agent/config/$agent_cfg" "$tmp_dir/agent_cfg.yaml"
  fi
  if [[ "$scenario_cfg" = /* ]]; then
    cp "$scenario_cfg" "$tmp_dir/scenario_cfg.yaml"
  else
    cp "$root_dir/safebench/scenario/config/$scenario_cfg" "$tmp_dir/scenario_cfg.yaml"
  fi
  agent_cfg="$tmp_dir/agent_cfg.yaml"
  scenario_cfg="$tmp_dir/scenario_cfg.yaml"

  ckpt_dir="$finetune_dir/$sid"
  mapfile -t ckpts < <(ls "$ckpt_dir"/model.sac.*.torch 2>/dev/null | sort)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoints found in $ckpt_dir" >&2
    exit 1
  fi

  for ckpt in "${ckpts[@]}"; do
    base="$(basename "$ckpt")"
    if [[ "$base" =~ ^model\.sac\.(-?[0-9]+)\.torch$ ]]; then
      ep="${BASH_REMATCH[1]}"
      if [[ "$ep" == -* ]]; then
        ep="-$((10#${ep#-}))"
      else
        ep="$((10#$ep))"
      fi
    else
      echo "Skipping unrecognized checkpoint name: $base" >&2
      continue
    fi
    tag="eval_av_sid${sid}_${ep}"
    PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
      python "$root_dir/scripts/run_eval_av.py" \
      --tag "$tag" \
      --agent_cfg "$agent_cfg" \
      --scenario_cfg "$scenario_cfg" \
      --scenario_id_spy "$sid" \
      --num_scenario "$num_scenario" \
      --load_dir "$finetune_dir/$sid" \
      --load_iteration "$ep" \
      --port "$port" \
      --tm_port "$tm_port"
  done
' _ {} "$ROOT_DIR" "$FINETUNE_DIR" "$AGENT_CFG" "$SCENARIO_CFG" "$NUM_SCENARIO" "$BASE_PORT" "$BASE_TM_PORT" "$PORT_STEP"
