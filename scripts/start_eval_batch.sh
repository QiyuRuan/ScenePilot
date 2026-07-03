#!/usr/bin/env bash
# Run scripts/run_eval.py sequentially for multiple scenario ids.
# Each scenario runs in a fresh Python process to generate distinct log_scenario/log_* and console files.
# This script snapshots the config files at start so later edits (e.g., when launching
# other batches with different policies/ports) will NOT affect runs already queued here.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/start_eval_batch.sh -s "1 2 3" [-a chatscene.yaml] [-c scenepilot.yaml] [-t eval] [-p 2000] [-M 8000] [-- extra_args]

  -s  Space-separated scenario ids to run (required), e.g. "1 2 3"
  -a  Agent config filename under safebench/agent/config (default: chatscene.yaml)
  -c  Scenario config filename under safebench/scenario/config (default: scenepilot.yaml)
  -t  Tag prefix used as <prefix>_<policy_name>_<scenario_id> (default: eval)
  -p  Carla port forwarded to run_eval.py (default: 2000)
  -M  Traffic Manager port forwarded to run_eval.py (default: 8000)
  --  Everything after -- is passed through to scripts/run_eval.py
EOF
}

scenario_ids=(3 4)
# agent_cfg="eval_gen.yaml"
agent_cfg="chatscene.yaml"
# scenario_cfg="scenepilot.yaml"
scenario_cfg="scenepilot.yaml"
tag_prefix="eval"
carla_port="3000"
tm_port="9000"

while getopts ":s:a:c:t:p:M:h" opt; do
  case "${opt}" in
    s) read -ra scenario_ids <<<"${OPTARG}" ;;
    a) agent_cfg="${OPTARG}" ;;
    c) scenario_cfg="${OPTARG}" ;;
    t) tag_prefix="${OPTARG}" ;;
    p) carla_port="${OPTARG}" ;;
    M) tm_port="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Unknown option: -${OPTARG}" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ ${#scenario_ids[@]} -eq 0 ]]; then
  echo "Error: -s \"<ids>\" is required." >&2
  usage
  exit 1
fi

# Extra args forwarded to run_eval.py after "--"
extra_args=("$@")

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

agent_cfg_path="$ROOT_DIR/safebench/agent/config/$agent_cfg"
policy_name="$agent_cfg"
scenario_cfg_path="$ROOT_DIR/safebench/scenario/config/$scenario_cfg"

# Snapshot configs to avoid mid-run edits affecting queued scenarios.
cfg_tmp_dir="$(mktemp -d "$ROOT_DIR/tmp_batch_cfg.XXXX")"
cp "$agent_cfg_path" "$cfg_tmp_dir/" 2>/dev/null || true
cp "$scenario_cfg_path" "$cfg_tmp_dir/" 2>/dev/null || true
agent_cfg_frozen="$(basename "$agent_cfg")"
scenario_cfg_frozen="$(basename "$scenario_cfg")"
agent_cfg_frozen_path="$cfg_tmp_dir/$agent_cfg_frozen"
scenario_cfg_frozen_path="$cfg_tmp_dir/$scenario_cfg_frozen"

if [[ -f "$agent_cfg_frozen_path" ]]; then
  # Best-effort read of policy_name from the YAML file.
  policy_name="$(python - "$agent_cfg_frozen_path" <<'PY' || true
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

path = Path(sys.argv[1])
fallback = path.stem
if yaml is None:
    print(fallback)
    sys.exit(0)

try:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(cfg.get("policy_name", fallback))
except Exception:
    print(fallback)
PY
  )"
fi

run_eval_py="$ROOT_DIR/scripts/run_eval.py"

# Clean up temp configs on exit.
cleanup() {
  rm -rf "$cfg_tmp_dir"
}
trap cleanup EXIT

for sid in "${scenario_ids[@]}"; do
  tag="${tag_prefix}_${policy_name}_${sid}"
  echo "[run_eval] scenario_id=${sid} tag=${tag}"
  cmd=(
    env PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    python "$run_eval_py"
    --agent_cfg "$agent_cfg_frozen_path"
    --scenario_cfg "$scenario_cfg_frozen_path"
    --scenario_id_spy "$sid"
    --port "$carla_port"
    --tm_port "$tm_port"
    --tag "$tag"
  )
  if [[ ${#extra_args[@]} -gt 0 ]]; then cmd+=("${extra_args[@]}"); fi

  (
    cd "$ROOT_DIR"
    "${cmd[@]}"
  )
done
