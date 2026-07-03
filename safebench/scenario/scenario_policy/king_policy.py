import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from safebench.scenario.scenario_policy.base_policy import BasePolicy


class KingPolicy(BasePolicy):
    """Replay pre-generated KING action sequences inside SafeBench scenarios."""

    name = "king"
    type = "unlearnable"

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_scenario = int(scenario_config["num_scenario"])
        self.mode = "eval"
        self.root_dir = scenario_config.get("ROOT_DIR", "")
        self.model_path = scenario_config.get("model_path", "")

        self.agent_index = int(scenario_config.get("king_agent_index", 0))
        self.opt_iter = int(scenario_config.get("king_opt_iter", -1))
        self.action_mode = str(scenario_config.get("king_action_mode", "throttle_minus_brake"))
        self.default_action = float(scenario_config.get("king_default_action", 0.0))

        self._action_sequences: List[List[float]] = [[] for _ in range(self.num_scenario)]
        self._step_counters: List[int] = [0 for _ in range(self.num_scenario)]
        self.continue_episode = 0

    def train(self, replay_buffer):
        return None

    def set_mode(self, mode):
        self.mode = mode

    def get_init_action(self, state, deterministic=False):
        batch = len(state) if state is not None else self.num_scenario
        return [None] * batch, None

    def _resolve_parameter_path(self, parameters: Any, scenario_id: int) -> Optional[str]:
        if not isinstance(parameters, str) or len(parameters) == 0:
            return None

        candidates = []
        if os.path.isabs(parameters):
            candidates.append(parameters)
        else:
            if self.root_dir:
                candidates.append(os.path.join(self.root_dir, parameters))
                if self.model_path:
                    candidates.append(os.path.join(self.root_dir, self.model_path, str(scenario_id), parameters))
            if self.model_path:
                candidates.append(os.path.join(self.model_path, str(scenario_id), parameters))
            candidates.append(parameters)

        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _pick_iter(self, all_iters: List[Any]) -> int:
        if len(all_iters) == 0:
            return -1
        if self.opt_iter < 0:
            return len(all_iters) - 1
        return min(self.opt_iter, len(all_iters) - 1)

    def _select_agent_value(self, value: Any) -> float:
        if isinstance(value, list):
            if len(value) == 0:
                return 0.0
            idx = min(self.agent_index, len(value) - 1)
            return self._select_agent_value(value[idx])
        try:
            return float(value)
        except Exception:
            return 0.0

    def _convert_adv_action(self, step_action: Dict[str, Any]) -> float:
        throttle = self._select_agent_value(step_action.get("throttle", 0.0))
        brake = self._select_agent_value(step_action.get("brake", 0.0))

        if self.action_mode == "throttle":
            return float(np.clip(throttle, 0.0, 1.0))
        if self.action_mode == "negative_brake":
            return float(np.clip(-brake, -1.0, 0.0))

        # default: signed command in [-1, 1]
        return float(np.clip(throttle - brake, -1.0, 1.0))

    def _load_sequence_from_file(self, file_path: str) -> List[float]:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Preferred KING format: scenario_records.json -> adv_actions[opt_iter][timestep]
        if isinstance(payload, dict) and isinstance(payload.get("adv_actions"), list):
            all_iters = payload["adv_actions"]
            iter_idx = self._pick_iter(all_iters)
            if iter_idx < 0:
                return []
            step_list = all_iters[iter_idx]
            if not isinstance(step_list, list):
                return []
            return [self._convert_adv_action(step_action) for step_action in step_list]

        # Optional fallback format: {"actions": [..]}
        if isinstance(payload, dict) and isinstance(payload.get("actions"), list):
            out = []
            for item in payload["actions"]:
                if isinstance(item, dict):
                    out.append(self._convert_adv_action(item))
                else:
                    try:
                        out.append(float(item))
                    except Exception:
                        out.append(self.default_action)
            return out

        return []

    def load_model(self, scenario_configs=None):
        self._action_sequences = [[] for _ in range(self.num_scenario)]
        self._step_counters = [0 for _ in range(self.num_scenario)]

        if scenario_configs is None:
            return

        for idx, cfg in enumerate(scenario_configs):
            if idx >= self.num_scenario:
                break

            path = self._resolve_parameter_path(getattr(cfg, "parameters", None), getattr(cfg, "scenario_id", -1))
            if path is None:
                if self.logger is not None:
                    self.logger.log(
                        f">> KING parameters not found for data_id={getattr(cfg, 'data_id', idx)}; using default action.",
                        color="yellow",
                    )
                continue

            sequence = self._load_sequence_from_file(path)
            self._action_sequences[idx] = sequence

            if self.logger is not None:
                self.logger.log(
                    f">> KING loaded actions ({len(sequence)} steps) from {path}",
                )

    def get_action(self, state, infos, deterministic=False):
        actions = []
        for fallback_idx, info in enumerate(infos):
            sid = int(info.get("scenario_id", fallback_idx))
            if sid < 0 or sid >= self.num_scenario:
                sid = min(max(fallback_idx, 0), self.num_scenario - 1)

            seq = self._action_sequences[sid]
            step = self._step_counters[sid]
            if step < len(seq):
                action_value = seq[step]
            else:
                action_value = self.default_action

            self._step_counters[sid] += 1
            actions.append([float(action_value)])

        return np.asarray(actions, dtype=np.float32)

    def save_model(self, episode):
        return None
