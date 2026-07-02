import os
from collections import deque

import numpy as np
import torch
from PIL import Image

from safebench.agent.base_policy import BasePolicy
from safebench.agent.transfuser_lib import GlobalConfig, TransFuser
from safebench.agent.transfuser_lib.utils import (
    lidar_to_histogram_features,
    scale_and_crop_image,
    transform_2d_points,
)
from safebench.king.driving_agents.king.common.utils.planner import RoutePlanner as KingRoutePlanner


class _DummyCmd:
    value = 4


class TransfuserAgent(BasePolicy):
    name = "transfuser"
    type = "unlearnable"

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mode = "eval"
        self.continue_episode = 0
        self.ego_vehicles = []

        root_dir = config.get("ROOT_DIR", "")
        configured_ckpt_dir = config.get(
            "transfuser_ckpt_dir",
            "safebench/agent/model_ckpt/transfuser/regular/transfuser",
        )
        ckpt_candidates = []
        if os.path.isabs(configured_ckpt_dir):
            ckpt_candidates.append(configured_ckpt_dir)
        else:
            ckpt_candidates.append(os.path.join(root_dir, configured_ckpt_dir) if root_dir else configured_ckpt_dir)

        # Allow fallback to the vendored KING weights bundled inside SafeBench.
        king_default = os.path.join(
            root_dir,
            "safebench/king/driving_agents/king/transfuser/model_checkpoints/regular/transfuser",
        ) if root_dir else "safebench/king/driving_agents/king/transfuser/model_checkpoints/regular/transfuser"
        if king_default not in ckpt_candidates:
            ckpt_candidates.append(king_default)

        self.ckpt_dir = ckpt_candidates[0]
        for candidate in ckpt_candidates:
            if os.path.isfile(os.path.join(candidate, "best_model.pth")):
                self.ckpt_dir = candidate
                break
        self.weights_path = os.path.join(self.ckpt_dir, "best_model.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = GlobalConfig()

        self.model_config.input_resolution = int(
            config.get("transfuser_input_resolution", self.model_config.input_resolution)
        )
        self.model_config.seq_len = int(config.get("transfuser_seq_len", self.model_config.seq_len))
        self.model_config.pred_len = int(config.get("transfuser_pred_len", self.model_config.pred_len))

        # Match KING inference setup.
        self.model_config.ignore_sides = True
        self.model_config.ignore_rear = True
        self.model_config.n_views = 1

        self.net = TransFuser(self.model_config, self.device)
        self._load_model()
        self.net.to(self.device)
        self.net.eval()

        self.command_planner_pool = {}
        self.step_pool = {}
        self.pred_wp_pool = {}
        self.input_buffer = {}
        self.stop_frames_pool = {}
        self.forced_move_frames_pool = {}

        # Keep the SafeBench creep recovery as a control-layer override:
        # only after the agent is stopped and still braking for several frames
        # do we briefly nudge it forward if the lidar safety box is clear.
        self.enable_creep = bool(config.get("transfuser_enable_creep", True))
        self.creep_speed_threshold = float(config.get("transfuser_creep_speed_threshold", 0.2))
        self.creep_hold_frames = int(config.get("transfuser_creep_hold_frames", 4))
        self.forced_move_duration = int(
            config.get("transfuser_forced_move_frames", config.get("transfuser_creep_duration", 8))
        )
        self.creep_throttle = float(config.get("transfuser_creep_throttle", 0.5))
        self.creep_steer_damping = float(config.get("transfuser_creep_steer_damping", 0.5))
        self.use_lidar_safe_check = bool(config.get("transfuser_use_lidar_safe_check", True))
        self.safety_box_x_min = float(config.get("transfuser_safety_box_x_min", -1.066))
        self.safety_box_x_max = float(config.get("transfuser_safety_box_x_max", 1.066))
        self.safety_box_y_min = float(config.get("transfuser_safety_box_y_min", -3.0))
        self.safety_box_y_max = float(config.get("transfuser_safety_box_y_max", 0.0))
        self.safety_box_z_min = float(config.get("transfuser_safety_box_z_min", -2.0))
        self.safety_box_z_max = float(config.get("transfuser_safety_box_z_max", -1.05))

    def _load_model(self):
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"TransFuser checkpoint not found: {self.weights_path}. "
                "Please place best_model.pth under transfuser_ckpt_dir."
            )

        state = torch.load(self.weights_path, map_location=self.device)
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                state = state["model_state_dict"]

        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported TransFuser checkpoint format: {type(state)}")

        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = self.net.load_state_dict(state, strict=True)
        if self.logger is not None:
            self.logger.log(f">> TransFuser loaded from {self.weights_path}")
            if len(missing) > 0 or len(unexpected) > 0:
                self.logger.log(
                    f">> TransFuser state mismatch, missing={len(missing)}, unexpected={len(unexpected)}",
                    "yellow",
                )

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.command_planner_pool = {}
        self.step_pool = {}
        self.pred_wp_pool = {}
        self.input_buffer = {}
        self.stop_frames_pool = {}
        self.forced_move_frames_pool = {}

        for i, info_i in enumerate(info):
            route_wps = info_i.get("route_waypoints", [])
            planner = KingRoutePlanner(4.0, 50.0)
            if len(route_wps) > 0:
                global_plan = [(wp.transform, _DummyCmd()) for wp in route_wps]
                planner.set_route(global_plan, gps=False)
            self.command_planner_pool[i] = planner
            self.step_pool[i] = -1
            self.pred_wp_pool[i] = None
            self.stop_frames_pool[i] = 0
            self.forced_move_frames_pool[i] = 0
            self.input_buffer[i] = {
                "lidar": deque(maxlen=self.model_config.seq_len),
                "gps": deque(maxlen=self.model_config.seq_len),
                "thetas": deque(maxlen=self.model_config.seq_len),
            }

    def train(self, replay_buffer):
        return None

    def set_mode(self, mode):
        self.mode = mode
        if mode == "eval":
            self.net.eval()
        else:
            self.net.train()

    @staticmethod
    def _get_speed_mps(ego_vehicle):
        v = ego_vehicle.get_velocity()
        return float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))

    @staticmethod
    def _get_ego_gps_and_compass(ego_vehicle):
        tf = ego_vehicle.get_transform()
        gps = np.array([tf.location.x, tf.location.y], dtype=np.float32)
        compass = float(tf.rotation.yaw / 180.0 * np.pi)
        return gps, compass

    def _get_obs_gps_and_compass(self, obs_item, ego_vehicle):
        gps = obs_item.get("gps", None)
        compass = obs_item.get("compass", None)

        if gps is None or compass is None:
            return self._get_ego_gps_and_compass(ego_vehicle)

        gps = np.asarray(gps, dtype=np.float32).reshape(-1)
        if gps.shape[0] < 2:
            return self._get_ego_gps_and_compass(ego_vehicle)

        compass = float(compass)
        if np.isnan(compass):
            compass = 0.0

        return gps[:2].copy(), compass

    def _compute_target_point(self, sid, pos_xy, compass):
        planner = self.command_planner_pool.get(sid)
        if planner is None:
            return None

        next_plan = planner.run_step(pos_xy)
        if next_plan is None:
            return None

        next_wp, _ = next_plan
        theta = compass + np.pi / 2.0
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )
        local_command_point = np.array([next_wp[0] - pos_xy[0], next_wp[1] - pos_xy[1]], dtype=np.float32)
        local_command_point = rot.T.dot(local_command_point)
        return local_command_point

    @staticmethod
    def _extract_obs_item(obs, idx):
        if isinstance(obs, dict):
            return obs
        if isinstance(obs, (list, tuple)):
            return obs[idx]
        if isinstance(obs, np.ndarray):
            item = obs[idx]
            if isinstance(item, np.ndarray) and item.shape == ():
                try:
                    return item.item()
                except Exception:
                    return item
            return item
        return obs

    def _preprocess_camera(self, camera):
        img = np.asarray(camera, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid camera shape for TransFuser: {img.shape}")
        img_t = torch.from_numpy(
            scale_and_crop_image(Image.fromarray(img), crop=self.model_config.input_resolution)
        ).unsqueeze(0)
        return img_t.to(self.device, dtype=torch.float32)

    @staticmethod
    def _normalize_lidar_points(lidar_points):
        points = np.asarray(lidar_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return points[:, :3].copy()

    def _lidar_to_features(self, sid):
        buf = self.input_buffer[sid]
        if len(buf["lidar"]) == 0:
            zeros = np.zeros(
                (2, self.model_config.input_resolution, self.model_config.input_resolution),
                dtype=np.float32,
            )
            return [torch.from_numpy(zeros).unsqueeze(0).to(self.device, dtype=torch.float32)]

        ego_theta = buf["thetas"][-1]
        ego_x, ego_y = buf["gps"][-1]

        lidar_processed = []
        for lidar_point_cloud, (curr_x, curr_y), curr_theta in zip(buf["lidar"], buf["gps"], buf["thetas"]):
            lidar_local = lidar_point_cloud.copy()
            if lidar_local.shape[0] > 0:
                lidar_local[:, 1] *= -1.0
                lidar_local = transform_2d_points(
                    lidar_local,
                    np.pi / 2 - curr_theta,
                    -curr_x,
                    -curr_y,
                    np.pi / 2 - ego_theta,
                    -ego_x,
                    -ego_y,
                )

            lidar_feat = lidar_to_histogram_features(
                lidar_local,
                crop=self.model_config.input_resolution,
            )
            lidar_t = torch.from_numpy(lidar_feat).unsqueeze(0)
            lidar_processed.append(lidar_t.to(self.device, dtype=torch.float32))

        return lidar_processed

    def _is_front_blocked(self, lidar_points):
        if not self.use_lidar_safe_check:
            return False

        points = np.asarray(lidar_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return False

        safety_box = points[:, :3].copy()
        safety_box[:, 1] *= -1.0
        safety_box = safety_box[safety_box[:, 2] > self.safety_box_z_min]
        safety_box = safety_box[safety_box[:, 2] < self.safety_box_z_max]
        safety_box = safety_box[safety_box[:, 1] > self.safety_box_y_min]
        safety_box = safety_box[safety_box[:, 1] < self.safety_box_y_max]
        safety_box = safety_box[safety_box[:, 0] > self.safety_box_x_min]
        safety_box = safety_box[safety_box[:, 0] < self.safety_box_x_max]
        return safety_box.shape[0] > 0

    def _predict_waypoints(self, sid, rgb, target_point, velocity):
        should_predict = (self.step_pool[sid] % 2 == 0) or (self.step_pool[sid] <= 4) or (self.pred_wp_pool[sid] is None)
        if should_predict:
            lidar_processed = self._lidar_to_features(sid)
            self.pred_wp_pool[sid] = self.net([rgb], lidar_processed, target_point, velocity)
        return self.pred_wp_pool[sid]

    def _run_one(self, sid, ego, obs_item):
        if not isinstance(obs_item, dict):
            raise RuntimeError(f"TransFuser expects dict obs for scenario {sid}, got {type(obs_item)!r}")

        camera = obs_item.get("camera", obs_item.get("img", None))
        lidar_points = obs_item.get("lidar_points", None)

        if camera is None:
            raise RuntimeError(f"TransFuser camera input missing for scenario {sid}")

        if lidar_points is None:
            lidar_points = np.zeros((0, 3), dtype=np.float32)

        gps, compass = self._get_obs_gps_and_compass(obs_item, ego)
        target_local = self._compute_target_point(sid, gps, compass)
        if target_local is None:
            raise RuntimeError(f"TransFuser target_point unavailable for scenario {sid}")

        rgb = self._preprocess_camera(camera)
        lidar = self._normalize_lidar_points(lidar_points)
        self.step_pool[sid] += 1
        self.input_buffer[sid]["lidar"].append(lidar)
        self.input_buffer[sid]["gps"].append(gps)
        self.input_buffer[sid]["thetas"].append(compass)

        if self.step_pool[sid] < self.model_config.seq_len:
            return [0.0, 0.0, 0.0]

        speed = float(obs_item.get("speed", self._get_speed_mps(ego)))
        target_point = torch.from_numpy(target_local).to(self.device, dtype=torch.float32).unsqueeze(0)
        velocity = torch.tensor([speed], dtype=torch.float32, device=self.device)

        pred_wp = self._predict_waypoints(sid, rgb, target_point, velocity)
        steer, throttle, brake, _ = self.net.control_pid(pred_wp, velocity)

        brake = float(brake)
        throttle = float(throttle)
        steer = float(steer)

        if self.enable_creep:
            front_blocked = self._is_front_blocked(lidar)
            is_stopped = speed < self.creep_speed_threshold

            if brake > 0.0 and is_stopped:
                self.stop_frames_pool[sid] += 1
            else:
                self.stop_frames_pool[sid] = 0

            should_forced_move = (
                self.stop_frames_pool[sid] >= self.creep_hold_frames
                and not front_blocked
                and self.forced_move_frames_pool[sid] < self.forced_move_duration
            )
            if should_forced_move:
                self.forced_move_frames_pool[sid] += 1
                brake = 0.0
                throttle = max(throttle, self.creep_throttle)
                steer *= self.creep_steer_damping
            elif front_blocked or self.stop_frames_pool[sid] < self.creep_hold_frames:
                self.forced_move_frames_pool[sid] = 0

        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        return [
            float(np.clip(throttle, 0.0, 1.0)),
            float(np.clip(steer, -1.0, 1.0)),
            float(np.clip(brake, 0.0, 1.0)),
        ]

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        with torch.no_grad():
            for i, info in enumerate(infos):
                sid = int(info["scenario_id"])
                ego = self.ego_vehicles[sid]
                obs_item = self._extract_obs_item(obs, i)
                actions.append(self._run_one(sid, ego, obs_item))
        return np.array(actions, dtype=np.float32)

    def load_model(self, episode=None):
        return None

    def save_model(self, episode):
        return None
