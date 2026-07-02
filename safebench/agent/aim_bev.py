import json
import os
from collections import deque

import numpy as np
import pygame
import torch

from safebench.agent.base_policy import BasePolicy
from safebench.agent.aim_bev_model import AimBev
from safebench.agent.aim_bev_render import DatagenBEVRenderer, MapImage


class _DummyCmd:
    value = 4


class _RoutePlanner:
    def __init__(self, min_distance=7.5, max_distance=25.0):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mean = np.array([0.0, 0.0], dtype=np.float32)
        self.scale = np.array([111324.60662786, 111319.490945], dtype=np.float32)

    def set_route(self, global_plan):
        self.route.clear()
        for pos, cmd in global_plan:
            p = np.array([pos.location.x, pos.location.y], dtype=np.float32)
            p -= self.mean
            self.route.append((p, cmd))

    def run_step(self, pos_xy):
        if len(self.route) == 0:
            return None
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i - 1][0])
            distance = np.linalg.norm(self.route[i][0] - pos_xy)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        return self.route[1] if len(self.route) > 1 else self.route[0]


class AimBEVAgent(BasePolicy):
    name = "aim_bev"
    type = "unlearnable"

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mode = "eval"
        self.continue_episode = 0
        self.ego_vehicles = []

        root_dir = config.get("ROOT_DIR", "")
        ckpt_dir = config.get(
            "aim_bev_ckpt_dir",
            "safebench/agent/model_ckpt/aim_bev/regular",
        )
        if not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(root_dir, ckpt_dir) if root_dir else ckpt_dir
        self.ckpt_dir = ckpt_dir

        args_path = os.path.join(self.ckpt_dir, "args.txt")
        self.args_map = {}
        if os.path.isfile(args_path):
            with open(args_path, "r", encoding="utf-8") as f:
                self.args_map = json.load(f)

        self.pred_len = int(config.get("pred_len", self.args_map.get("pred_len", 4)))
        self.seq_len = int(config.get("seq_len", self.args_map.get("seq_len", 1)))
        self.detection_radius = float(config.get("detection_radius", 30.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_pool = {}
        self.command_planner_pool = {}
        self.weights_path = os.path.join(self.ckpt_dir, "model.pth")
        self.renderer = None
        self.global_map = None
        self.vehicle_template = torch.ones(1, 1, 22, 9, device=self.device)
        self._build_models(int(config.get("num_scenario", 1)))

    def _build_models(self, num_scenario):
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"AIM-BEV checkpoint not found: {self.weights_path}")

        self.model_pool = {}
        for sid in range(max(1, num_scenario)):
            net = AimBev(
                device=self.device,
                pred_len=self.pred_len,
                batch_size=1,
            )
            state = torch.load(self.weights_path, map_location=self.device)
            net.load_state_dict(state)
            net.to(self.device)
            net.eval()
            self.model_pool[sid] = net
            self.command_planner_pool[sid] = _RoutePlanner()

        if self.logger is not None:
            self.logger.log(f">> AIM-BEV loaded from {self.weights_path}")

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        if len(ego_vehicles) > len(self.model_pool):
            self._build_models(len(ego_vehicles))
        for net in self.model_pool.values():
            net.speed_controller.reset()
            net.turn_controller.reset()
        for sid in list(self.command_planner_pool.keys()):
            self.command_planner_pool[sid].route.clear()

        # Build static semantic map exactly as KING's datagen pipeline.
        if len(self.ego_vehicles) > 0:
            world = self.ego_vehicles[0].get_world()
            world_map = world.get_map()
            map_image = MapImage(world, world_map)
            make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
            road = make_image(map_image.map_surface)
            lane = make_image(map_image.lane_surface)
            channels = 4
            global_map = np.zeros((1, channels) + road.shape, dtype=np.float32)
            global_map[:, 0, ...] = road / 255.0
            global_map[:, 1, ...] = lane / 255.0
            self.global_map = torch.tensor(global_map, device=self.device, dtype=torch.float32)
            world_offset = torch.tensor(map_image._world_offset, device=self.device, dtype=torch.float32)
            map_dims = self.global_map.shape[2:4]
            self.renderer = DatagenBEVRenderer(self.device, world_offset, map_dims, data_generation=False)

        # Build per-scenario KING-style command planners.
        for i, info_i in enumerate(info):
            route_wps = info_i.get("route_waypoints", [])
            if len(route_wps) == 0:
                continue
            global_plan = [(wp.transform, _DummyCmd()) for wp in route_wps]
            self.command_planner_pool[i].set_route(global_plan)

    def train(self, replay_buffer):
        return None

    def set_mode(self, mode):
        self.mode = mode
        for net in self.model_pool.values():
            if mode == "eval":
                net.eval()
            else:
                net.train()

    def _render_bev(self, ego_vehicle):
        if self.renderer is None or self.global_map is None:
            raise RuntimeError("AIM-BEV renderer not initialized; call set_ego_and_route first.")

        birdview = self.renderer.get_local_birdview(
            self.global_map,
            torch.tensor([ego_vehicle.get_transform().location.x, ego_vehicle.get_transform().location.y], device=self.device, dtype=torch.float32),
            torch.tensor([ego_vehicle.get_transform().rotation.yaw / 180 * np.pi], device=self.device, dtype=torch.float32),
        )

        actors = ego_vehicle.get_world().get_actors()
        ego_pos = torch.tensor([ego_vehicle.get_transform().location.x, ego_vehicle.get_transform().location.y], device=self.device, dtype=torch.float32)
        ego_yaw = torch.tensor([ego_vehicle.get_transform().rotation.yaw / 180 * np.pi], device=self.device, dtype=torch.float32)

        for vehicle in actors.filter("*vehicle*"):
            if vehicle.id == ego_vehicle.id:
                continue
            if vehicle.get_location().distance(ego_vehicle.get_location()) >= self.detection_radius:
                continue
            pos = torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y], device=self.device, dtype=torch.float32)
            yaw = torch.tensor([vehicle.get_transform().rotation.yaw / 180 * np.pi], device=self.device, dtype=torch.float32)
            self.renderer.render_agent_bv(
                birdview, ego_pos, ego_yaw, self.vehicle_template, pos, yaw, channel=2
            )

        return birdview

    def _get_speed_mps(self, ego_vehicle):
        v = ego_vehicle.get_velocity()
        return float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))

    def _compute_target_point(self, sid, ego_vehicle):
        planner = self.command_planner_pool.get(sid)
        if planner is None:
            return None
        pos = np.array([ego_vehicle.get_location().x, ego_vehicle.get_location().y], dtype=np.float32)
        next_plan = planner.run_step(pos)
        if next_plan is None:
            return None
        next_wp, _ = next_plan

        yaw_rad = ego_vehicle.get_transform().rotation.yaw / 180.0 * np.pi
        theta = yaw_rad + np.pi / 2.0
        r = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]], dtype=np.float32)
        local_command_point = r.T.dot(local_command_point)
        return local_command_point

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        with torch.no_grad():
            for i, info in enumerate(infos):
                sid = int(info.get("scenario_id", i))
                if sid >= len(self.ego_vehicles) or sid not in self.model_pool:
                    sid = i if i < len(self.ego_vehicles) and i in self.model_pool else 0
                net = self.model_pool.get(sid, self.model_pool[0])
                ego = self.ego_vehicles[sid]

                target_local = self._compute_target_point(sid, ego)
                if target_local is None:
                    actions.append([0.0, 0.0, 0.0])
                    continue

                target_point = torch.from_numpy(target_local).to(self.device).unsqueeze(0)
                light_hazard = torch.tensor(
                    [[float(info.get("light_hazard", 0.0))]],
                    dtype=torch.float32,
                    device=self.device,
                )
                velocity = torch.tensor([[self._get_speed_mps(ego)]], dtype=torch.float32, device=self.device)

                bev = self._render_bev(ego)
                encoding = net.image_encoder([bev])
                pred_wp = net([encoding], target_point, light_hazard=light_hazard)
                steer, throttle, brake = net.control_pid(pred_wp, velocity)

                actions.append(
                    [
                        float(torch.clamp(throttle[0, 0], 0.0, 1.0).item()),
                        float(torch.clamp(steer[0, 0], -1.0, 1.0).item()),
                        float(torch.clamp(brake[0, 0], 0.0, 1.0).item()),
                    ]
                )

        return np.array(actions, dtype=np.float32)

    def load_model(self, episode=None):
        return None

    def save_model(self, episode):
        return None
