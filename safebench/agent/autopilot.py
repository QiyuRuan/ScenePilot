import numpy as np

from safebench.agent.base_policy import BasePolicy
from safebench.agent.expert.autopilot import AutoPilot
from safebench.agent.expert.nav_planner import interpolate_trajectory


class AutoPilotAgent(BasePolicy):
    name = 'autopilot'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mode = 'train'
        self.continue_episode = 0
        self.viz_route = bool(config.get('viz_route', False))
        self.controller_list = [AutoPilot(config, logger) for _ in range(config['num_scenario'])]

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        for i, ego in enumerate(ego_vehicles):
            route_waypoints = info[i].get('route_waypoints', [])
            world_map = ego.get_world().get_map()
            locs = [wp.transform.location for wp in route_waypoints]

            # Fallback for degenerate routes (e.g., single-point routes).
            if len(locs) < 2:
                tf = ego.get_transform()
                locs = [
                    tf.location,
                    tf.location + tf.get_forward_vector() * 30.0,
                ]

            gps_route, route = interpolate_trajectory(world_map, locs)
            self.controller_list[i].set_planner(ego, gps_route, route)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for info_i in infos:
            idx = info_i['scenario_id']
            control = self.controller_list[idx].run_step(input_data=None, viz_route=self.viz_route)
            throttle = float(control.throttle)
            steer = float(control.steer)
            brake = float(control.brake)
            actions.append([throttle, steer, brake])

        return np.array(actions, dtype=np.float32)

    def load_model(self, episode=None):
        pass

    def save_model(self, episode):
        pass
