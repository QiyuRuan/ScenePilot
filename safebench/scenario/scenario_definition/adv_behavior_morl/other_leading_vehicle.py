''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 17:45:21
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
import numpy as np 

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance

from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class OtherLeadingVehicle(BasicScenario):
    """
        The user-controlled ego vehicle follows a leading car driving down a given road. 
        At some point the leading car has to decelerate. The ego vehicle has to react accordingly by changing lane 
        to avoid a collision and follow the leading car in other lane. The scenario ends either via a timeout, 
        or if the ego vehicle drives some distance. (Traffic Scenario 05)
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OtherLeadingVehicle, self).__init__("OtherLeadingVehicle-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None

        self.dece_distance = 5
        self.need_decelerate = False
        self._ego_passed_first = False

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35

        self.acc_max = 3.0
        self.steering_max = 0.35

    def convert_actions(self, actions):
        """
        actions: [acc, steer], from ScenePilot, roughly in [-1, 1].
        Returns: carla.VehicleControl
        """
        acc = float(actions[0]) * self.acc_max
        steer = float(actions[1]) * self.steering_max

        acc = max(-self.acc_max, min(self.acc_max, acc))
        steer = max(-self.steering_max, min(self.steering_max, steer))

        if acc > 0:
            throttle = np.clip(acc / 3.0, 0.0, 1.0)
            brake = 0.0
            reverse = False
        else:
            throttle = 0.0
            brake = np.clip(-acc / 8.0, 0.0, 1.0)
            reverse = False

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            reverse=reverse
        )

    def initialize_actors(self):
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location, first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location, second_vehicle_waypoint.transform.rotation)
        
        self.actor_type_list = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        self.actor_transform_list = [first_vehicle_transform, second_vehicle_transform]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'
        self._first_vehicle_location = 35
        self._second_vehicle_location = self._first_vehicle_location + 2.5
        _first_vehicle_speed = 12
        _second_vehicle_speed = 10.5
        self.other_actor_speed = [_first_vehicle_speed, _second_vehicle_speed]

    def update_behavior(self, scenario_action):
        control_decel = self.convert_actions(scenario_action)

        cur_distance = calculate_distance_transforms(
            self.actor_transform_list[0],
            CarlaDataProvider.get_transform(self.other_actors[0])
        )
        if cur_distance > self.dece_distance:
            self.need_decelerate = True

        if self.need_decelerate:
            self.other_actors[0].apply_control(control_decel)
        else:
            self.scenario_operation.go_straight(self.other_actor_speed[0], 0)

        if not self._ego_passed_first:
            ego_tf = CarlaDataProvider.get_transform(self.ego_vehicle)
            first_tf = CarlaDataProvider.get_transform(self.other_actors[0])
            if ego_tf is not None and first_tf is not None:
                ego_loc = ego_tf.location
                first_loc = first_tf.location
                try:
                    wp = self._map.get_waypoint(ego_loc, project_to_road=True)
                    if wp is not None:
                        lane_yaw = wp.transform.rotation.yaw / 180 * np.pi
                        lane_dir = np.array([np.cos(lane_yaw), np.sin(lane_yaw)], dtype=np.float32)
                        rel = np.array([first_loc.x - ego_loc.x, first_loc.y - ego_loc.y], dtype=np.float32)
                        if float(np.dot(rel, lane_dir)) < -1.0:
                            self._ego_passed_first = True
                    else:
                        fwd = ego_tf.get_forward_vector()
                        rel = np.array([first_loc.x - ego_loc.x, first_loc.y - ego_loc.y], dtype=np.float32)
                        if float(rel[0] * fwd.x + rel[1] * fwd.y) < -1.0:
                            self._ego_passed_first = True
                except Exception:
                    pass

        if self._ego_passed_first:
            self.other_actors[1].apply_control(control_decel)
        else:
            self.scenario_operation.go_straight(self.other_actor_speed[1], 1)


    def check_stop_condition(self):
        pass
