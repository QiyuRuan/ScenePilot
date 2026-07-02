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

        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35

        self.acc_max = 3.0          # Maximum longitudinal acceleration (m/s^2)
        self.steering_max = 0.2    # Maximum steering angle in radians

    def convert_actions(self, actions):
        """
        actions: [acc, steer], from ScenePilot, roughly in [-1, 1]
        Returns: carla.VehicleControl
        """
        # 1) Unpack action
        acc   = float(actions[0])   # Normalized longitudinal acceleration
        steer = float(actions[1])   # Normalized steering

        # 2) Scale to physical range and clip
        acc   = acc * self.acc_max
        steer = steer * self.steering_max

        acc   = max(-self.acc_max,      min(self.acc_max,      acc))
        steer = max(-self.steering_max, min(self.steering_max, steer))

        # 3) Map acceleration to throttle/brake
        if acc > 0:
            throttle = np.clip(acc / 3.0, 0.0, 1.0)
            brake    = 0.0
            reverse  = False
        else:
            # Keep reverse disabled; negative acceleration maps to brake.
            throttle = 0.0
            brake    = np.clip(-acc / 8.0, 0.0, 1.0)
            reverse  = False

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            reverse=reverse
        )
        return control

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
        _second_vehicle_speed = 12
        self.other_actor_speed = [_first_vehicle_speed, _second_vehicle_speed]

    def update_behavior(self, scenario_action):
        # RL action controls the lead vehicle (index 0) during deceleration.
        control_decel = self.convert_actions(scenario_action)

        # Compute the lead vehicle's distance to the trigger point, preserving existing logic.
        cur_distance = calculate_distance_transforms(
            self.actor_transform_list[0],
            CarlaDataProvider.get_transform(self.other_actors[0])
        )
        if cur_distance > self.dece_distance:
            self.need_decelerate = True

        # --- Lead vehicle: choose fixed speed or RL control based on need_decelerate. ---
        if self.need_decelerate:
            # During deceleration, control the lead vehicle with RL output, allowing slight lane-preserving steer.
            self.other_actors[0].apply_control(control_decel)
        else:
            # Before the trigger point, keep driving straight at the configured speed.
            self.scenario_operation.go_straight(self.other_actor_speed[0], 0)

        # --- Second vehicle: always drives straight at the configured speed as background traffic. ---
        self.scenario_operation.go_straight(self.other_actor_speed[1], 1)


    def check_stop_condition(self):
        pass
