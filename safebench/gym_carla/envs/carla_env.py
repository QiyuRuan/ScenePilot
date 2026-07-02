''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 19:41:22
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py>
    Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import random

import numpy as np
import pygame
from skimage.transform import resize
import gym
from gym import spaces
import carla
import math

from shapely.geometry import Polygon

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb, 
    rgb_to_display_surface, 
    get_lane_dis, 
    get_pos, 
    get_preview_lane_dis
)
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.scenario_definition.perception_scenario import PerceptionScenario
from safebench.scenario.scenario_definition.scenic_scenario import ScenicScenario
from safebench.scenario.scenario_manager.scenario_manager import ScenarioManager
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.route_manipulation import interpolate_trajectory

from safebench.gym_carla.envs.phys_safe import (phys_safe_lat_opp,phys_safe_lat_same,phys_safe_opposite,phys_safe_same_lane,set_from_cfg)
from types import SimpleNamespace
from safebench.gym_carla.envs import av_safe

class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """
    def __init__(self, env_params, birdeye_render=None, display=None, world=None, logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

        self._term_flag = False  
        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # rl config
        self.rl_config = env_params['rl_config']
        av_safe.set_from_cfg(self.rl_config['av_safe'])
        set_from_cfg(self.rl_config['phys_safe'])

        # av_safe
        self._av_training_started = False
        self.route_level_avsafe_training = bool(env_params.get('route_level_avsafe_training', False))


        # NPC waypoint
        self.npc_goal_waypoint = None          #  local npc target from frea
        self.target_waypoint = None
        self.current_waypoint = None
        self.red_light = False
        self._npc_goal_dist = {}           # {npc_id: last_dist_to_goal}

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.is_running = True
        self.env_id = None
        self.ego_vehicle = None
        self.auto_ego = env_params['auto_ego']

        # debug carla collsion sensor
        self._bbox_collision_reported = False
        self._tilt_collision_reported = False
        self._pending_collision = None
        self._pending_collision_step = None
        self._last_rot = None
        self._last_rot_step = None
        

        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.lidar_data = None
        self.lidar_height = 2.1

        
        # scenario manager
        use_scenic = True if  env_params['scenario_category'] == 'scenic' else False
        self.scenario_manager = ScenarioManager(self.logger, use_scenic=use_scenic)

        # for birdeye view and front view visualization
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']
        self.obs_type = env_params['obs_type']

        # for env wrapper
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        # For autopilot: share desired speed via global provider.
        CarlaDataProvider.set_ego_desired_speed(self.desired_speed)
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.scenario_category = env_params['scenario_category']
        self.warm_up_steps = env_params['warm_up_steps']

        if self.scenario_category in ['planning', 'scenic']:
            self.obs_size = int(self.obs_range/self.lidar_bin)
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([-2, -1, -5, 0], dtype=np.float32), np.array([2, 1, 30, 1], dtype=np.float32), dtype=np.float32)
            }
        elif self.scenario_category == 'perception':
            self.obs_size = env_params['image_sz']
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            }
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # define obs space
        self.observation_space = spaces.Dict(observation_space_dict)

        # action and observation spaces
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            # assume the output of NN is from -1 to 1
            self.action_space = spaces.Box(np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32), dtype=np.float32)  # acc, steer

    def _create_sensors(self):
        # collision sensor
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        transfuser_mode = (self.obs_type == 4 and self.scenario_category != 'perception')

        if self.scenario_category != 'perception':
            # lidar sensor
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            if transfuser_mode:
                # Match KING TransFuser sensor configuration.
                self.lidar_height = 2.5
                self.lidar_trans = carla.Transform(
                    carla.Location(x=1.3, z=2.5),
                    carla.Rotation(yaw=-90.0),
                )
                self.lidar_bp.set_attribute('range', '85')
                self.lidar_bp.set_attribute('rotation_frequency', '20')
                self.lidar_bp.set_attribute('channels', '64')
                self.lidar_bp.set_attribute('upper_fov', '10')
                self.lidar_bp.set_attribute('lower_fov', '-30')
                self.lidar_bp.set_attribute('points_per_second', str(2 * 600000))
                self.lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')
                self.lidar_bp.set_attribute('dropoff_general_rate', '0.45')
                self.lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
                self.lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
            else:
                self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
                self.lidar_bp.set_attribute('channels', '16')
                self.lidar_bp.set_attribute('range', '1000')

        # camera sensor
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        if transfuser_mode:
            # Match KING TransFuser front camera.
            self.camera_img = np.zeros((300, 400, 3), dtype=np.uint8)
            self.camera_trans = carla.Transform(
                carla.Location(x=1.3, z=2.3),
                carla.Rotation(yaw=0.0),
            )
            self.camera_bp.set_attribute('image_size_x', '400')
            self.camera_bp.set_attribute('image_size_y', '300')
            self.camera_bp.set_attribute('fov', '100')
        else:
            self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
            self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
            self.camera_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        # create scenario accoridng to different types
        if self.scenario_category == 'perception':
            scenario = PerceptionScenario(
                world=self.world, 
                config=config, 
                ROOT_DIR=self.ROOT_DIR, 
                ego_id=env_id, 
                logger=self.logger,
            )
        elif self.scenario_category == 'planning':
            scenario = RouteScenario(
                world=self.world, 
                config=config, 
                ego_id=env_id, 
                max_running_step=self.max_episode_step, 
                logger=self.logger
            )
        elif self.scenario_category == 'scenic':
            scenario = ScenicScenario(
                world=self.world, 
                config=config, 
                ego_id=env_id, 
                max_running_step=self.max_episode_step, 
                logger=self.logger
            )
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)

    def _run_scenario(self, scenario_init_action):
        self.scenario_manager.run_scenario(scenario_init_action)

    def _parse_route(self, config):
        # interp waypoints as init waypoints
        route = interpolate_trajectory(self.world, config.trajectory)

        # TODO: these waypoints can be directly got from scenario
        waypoints_list = []
        carla_map = self.world.get_map()
        for node in route:
            loc = node[0].location
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            waypoints_list.append(waypoint)
        return waypoints_list

    def get_static_obs(self, config):
        """
            This function returns static observation used for static scenario generation
        """
        # get route
        route = interpolate_trajectory(self.world, config.trajectory, 5.0)

        # get [x, y] along the route
        waypoint_xy = []
        for transform_tuple in route:
            waypoint_xy.append([transform_tuple[0].location.x, transform_tuple[0].location.y])
        
        # combine state obs    
        state = {
            'route': np.array(waypoint_xy),   # [n, 2]
            'target_speed': self.desired_speed,
        }
        return state

    def reset(self, config, env_id, scenario_init_action):
        self.config = config
        self.env_id = env_id
        self._bbox_collision_reported = False
        self._tilt_collision_reported = False
        self._pending_collision = None
        self._pending_collision_step = None
        self._last_rot = None
        self._last_rot_step = None

        # create sensors, load and run scenarios
        self._create_sensors()
        self._create_scenario(config, env_id)
        self._run_scenario(scenario_init_action)

        self._attach_sensor()

        self._latest_feat = None          # Used by AV-safe TD-0
        if not self.route_level_avsafe_training:
            self._av_training_started = False

        # route planner for ego vehicle
        self.route_waypoints = self._parse_route(config)
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.route_waypoints)
        (
            self.waypoints,
            self.target_road_option,
            self.current_waypoint,
            self.target_waypoint,
            self.npc_goal_waypoint,
            self.red_light,
            self.vehicle_front,
        ) = self.routeplanner.run_step()

        #  Clear the cached NPC-to-goal distance on each reset.
        self._npc_goal_dist = {}

        # change view point
        #location = carla.Location(x=100, y=100, z=300)
        #spectator = self.world.get_spectator()
        #spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=270.0, pitch=-90.0)))
    
        # Get actors polygon list (for visualization)
        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # Get actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories = [vehicle_info_dict_list[0]]
        self.vehicle_accelerations = [vehicle_info_dict_list[1]]
        self.vehicle_angular_velocities = [vehicle_info_dict_list[2]]
        self.vehicle_velocities = [vehicle_info_dict_list[3]]

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # applying setting can tick the world and get data from sensros
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()
        return self._get_obs(), self._get_info()

    def _attach_sensor(self):
        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Add lidar sensor
        if self.scenario_category != 'perception' and not self.disable_lidar:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):            
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array



    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
                # get update on evaluation results before getting update of running status
                if self.scenario_category in ['perception']:
                    assert isinstance(ego_action, dict), 'ego action in ObjectDetectionScenario should be a dict'
                    world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
                    fov = self.camera_bp.get_attribute('fov').as_float()
                    image_w, image_h = self.obs_size, self.obs_size
                    self.scenario_manager.background_scenario.evaluate(ego_action, world_2_camera, image_w, image_h, fov, self.camera_img)
                    ego_action = ego_action['ego_action']

                # pass scenario action into manager
                self.scenario_manager.get_update(timestamp, scenario_action)
                self.is_running = self.scenario_manager._running

                # Calculate acceleration and steering
                if not self.auto_ego:
                    if self.discrete:
                        acc = self.discrete_act[0][ego_action // self.n_steer]
                        steer = self.discrete_act[1][ego_action % self.n_steer]
                        # normalize and clip the action
                        acc = acc * self.acc_max
                        steer = steer * self.steering_max
                        acc = max(min(self.acc_max, acc), -self.acc_max)
                        steer = max(min(self.steering_max, steer), -self.steering_max)

                        # Convert acceleration to throttle and brake
                        if acc > 0:
                            throttle = np.clip(acc / 3, 0, 1)
                            brake = 0
                        else:
                            throttle = 0
                            brake = np.clip(-acc / 8, 0, 1)
                    else:
                        # Dual-path executor:
                        # 2D action -> legacy [acc_norm, steer_norm]
                        # 3D action -> direct [throttle, steer, brake] passthrough
                        act_arr = np.asarray(ego_action, dtype=np.float32).reshape(-1)
                        if act_arr.size >= 3:
                            throttle = float(act_arr[0])
                            steer = float(act_arr[1])
                            brake = float(act_arr[2])
                        else:
                            acc = ego_action[0]
                            steer = ego_action[1]

                            # normalize and clip the action
                            acc = acc * self.acc_max
                            steer = steer * self.steering_max
                            acc = max(min(self.acc_max, acc), -self.acc_max)
                            steer = max(min(self.steering_max, steer), -self.steering_max)

                            # Convert acceleration to throttle and brake
                            if acc > 0:
                                throttle = np.clip(acc / 3, 0, 1)
                                brake = 0
                            else:
                                throttle = 0
                                brake = np.clip(-acc / 8, 0, 1)

                    # apply control
                    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    self.ego_vehicle.apply_control(act)
            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # Append actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        while len(self.vehicle_trajectories) > self.max_past_step:
            self.vehicle_trajectories.pop(0)
        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        while len(self.vehicle_accelerations) > self.max_past_step:
            self.vehicle_accelerations.pop(0)
        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        while len(self.vehicle_angular_velocities) > self.max_past_step:
            self.vehicle_angular_velocities.pop(0)
        self.vehicle_velocities.append(vehicle_info_dict_list[3])
        while len(self.vehicle_velocities) > self.max_past_step:
            self.vehicle_velocities.pop(0)

        # route planner
        (
            self.waypoints,
            self.target_road_option,
            self.current_waypoint,
            self.target_waypoint,
            self.npc_goal_waypoint,
            self.red_light,
            self.vehicle_front,
        ) = self.routeplanner.run_step()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        # Supplement for a CARLA collision-sensor bug.
        if len(self.collision_hist) == 0 and self._has_bbox_collision():
            self._set_pending_collision("bbox overlap")
        if len(self.collision_hist) == 0 and self._has_tilt_collision():
            self._set_pending_collision("tilt")
        self._check_pending_collision()
        # if self.config is not None and self.time_step % 10 == 0:
        #     rot = self.ego_vehicle.get_transform().rotation
        #     msg = (
        #         f"[Debug][CarlaEnv] roll={rot.roll:.3f}, pitch={rot.pitch:.3f}, "
        #         f"yaw={rot.yaw:.3f} (data_id={self.config.data_id})"
        #     )
        #     if self.logger is not None:
        #         self.logger.log(msg)
        #     else:
        #         print(msg)

        # ---------- Risk-feature ----------
        # only vehicle but scenario 1 get walker
        # npc, dist  = self._get_closest_vehicle()
        npc, dist  = self._get_closest_npc()
        feat = self._risk_feature(npc, dist)            # s_{t+1}

        # ---------- Potential-based shaping ----------
        r = av_safe.gamma()                           # Read gamma directly from av_safe.
        P_curr = feat[-2]                                  # inv_d(s')
        P_prev = self._latest_feat[-2] if self._latest_feat is not None else P_curr
        shaping = r * P_curr - P_prev
        
        # ---------- dense reward ----------
        r_td = (1.0 if len(self.collision_hist) else 0.0) + shaping

        collision = len(self.collision_hist) > 0
        if av_safe._enabled and av_safe._training and collision and not self._av_training_started:
            self._av_training_started = True
            self.logger.log('[AV-safe] first collision – start TD-0 training', 'yellow')

        # ---------- TD-0 update ----------
        if av_safe._enabled and av_safe._training and self._av_training_started and self._latest_feat is not None and not av_safe.ready():
            #Returns 0 if warm-up is not ready.
            av_safe.update(self._latest_feat,    # s_t
                        feat,                 # s_{t+1}
                        r_td,            # # dense reward
                        self._term_flag)      # Termination flag

        self._latest_feat = feat      # Cache the current frame for the next frame.


        return (self._get_obs(), self._get_reward(), self._terminal(), self._get_info())
    
    def _get_info(self):
        # state information
        info = {
            'waypoints': self.waypoints,
            'route_waypoints': self.route_waypoints,
            'target_waypoint': self.target_waypoint,
            'current_waypoint': self.current_waypoint,
            'light_hazard': float(self.red_light),
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost(),

            'npc_moreward_vec': self._get_scenario_moreward_vec(),

            # ChatScene train_agent path
            'collision': int(len(self.collision_hist) > 0),
        }

        # info from scenarios
        info.update(self.scenario_manager.background_scenario.update_info())
        return info


    def _init_traffic_light(self):
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                actor.set_red_time(3)
                actor.set_green_time(3)
                actor.set_yellow_time(1)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    # Supplement for a CARLA collision-sensor bug.
    def _has_bbox_collision(self):
        if not self.vehicle_polygons:
            return False
        ego_id = self.ego_vehicle.id
        ego_poly = self.vehicle_polygons[-1].get(ego_id)
        if ego_poly is None or len(ego_poly) < 3:
            return False
        ego_shape = Polygon(ego_poly)
        for actor_id, poly in self.vehicle_polygons[-1].items():
            if actor_id == ego_id or len(poly) < 3:
                continue
            if ego_shape.intersects(Polygon(poly)):
                return True
        if self.walker_polygons:
            for _, poly in self.walker_polygons[-1].items():
                if len(poly) < 3:
                    continue
                if ego_shape.intersects(Polygon(poly)):
                    return True
        return False

    def _has_tilt_collision(self):
        cfg = self.rl_config.get('collision_fallback', {})
        min_speed = float(cfg.get('min_speed', 1.0))
        roll_th = float(cfg.get('roll_deg', 20.0))
        pitch_th = float(cfg.get('pitch_deg', 15.0))
        roll_rate_th = float(cfg.get('roll_rate_deg', 7.0))
        pitch_rate_th = float(cfg.get('pitch_rate_deg', 7.0))
        v = self.ego_vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        if speed < min_speed:
            return False
        rot = self.ego_vehicle.get_transform().rotation
        if abs(rot.roll) >= roll_th or abs(rot.pitch) >= pitch_th:
            self._last_rot = rot
            self._last_rot_step = self.time_step
            return True
        if self._last_rot is None or self._last_rot_step is None:
            self._last_rot = rot
            self._last_rot_step = self.time_step
            return False
        step_delta = max(1, self.time_step - self._last_rot_step)
        roll_rate = (rot.roll - self._last_rot.roll) / step_delta
        pitch_rate = (rot.pitch - self._last_rot.pitch) / step_delta
        self._last_rot = rot
        self._last_rot_step = self.time_step
        if abs(roll_rate) >= roll_rate_th or abs(pitch_rate) >= pitch_rate_th:
            return True
        return False

    def _set_pending_collision(self, reason):
        if self._pending_collision is None:
            self._pending_collision = reason
            self._pending_collision_step = self.time_step
            # msg = f"[Debug][CarlaEnv] pending collision reason={reason}"
            # if self.logger is not None:
            #     self.logger.log(msg)
            # else:
            #     print(msg)

    def _check_pending_collision(self):
        if self._pending_collision is None:
            return
        if len(self.collision_hist) > 0:
            self._pending_collision = None
            self._pending_collision_step = None
            return
        cfg = self.rl_config.get('collision_fallback', {})
        delay_steps = int(cfg.get('delay_steps', 1))
        if self.time_step - self._pending_collision_step < delay_steps:
            return
        self.collision_hist.append(0.0)
        self._mark_bbox_collision_failure()
        if self._pending_collision == "bbox overlap" and not self._bbox_collision_reported:
            # msg = "[Debug][CarlaEnv] bbox overlap confirmed after delay; treating as collision fallback"
            # if self.logger is not None:
            #     self.logger.log(msg)
            # else:
            #     print(msg)
            self._bbox_collision_reported = True
        if self._pending_collision == "tilt" and not self._tilt_collision_reported:
            # msg = "[Debug][CarlaEnv] tilt confirmed after delay; treating as collision fallback"
            # if self.logger is not None:
            #     self.logger.log(msg)
            # else:
            #     print(msg)
            self._tilt_collision_reported = True
        self._pending_collision = None
        self._pending_collision_step = None

     # Supplement for a CARLA collision-sensor bug.
    def _mark_bbox_collision_failure(self):
        scenario = getattr(self.scenario_manager, "background_scenario", None)
        criteria = getattr(scenario, "criteria", None) if scenario is not None else None
        if not criteria or "collision" not in criteria:
            return
        criterion = criteria["collision"]
        if getattr(criterion, "test_status", None) == "FAILURE":
            return
        criterion.test_status = "FAILURE"
        try:
            criterion.actual_value += 1
        except Exception:
            pass

    def _get_actor_info(self, filt):
        actor_trajectory_dict = {}
        actor_acceleration_dict = {}
        actor_angular_velocity_dict = {}
        actor_velocity_dict = {}

        for actor in self.world.get_actors().filter(filt):
            actor_trajectory_dict[actor.id] = actor.get_transform()
            actor_acceleration_dict[actor.id] = actor.get_acceleration()
            actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()
        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        # State observation
        ego_trans = self.ego_vehicle.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw = np.arcsin(np.cross(w, yaw))

        v = self.ego_vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        acc = self.ego_vehicle.get_acceleration()
        # state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front], dtype=np.float32)
        self._cached_state = state              # Used by risk features

        if self.scenario_category != 'perception': 
            # set ego information for birdeye_render
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            # render birdeye image with the birdeye_render
            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0]/2), int(birdeye_surface.shape[1]/2))
            width = height = int(self.display_size/2)
            birdeye = birdeye_surface[center[0]-width:center[0]+width, center[1]-height:center[1]+height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            if not self.disable_lidar:
                if self.lidar_data is not None:
                    # get Lidar image
                    point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                    x = point_cloud[:, 0:1]
                    y = point_cloud[:, 1:2]
                    z = point_cloud[:, 2:3]
                    intensity = point_cloud[:, 3:4]
                    point_cloud = np.concatenate([y, -x, z], axis=1)
                    # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin, and z is set to be two bins.
                    y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                    x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                    z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
                    # Get lidar image according to the bins
                    lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                    lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                    lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                    wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
                    wayptimg = np.expand_dims(wayptimg, axis=2)
                    wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                    # Get the final lidar image
                    lidar = np.concatenate((lidar, wayptimg), axis=2)
                    lidar = np.flip(lidar, axis=1)
                    lidar = np.rot90(lidar, 1) * 255
                else:
                    lidar = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                # display lidar image
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id*self.display_size))

                # display camera image
                camera_raw = self.camera_img.astype(np.uint8)
                camera_display = resize(camera_raw, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera_display, self.display_size)
                self.display.blit(camera_surface, (self.display_size*2, self.env_id*self.display_size))
            else:
                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                # display camera image
                camera_raw = self.camera_img.astype(np.uint8)
                camera_display = resize(camera_raw, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera_display, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id*self.display_size))

            camera_obs = camera_raw if self.obs_type == 4 else camera_display.astype(np.uint8)
            obs = {
                'camera': camera_obs,
                'lidar': None if self.disable_lidar else lidar.astype(np.uint8),
                'birdeye': birdeye.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        else:
            """ Get the observations for object detection. """
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (0, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        return obs

    def _get_reward(self):
        """ Calculate the step reward. """
        # TODO: reward for collision, there should be a signal from scenario
        r_collision = -1 if len(self.collision_hist) > 0 else 0

        # reward for steering:
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        # reward for speed tracking
        v = self.ego_vehicle.get_velocity()

        # cost for too fast
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed else 0

        # cost for lateral acceleration
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon**2

        # combine all rewards
        r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        return r

    def _get_cost(self):
        # cost for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1
        return r_collision

    # Physical-safety additions
     # ---------- Helper: find the NPC closest to AV ----------
    def _get_closest_vehicle(self, radius=60.0):
        ego_loc = self.ego_vehicle.get_transform().location
        min_d, closest = 1e9, None
        for veh in self.world.get_actors().filter('vehicle.*'):
            if veh.id == self.ego_vehicle.id:
                continue
            # Only scalar distance is available, not a direction vector.
            d = ego_loc.distance(veh.get_transform().location)
            if d < min_d and d < radius:
                min_d, closest = d, veh
        return closest, min_d
    
    #  Allow any new NPC type.
    def _get_closest_npc(self, radius=500.0):
        ego_loc = self.ego_vehicle.get_transform().location
        closest, min_d, kind = None, 1e9, None
        for actor in list(self.world.get_actors().filter('vehicle.*')) + \
                    list(self.world.get_actors().filter('walker.*')):
            if actor.id == self.ego_vehicle.id:
                continue
            d = ego_loc.distance(actor.get_transform().location)
            if d < min_d and d < radius:
                closest, min_d = actor, d
                kind = 'walker' if actor.type_id.startswith('walker.') else 'vehicle'
        return closest, min_d

    def safe_lon_distance(self, v_av_lon, v_npc_lon,
                      yaw_av, yaw_npc):
        """
        Longitudinal physical safety distance d_safe^{lon}

        yaw: deg, CARLA 0 deg -> +x
        Rules:
        - |Δyaw| <=  th_same      -> same-direction formula
        - |Δyaw| >=  th_oppo      -> opposite-direction formula
        - otherwise, in lateral crossing zones     -> use the opposite-direction formula for a conservative bound
        """
        ps_cfg = self.rl_config['phys_safe']
        dyaw = (yaw_npc - yaw_av + 180) % 360 - 180   # [-180,180]

        if abs(dyaw) <= ps_cfg['th_same_dir']:
            # Same direction: rear-end or cut-in.
            return phys_safe_same_lane(v_av_lon, v_npc_lon)
        else:
            # Opposite direction or lateral crossing: use the head-on formula for a conservative bound.
            return phys_safe_opposite(v_av_lon, v_npc_lon)
        

    def safe_lat_distance(self, v_av_lat, v_npc_lat):
        """
        Return lateral physical safety distance d_safe^{lat}
        """
        ps_cfg = self.rl_config['phys_safe']
        eps = ps_cfg['eps']
        same_dir = np.sign(v_av_lat) == np.sign(v_npc_lat) and \
                abs(v_av_lat) > eps and abs(v_npc_lat) > eps
        if same_dir:
            return phys_safe_lat_same(v_av_lat, v_npc_lat)
        else:
            return phys_safe_lat_opp(v_av_lat, v_npc_lat)
        

    def _get_phys_reward(self):

        # Used only when solving AV-safe.

        ps_cfg = self.rl_config['phys_safe']
        INF_DIST = ps_cfg['INF_DIST']
        eps =ps_cfg['eps']
        DENOM_MIN = 0.05                  # Avoid a very small denominator.
        # npc, _ = self._get_closest_vehicle()
        npc, _ = self._get_closest_npc()
        if npc is None:
            return 0.0

        # -- Transform NPC pose and velocity into the ego body frame. --
        ego_tf = self.ego_vehicle.get_transform()
        npc_tf = npc.get_transform()

        # 1) Distance
        rel_w = np.array([npc_tf.location.x - ego_tf.location.x,
                        npc_tf.location.y - ego_tf.location.y])
        yaw_av = ego_tf.rotation.yaw
        yaw    = np.deg2rad(yaw_av)
        R      = np.array([[ np.cos(yaw),  np.sin(yaw)],
                        [-np.sin(yaw),  np.cos(yaw)]])
        dx, dy = R @ rel_w                         # lon, lat Distance

        # 2) Velocity components
        # v_av_w  = np.array([*self.ego_vehicle.get_velocity()[:2]])
        # v_npc_w = np.array([*npc.get_velocity()[:2]])

        v_av_vec  = self.ego_vehicle.get_velocity()
        v_npc_vec = npc.get_velocity()

        v_av_w  = np.array([v_av_vec.x,  v_av_vec.y])
        v_npc_w = np.array([v_npc_vec.x, v_npc_vec.y])
        v_av, v_npc = R @ v_av_w, R @ v_npc_w      # lon/lat
        v_av_lon, v_av_lat   = v_av
        v_npc_lon, v_npc_lat = v_npc
        yaw_npc = npc_tf.rotation.yaw

        # Step 3: compute safety distances
        d_safe_lon = self.safe_lon_distance(v_av_lon, v_npc_lon,
                                    yaw_av, yaw_npc)

        d_safe_lat = self.safe_lat_distance(v_av_lat, v_npc_lat)

        # A very large required safety distance means the state is unsafe or physically unrealistic.
        if d_safe_lat >= INF_DIST * 0.9 or d_safe_lon >= INF_DIST * 0.9:
            return -1.0
        # A near-zero safety distance means the state is effectively safe.
        if d_safe_lon <= eps and d_safe_lat <= eps:
            return 0                  # Effectively safe; no reward.
        
        # 4) Compute margins Δ  (>0 ⇒ safe region;<0 ⇒ danger region)
        Δx = abs(dx) - d_safe_lon      # Longitudinal margin
        Δy = abs(dy) - d_safe_lat      # Lateral margin

        # If one axis is safe, evaluate only the other axis.
        if Δx > 0 and Δy > 0:
            return 0.0                 # Both axes are safe -> score 0.
        elif Δx > 0:                   # Longitudinally safe; evaluate lateral only.
            norm = Δy / (d_safe_lat + eps)
        elif Δy > 0:                   # Laterally safe; evaluate longitudinal only.
            norm = Δx / (d_safe_lon + eps)
        else:                          # Both axes are unsafe; apply joint penalty.
            norm = -(np.sqrt((-Δx)/(d_safe_lon+eps))**2 + \
                np.sqrt((-Δy)/(d_safe_lat+eps))**2)

        # norm ∈ [0, ∞),0 means exactly at the boundary,>0 danger
        return norm                  # Higher norm means a larger penalty.


    def _get_phys_safe(self):
        """
        pooling to evaluate the phys safe
        σ = 1 - N_p,  N_p = ( (need_y_hat/dy_phys)^p + (need_x_hat/dx_phys)^p )^(1/p)
        Where:
        need_x = max(0, -Δx), Δx = d_x^realtime - d_x^phys_safe - W_car
        need_y = max(0, -Δy), Δy = d_y^realtime - d_y^phys_safe - L_car
        need_hat = max(0, need - l), l = ΔV * t + 0.5 * Δa * t^2
        Returns:
        float σ;Returns None for infinite or abnormal safety distances, which triggers the existing hard-penalty path.
        """
        ps = self.rl_config['phys_safe']
        INF = ps['INF_DIST']
        eps = ps['eps']
        p   = ps.get('p_norm', 2)
        ttc_eps = ps.get('ttc_eps', 1e-4)

        # Closest NPC
        npc, _ = self._get_closest_npc()
        if npc is None:
            return 0.0  # Open environment; treat as neutral sigma=0 so bars give zero reward.

        # === Coordinate transform into the ego body frame ===
        ego_tf = self.ego_vehicle.get_transform()
        npc_tf = npc.get_transform()
        rel_w = np.array([npc_tf.location.x - ego_tf.location.x,
                        npc_tf.location.y - ego_tf.location.y])
        yaw = np.deg2rad(ego_tf.rotation.yaw)
        R = np.array([[ np.cos(yaw),  np.sin(yaw)],
                    [-np.sin(yaw),  np.cos(yaw)]])
        dx, dy = R @ rel_w   # Signed longitudinal and lateral center distances

        v_av = self.ego_vehicle.get_velocity()
        v_np = npc.get_velocity()
        v_av_w  = np.array([v_av.x,  v_av.y])
        v_npc_w = np.array([v_np.x,  v_np.y])
        v_av_b, v_npc_b = R @ v_av_w, R @ v_npc_w
        vax, vay = v_av_b
        vnx, vny = v_npc_b

        # === Physical safety distances, step 1===
        yaw_av  = ego_tf.rotation.yaw
        yaw_npc = npc_tf.rotation.yaw
        d_phys_x = self.safe_lon_distance(vax, vnx, yaw_av, yaw_npc)
        d_phys_y = self.safe_lat_distance(vay, vny)

        # Unsatisfiable; return None so the hard penalty is applied later.
        if d_phys_x >= 0.9*INF or d_phys_y >= 0.9*INF:
            return None

        # === Δx, Δy(step 2)===
        # d^realtime uses absolute center distance; longitudinal subtracts car width and lateral subtracts car length, matching W_car/L_car in the slides.
        bb_av  = self.ego_vehicle.bounding_box.extent   # half-length/half-width
        bb_npc = npc.bounding_box.extent
        L_av,  W_av  = float(bb_av.x),  float(bb_av.y)
        L_npc, W_npc = float(bb_npc.x), float(bb_npc.y)
        dyaw = np.deg2rad((yaw_npc - yaw_av + 180) % 360 - 180)

        # previous
        c, s = abs(np.cos(dyaw)), abs(np.sin(dyaw))
        # # === Δx, Δy(step 2)===
        # # Subtract physical safety distance plus geometric buffer from center distance.
        # delta_x = abs(dx) - d_phys_x - clear_x_geom
        # delta_y = abs(dy) - d_phys_y - clear_y_geom

        # === need_x/need_y(step 3)===
        # need_x = max(0.0, -delta_x)
        # need_y = max(0.0, -delta_y)

        # Center-distance projections on four axes
        d_e_x  = abs(dx)                          # ego-x
        d_e_y  = abs(dy)                          # ego-y
        d_n_xp = abs(dx * np.cos(dyaw) + dy * np.sin(dyaw))   # npc-x'
        d_n_yp = abs(-dx * np.sin(dyaw) + dy * np.cos(dyaw))  # npc-y'

        # Geometric clearances on four axes, using SAT support width.
        clear_e_x  = L_av + c * L_npc + s * W_npc
        clear_e_y  = W_av + c * W_npc + s * L_npc
        clear_n_xp = L_npc + c * L_av  + s * W_av
        clear_n_yp = W_npc + c * W_av  + s * L_av

        # Physical safety distance on each axis, conservatively mixed by alignment weights.
        def d_phys_on(u_x, u_y):
            return abs(u_x) * d_phys_x + abs(u_y) * d_phys_y

        dphys_e_x  = d_phys_on(1.0, 0.0)
        dphys_e_y  = d_phys_on(0.0, 1.0)
        dphys_n_xp = d_phys_on(np.cos(dyaw), np.sin(dyaw))
        dphys_n_yp = d_phys_on(-np.sin(dyaw), np.cos(dyaw))

        # Gaps on four axes
        need_e_x  = max(0.0, -(d_e_x  - clear_e_x  - dphys_e_x))
        need_e_y  = max(0.0, -(d_e_y  - clear_e_y  - dphys_e_y))
        need_n_xp = max(0.0, -(d_n_xp - clear_n_xp - dphys_n_xp))
        need_n_yp = max(0.0, -(d_n_yp - clear_n_yp - dphys_n_yp))

        # Aggregate into two axis needs using the tighter value.
        need_x = max(need_e_x, need_n_xp)   # long
        need_y = max(need_e_y, need_n_yp)   # lat


        # If both axes are safe (need=0), sigma directly returns a positive margin.:N_p=0 -> σ=1
        if need_x == 0.0 and need_y == 0.0:
            return 1.0

        # === Closing check and t_x/t_y, step 4===
        # Rate of change for |dx|: d|dx|/dt = sign(dx)*(v_npc_x - v_av_x)
        # Closing condition: sign(dx)*(v_av_x - v_npc_x) > 0
        def closing_and_relspd(d, va, vn):
            s = 1.0 if d >= 0 else -1.0
            rel_closing = s*(va - vn)   # >0 means approaching
            return rel_closing > 0.0, max(0.0, rel_closing)

        is_close_x, dVx = closing_and_relspd(dx, vax, vnx)
        is_close_y, dVy = closing_and_relspd(dy, vay, vny)

        if not is_close_x and (abs(dx) > max(clear_n_xp, clear_e_x)):
            need_x = 0.0
        if not is_close_y and (abs(dy) > max(clear_n_yp, clear_e_y)):
            need_y = 0.0

        # Required compensation time, meaningful only while closing.
        t_x = need_x / max(dVx, ttc_eps) if is_close_x and need_x > 0 else float('inf')
        t_y = need_y / max(dVy, ttc_eps) if is_close_y and need_y > 0 else float('inf')

        # Use the joint time(t = min(t_x, t_y))
        t = min(t_x, t_y)
        if not np.isfinite(t):  # Neither axis is closing, or no compensation is needed.
            t = 0.0

        # === Compensation distances l_x/l_y, step 5===
        # Delta a: equivalent maximum deceleration toward reducing closing speed, conservatively combining AV and NPC.
        # a_lon_eq = self.rl_config['phys_safe']['a_lon_dec_av'] + self.rl_config['phys_safe']['a_lon_dec_npc']
        # a_lat_eq = self.rl_config['phys_safe']['a_lat_dec_av'] + self.rl_config['phys_safe']['a_lat_dec_npc']

        a_lon_eq = self.rl_config['phys_safe']['a_lon_dec_av']
        a_lat_eq = self.rl_config['phys_safe']['a_lat_dec_av']

        
        # l_x = (dVx * t + 0.5 * a_lon_eq * t**2) if is_close_x else 0.0
        # l_y = (dVy * t + 0.5 * a_lat_eq * t**2) if is_close_y else 0.0
        l_x = ( 0.5 * a_lon_eq * t**2) if is_close_x else 0.0
        l_y = ( 0.5 * a_lat_eq * t**2) if is_close_y else 0.0

        # Rule: do not compensate the dangerous axis because it already reflects maximum-braking distance; compensate only the orthogonal axis.
        # if t_x < t_y:          # x axis is more dangerous
        #     l_x = 0.0
        #     l_y = 0.5 * a_lat_eq * (t ** 2) if is_close_y else 0.0
        # elif t_y < t_x:        # y axis is more dangerous
        #     l_x = 0.5 * a_lon_eq * (t ** 2) if is_close_x else 0.0
        #     l_y = 0.0
        # else:                  # t_x == t_y: tied danger, so treat both axes as dangerous and do not compensate.
        #     l_x = 0.0
        #     l_y = 0.0


        # === Margin, step 6===
        need_x_hat = max(0.0, need_x - l_x)
        need_y_hat = max(0.0, need_y - l_y)

        # === p-norm normalization + σ(steps 7-8)===
        nx = need_x_hat / max(max(dphys_e_x  + clear_e_x,  dphys_n_xp + clear_n_xp), eps)
        ny = need_y_hat / max(max(dphys_e_y  + clear_e_y,  dphys_n_yp + clear_n_yp), eps)
        Np = (nx**p + ny**p)**(1.0/p)

        sigma = 1.0 - Np
        return sigma


    def _get_adv_reward(self):
        npc, dist = self._get_closest_npc()
        if npc is None:
            return 0.0

        adv = self.rl_config['adv_reward']

        # --- Relative quantities ---
        rel = npc.get_transform().location - self.ego_vehicle.get_transform().location
        vrel = npc.get_velocity() - self.ego_vehicle.get_velocity()
        dist = max(np.hypot(rel.x, rel.y), 1e-3)

        # Keep only the positive approaching velocity component; non-approaching speed is 0.
        closing_speed = max(0.0, - (rel.x*vrel.x + rel.y*vrel.y) / dist)

        # Optional front-object filter; currently any approaching target, including rear-end cases, contributes.
        # yaw = np.deg2rad(self.ego_vehicle.get_transform().rotation.yaw)
        # fwd = np.array([np.cos(yaw), np.sin(yaw)])
        # ahead = (rel.x*fwd[0] + rel.y*fwd[1]) > 0
        # if not ahead:
        #     closing_speed = 0.0

        # 1) TTC reward, valid only while approaching.
        if closing_speed > 0:
            ttc = dist / closing_speed
            r_ttc = 1.0 / (1.0 + ttc / adv['ttc_horizon'])
        else:
            r_ttc = 0.0

        return adv['w_ttc']*r_ttc

    


    
    # AV-safe solution
    def _risk_feature(self, npc, dist):
        """
        Returns a 6D feature vector:
        - ego   : 4D native state        (lat_err, yaw_err, speed, veh_front)
        - inv_d : 1/(dist+1)         -- larger when distance is smaller ∈ (0,1]
        - close : tanh(closing /10)  -- normalized relative closing speed ∈ (-1,1)
        """
        # # ---- 4D native state ----
        # ego = self._get_obs()['state'][:4]          # lat_err, yaw_err, speed, veh_front
        ego = self._cached_state                # Cached in _get_obs.
        # # ---- Two additional scalars (dist, closing_speed) ----
        # v_rel = npc.get_velocity() - self.ego_vehicle.get_velocity()
        # rel   = npc.get_transform().location - self.ego_vehicle.get_transform().location
        # closing = (v_rel.x*rel.x + v_rel.y*rel.y) / max(dist,1e-3)
        # return np.array([*ego, dist, closing], dtype=np.float32)   # 6D
        if npc is None:
            inv_d = 0.0
            closing_norm = 0.0
        else:
            inv_d = 1.0 / (dist + 1.0)
            v_rel = npc.get_velocity() - self.ego_vehicle.get_velocity()
            rel   = npc.get_transform().location - self.ego_vehicle.get_transform().location
            # Approaching is negative.
            closing = (v_rel.x*rel.x + v_rel.y*rel.y) / max(dist,1e-3)
            closing_norm = np.tanh(closing / 10.0)
        return np.concatenate([ego, [inv_d, closing_norm]]).astype(np.float32)
    
    #Original AV reward
    # def _get_av_reward(self):
    #     npc, dist = self._get_closest_vehicle()
    #     if npc is None:
    #         return 0.0
    #     feat  = self._risk_feature(npc, dist)
    #     risk  = av_safe.risk(feat)
    #     return risk

    

    def _get_av_safe(self):
        # npc, dist = self._get_closest_vehicle()
        npc, dist = self._get_closest_npc()
        if npc is None:
            return 0.0
        feat  = self._risk_feature(npc, dist)
        risk  = av_safe.risk(feat)
        return min(1,risk)
    

        
    def _fmt_arr(slef, a):
        a = np.asarray(a).reshape(-1)
        return "[" + ", ".join(f"{x:.3f}" for x in a) + "]"

        
    def _get_scenario_moreward_vec(self):

        # npc, _ = self._get_closest_vehicle()
        npc, _ = self._get_closest_npc()
        if npc is None:
            print('[Reward] No NPC found, returning 0.0')
            return np.array([0.0, 0.0], dtype=np.float32)
        
        
        sigma = self._get_phys_safe()

        if av_safe._enabled and av_safe._training and not av_safe.ready():
            r_adv = float(self._get_adv_reward())
            sigma_val = -1.0 if sigma is None else float(sigma)
            npc_moreward = np.array([r_adv, sigma_val], dtype=np.float32)
            # sigma_str = f"{sigma_val:.3f}" if sigma is not None else "NaN"
            # print(f"[Reward] adv: {r_adv:.3f}, sigma: {sigma_str}, npc_moreward: {self._fmt_arr(npc_moreward)}")
            return npc_moreward

        risk = float(self._get_av_safe())

        # if sigma is None or sigma < 0.0:
        if sigma is None:
            npc_moreward = np.array([risk, -1], dtype=np.float32)  
        else:
            # sigma always 1, so it signal is low
            sigma = float(sigma)
            npc_moreward = np.array([risk, sigma], dtype=np.float32)

        sigma_str = f"{sigma:.3f}" if sigma is not None else "NaN"
        np_str = f"{1-sigma:.3f}" if sigma is not None else "NaN"
        # Scenario-training only; NPC reward can be disabled while training AV-safe.
        print(f"[Reward] risk: {risk:.3f}, sigma: {sigma_str}, np: {np_str}, npc_moreward: {self._fmt_arr(npc_moreward)}")
        return npc_moreward
    


    # def _terminal(self):
    #     return not self.scenario_manager._running 
        # ---------- 1) Termination flag, also cached for AV-safe ----------
    def _terminal(self):
        self._term_flag = not self.scenario_manager._running
        return self._term_flag

    def _remove_sensor(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None



    def _remove_ego(self):
        # TODO: ego can be reused.
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def clean_up(self):
        self._remove_sensor()
        if self.scenario_category != 'scenic':
            self._remove_ego()
        self.scenario_manager.clean_up()
