''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 17:32:41
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_helper import get_crossing_point, get_junction_topology


def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
        Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.x = trigger_location.x + 20
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
        Gets the driving / parking lane that is most to the right of the waypoint as well as the number of lane changes done
    """
    lane_changes = 0
    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next
    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
        This function filters false negative Shoulder which are in reality Parking lanes.
        These are differentiated from the others because, similar to the driving lanes,
        they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True
    return False


class VehicleTurningRoute(BasicScenario):
    """
        The ego vehicle is passing through a road and encounters a cyclist after taking a turn. 
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute-Init-State", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.running_distance = 10
        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 500
        self.ego_max_driven_distance = 180
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

    def convert_actions(self, actions, x_scale, y_scale, x_mean, y_mean):

        x = x_mean
        y = y_mean
        yaw = yaw_mean
        dist = dist_mean
        return [x, y, yaw, dist]

    def convert_actions(self, actions):
        base_speed = 5.0
        speed_scale = 5.0
        speed = actions[0] * speed_scale + base_speed
        return speed

    # def initialize_actors(self):
    #     cross_location = get_crossing_point(self.ego_vehicle)
    #     cross_waypoint = CarlaDataProvider.get_map().get_waypoint(cross_location)
    #     entry_wps, exit_wps = get_junction_topology(cross_waypoint.get_junction())
    #     assert len(entry_wps) == len(exit_wps)
    #     x_mean = y_mean = 0
    #     max_x_scale = max_y_scale = 0
    #     for i in range(len(entry_wps)):
    #         x_mean += entry_wps[i].transform.location.x + exit_wps[i].transform.location.x
    #         y_mean += entry_wps[i].transform.location.y + exit_wps[i].transform.location.y
    #     x_mean /= len(entry_wps) * 2
    #     y_mean /= len(entry_wps) * 2
    #     for i in range(len(entry_wps)):
    #         max_x_scale = max(max_x_scale, abs(entry_wps[i].transform.location.x - x_mean), abs(exit_wps[i].transform.location.x - x_mean))
    #         max_y_scale = max(max_y_scale, abs(entry_wps[i].transform.location.y - y_mean), abs(exit_wps[i].transform.location.y - y_mean))
    #     max_x_scale *= 0.8
    #     max_y_scale *= 0.8

    #     x = x_mean
    #     y = y_mean
    #     yaw = 180
    #     other_actor_transform = carla.Transform(carla.Location(x, y, 0), carla.Rotation(yaw=yaw))
        
    #     self.actor_transform_list = [other_actor_transform]
    #     self.actor_type_list = ['vehicle.diamondback.century']
    #     self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
    #     self.reference_actor = self.other_actors[0] # used for triggering this scenario

    def _get_crossing_point(self, actor):
        """
        Get the next crossing point location in front of the ego vehicle.

        NOTE:
        - If the ego is already inside a junction, we first move forward until
          we leave the current junction, and then search for the NEXT one.
        """
        amap = CarlaDataProvider.get_map()
        wp_cross = amap.get_waypoint(actor.get_location())

        # Move forward slightly to avoid unstable topology near the start point.
        ahead = wp_cross.next(2.0)
        if ahead:
            wp_cross = ahead[0]

        # If already inside an intersection, first move out of it.
        safe_iter = 0
        while wp_cross.is_intersection and safe_iter < 50:
            next_wps = wp_cross.next(2.0)
            if not next_wps:
                break
            wp_cross = next_wps[0]
            safe_iter += 1

        # After leaving the current intersection, search for the next one.
        safe_iter = 0
        while not wp_cross.is_intersection and safe_iter < 200:
            next_wps = wp_cross.next(2.0)
            if not next_wps:
                break
            wp_cross = next_wps[0]
            safe_iter += 1

        crossing = carla.Location(
            x=wp_cross.transform.location.x,
            y=wp_cross.transform.location.y,
            z=wp_cross.transform.location.z
        )
        return crossing


    def _route_locs(self):
        route = CarlaDataProvider.get_ego_vehicle_route()
        if not route: 
            return None
        locs = []
        for item in route:
            node = item[0] if isinstance(item, (list, tuple)) else item
            if hasattr(node, "location"):          # Transform
                locs.append(node.location)
            elif hasattr(node, "transform"):       # Waypoint
                locs.append(node.transform.location)
            else:                                  # Location
                locs.append(node)
        return locs

    def _closest_idx(self, locs, ref):
        return min(range(len(locs)), key=lambda i: (locs[i].x-ref.x)**2+(locs[i].y-ref.y)**2)

    def _dist_point_to_seg(self, px, py, ax, ay, bx, by):
        # Point-to-segment distance in 2D
        vx, vy = bx-ax, by-ay
        wx, wy = px-ax, py-ay
        vv = vx*vx + vy*vy
        if vv < 1e-9:
            # Degenerate segment treated as a point
            dx, dy = px-ax, py-ay
            return (dx*dx + dy*dy) ** 0.5
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
        cx, cy = ax + t*vx, ay + t*vy
        dx, dy = px - cx, py - cy
        return (dx*dx + dy*dy) ** 0.5

    def _min_dist_cross_to_route(self, start_loc, dir_vec, locs, idx_center, window=8, L=28.0, step=1.0):
        # Sample the pedestrian crossing line and compute its minimum distance to route segments.
        # dir_vec is a unit vector; yaw=180 points toward (-1, 0).
        pts = []
        t = 0.0
        while t <= L:
            pts.append((start_loc.x + dir_vec[0]*t, start_loc.y + dir_vec[1]*t))
            t += step
        j0 = max(0, idx_center - window)
        j1 = min(len(locs)-2, idx_center + window)
        mind = 1e9
        for (px, py) in pts:
            for j in range(j0, j1+1):
                a, b = locs[j], locs[j+1]
                d = self._dist_point_to_seg(px, py, a.x, a.y, b.x, b.y)
                if d < mind:
                    mind = d
        return mind

    def _scan_to_sidewalk(self, amap, base, yaw, sign, start, maxd=12.0, step=0.5):
        # Scan along the chosen side until Sidewalk or Shoulder is found.
        d = start
        while d <= maxd:
            probe = self._move(base, yaw_deg=yaw, right_m=sign*d)
            wp = amap.get_waypoint(probe, project_to_road=False, lane_type=carla.LaneType.Any)
            if wp and wp.lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder):
                return d
            d += step
        return min(maxd, d)



    def _move(self, loc, yaw_deg, forward_m=0.0, right_m=0.0):
        r  = math.radians(yaw_deg)
        fx, fy = math.cos(r), math.sin(r)
        # Right normal = yaw - 90 degrees.
        rx, ry = math.cos(r - math.pi/2.0), math.sin(r - math.pi/2.0)
        return carla.Location(loc.x + forward_m*fx + right_m*rx,
                            loc.y + forward_m*fy + right_m*ry,
                            loc.z)


    def initialize_actors(self):
        world = CarlaDataProvider.get_world()
        amap  = CarlaDataProvider.get_map()
        assert amap is not None, "Map not ready: CarlaDataProvider.get_map() returned None."

        cross_location = self._get_crossing_point(self.ego_vehicle)
        cross_waypoint = amap.get_waypoint(cross_location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)

        # Keep the original geometric-center logic.
        entry_wps, exit_wps = get_junction_topology(cross_waypoint.get_junction())
        assert len(entry_wps) == len(exit_wps)
        x_mean = y_mean = 0.0
        for i in range(len(entry_wps)):
            x_mean += entry_wps[i].transform.location.x + exit_wps[i].transform.location.x
            y_mean += entry_wps[i].transform.location.y + exit_wps[i].transform.location.y
        x_mean /= (len(entry_wps) * 2.0)
        y_mean /= (len(entry_wps) * 2.0)

        base = carla.Location(x_mean, y_mean, 0.35)
        yaw  = 180.0                                           


        d = cross_waypoint.lane_width  # Start offset
        LONG_BIAS_M = -8  # Push spawn point to one end of the crosswalk; use +8.0 for the other end.

        # Two candidate spawn points: offset once to the right (+1) and once to the left (-1).
        cand_R = self._move(base, yaw, forward_m=LONG_BIAS_M, right_m=+1.2*d)
        cand_L = self._move(base, yaw, forward_m=LONG_BIAS_M, right_m=-1.2*d)

        # Pedestrian walking direction, consistent with yaw=180.
        walk_dir = (-1.0, 0.0)

        # Use only route segments near the intersection.
        locs = self._route_locs()
        idxc = self._closest_idx(locs, base) if locs else 0

        # Evaluate which side intersects or approaches the AV trajectory more.
        dR = self._min_dist_cross_to_route(cand_R, walk_dir, locs, idxc)
        dL = self._min_dist_cross_to_route(cand_L, walk_dir, locs, idxc)
        pref_sign = +1 if dR <= dL else -1

        # Adaptively scan along the right normal until Sidewalk or Shoulder is found.
        
        SIDE_OFFSET_M = None
        while d <= 12.0:  # Scan up to 12 m.
            probe = self._move(base, yaw, right_m=d)   # _move uses right normal = yaw - 90 degrees.
            wp = amap.get_waypoint(probe, project_to_road=False, lane_type=carla.LaneType.Any)
            if wp and wp.lane_type in (carla.LaneType.Sidewalk, carla.LaneType.Shoulder):
                SIDE_OFFSET_M = d
                break
            d += 1
        if SIDE_OFFSET_M is None:
            SIDE_OFFSET_M = d

        

        spawn_loc = self._move(base, yaw, forward_m=LONG_BIAS_M, right_m=pref_sign *SIDE_OFFSET_M)
        spawn_loc.z = 0.3
        spawn_tf  = carla.Transform(spawn_loc, carla.Rotation(yaw=yaw))
        # print(f'[VehicleTurningRoute] spawn_loc = {spawn_loc}; [VehicleTurningRoute] base = {base}')
        # world.debug.draw_string(base, "base", False, carla.Color(255,255,0), 5.0)
        # world.debug.draw_string(spawn_tf.location, "NPC", False, carla.Color(0,255,0), 5.0)

        self.actor_transform_list = [spawn_tf]
        self.actor_type_list      = ['vehicle.diamondback.century']
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(
            self.actor_transform_list, self.actor_type_list
        )
        self.reference_actor = self.other_actors[0]




        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'

    def update_behavior(self, scenario_action):
        cur_actor_target_speed = self.convert_actions(scenario_action)
        self.scenario_operation.go_straight(cur_actor_target_speed, 0)

    def check_stop_condition(self):
        pass
