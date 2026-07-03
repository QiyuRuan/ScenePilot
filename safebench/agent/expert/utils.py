import numpy as np
import carla


def inverse_conversion_2d(point, translation, yaw):
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return rotation_matrix.T @ (point - translation)


def draw_route(world, vehicle=None, waypoint_route=None, life_time=0.11):
    if world is None or waypoint_route is None or len(waypoint_route) < 2:
        return
    color = carla.Color(0, 255, 255, 255)
    max_len = min(len(waypoint_route), 50)
    for i in range(max_len - 1):
        p0 = waypoint_route[i]
        p1 = waypoint_route[i + 1]
        a = carla.Location(float(p0[0]), float(p0[1]), 0.2)
        b = carla.Location(float(p1[0]), float(p1[1]), 0.2)
        world.debug.draw_line(a, b, thickness=0.08, color=color, life_time=life_time)
