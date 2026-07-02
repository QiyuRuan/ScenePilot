import os

import numpy as np
import pygame
import torch
import torch.nn.functional as F


PIXELS_PER_METER = 5


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter=PIXELS_PER_METER):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        if not pygame.get_init():
            pygame.init()
        # Do not override existing runner display (it breaks eval video layout).
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1), 0, 32)

        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)
        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.draw_road_map(
            self.big_map_surface,
            self.big_lane_surface,
            carla_map,
            self.world_to_pixel,
        )
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(self, map_surface, lane_surface, carla_map, world_to_pixel):
        map_surface.fill((0, 0, 0))
        lane_surface.fill((0, 0, 0))
        precision = 0.05

        def draw_lane_marking(surface, points, solid=True):
            if solid:
                pygame.draw.lines(surface, (255, 255, 255), False, points, 2)
            else:
                broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
                for line in broken_lines:
                    pygame.draw.lines(surface, (255, 255, 255), False, line, 2)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
            if w is None or w.road_id != waypoint.road_id:
                return True
            return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        for waypoint in topology:
            waypoints = [waypoint]
            nxt_list = waypoint.next(precision)
            if not nxt_list:
                continue
            nxt = nxt_list[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt_list = nxt.next(precision)
                if not nxt_list:
                    break
                nxt = nxt_list[0]

            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]

            if len(polygon) > 2:
                pygame.draw.polygon(map_surface, (255, 255, 255), polygon, 10)
                pygame.draw.polygon(map_surface, (255, 255, 255), polygon)

            if not waypoint.is_intersection:
                sample = waypoints[int(len(waypoints) / 2)]
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1),
                )
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1),
                )

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]


class DatagenBEVRenderer(object):
    def __init__(self, device, map_offset, map_dims, data_generation=False):
        self.device = device
        if data_generation:
            self.PIXELS_AHEAD_VEHICLE = 0
            self.crop_dims = (500, 500)
        else:
            self.PIXELS_AHEAD_VEHICLE = 110
            self.crop_dims = (192, 192)

        self.map_offset = map_offset
        self.map_dims = map_dims
        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0],
        )

    def world_to_pix(self, pos):
        return (pos - self.map_offset) * PIXELS_PER_METER

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / torch.tensor([self.map_dims[1], self.map_dims[0]], device=self.device)
        return pos_rel * 2 - 1

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw):
        crop_yaw = crop_yaw + np.pi / 2
        rotation = torch.tensor(
            [
                [torch.cos(crop_yaw), -torch.sin(crop_yaw)],
                [torch.sin(crop_yaw), torch.cos(crop_yaw)],
            ],
            device=self.device,
        )
        crop_pos_px = self.world_to_pix(crop_pos)
        query_pos_px_map = self.world_to_pix(query_pos)
        shift = torch.tensor([0.0, -self.PIXELS_AHEAD_VEHICLE], device=self.device)
        query_pos_px = rotation.T @ (query_pos_px_map - crop_pos_px) - shift
        return query_pos_px + torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2], device=self.device)

    def get_local_birdview(self, grid, position, orientation):
        position = self.world_to_rel(position)
        orientation = orientation + np.pi / 2

        scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0], [0, self.crop_scale[0], 0], [0, 0, 1]],
            device=self.device,
        ).view(1, 3, 3)
        rotation_transform = torch.tensor(
            [[torch.cos(orientation), -torch.sin(orientation), 0], [torch.sin(orientation), torch.cos(orientation), 0], [0, 0, 1]],
            device=self.device,
        ).view(1, 3, 3)

        shift = torch.tensor([0.0, -2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]], device=self.device)
        position = position + rotation_transform[0, 0:2, 0:2] @ shift
        translation_transform = torch.tensor(
            [
                [1, 0, position[0] / self.crop_scale[0]],
                [0, 1, position[1] / self.crop_scale[1]],
                [0, 0, 1],
            ],
            device=self.device,
        ).view(1, 3, 3)
        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (1, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )
        return F.grid_sample(grid, affine_grid, align_corners=True)

    def render_agent_bv(self, grid, grid_pos, grid_orientation, vehicle, position, orientation, channel):
        orientation = orientation + np.pi / 2
        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)
        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.device)
        pos_rel_bv = (pos_rel_bv * 2 - 1) * -1

        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.device)
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.device)
        scale_transform = torch.tensor(
            [[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]],
            device=self.device,
        ).view(1, 3, 3)

        grid_orientation = grid_orientation + np.pi / 2
        rotation_transform = torch.tensor(
            [
                [torch.cos(orientation - grid_orientation), torch.sin(orientation - grid_orientation), 0],
                [-torch.sin(orientation - grid_orientation), torch.cos(orientation - grid_orientation), 0],
                [0, 0, 1],
            ],
            device=self.device,
        ).view(1, 3, 3)
        translation_transform = torch.tensor(
            [[1, 0, pos_rel_bv[0]], [0, 1, pos_rel_bv[1]], [0, 0, 1]],
            device=self.device,
        ).view(1, 3, 3)
        affine_transform = scale_transform @ rotation_transform @ translation_transform
        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],
            (1, 1, grid.shape[2], grid.shape[3]),
            align_corners=True,
        )
        vehicle_rendering = F.grid_sample(vehicle, affine_grid, align_corners=True)
        grid[:, channel, ...] += vehicle_rendering.squeeze()
