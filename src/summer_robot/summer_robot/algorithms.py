#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque
from typing import List, Optional, Tuple

import cv2
from nav_msgs.msg import OccupancyGrid
import numpy as np


class MapData:
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.width      = occupancy_grid.info.width
        self.height     = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution
        self.origin_x   = occupancy_grid.info.origin.position.x
        self.origin_y   = occupancy_grid.info.origin.position.y
        self.data = np.array(
            occupancy_grid.data, 
            dtype=np.int16
        ).reshape(self.height, self.width)

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        wx = mx * self.resolution + self.origin_x + self.resolution / 2.0
        wy = my * self.resolution + self.origin_y + self.resolution / 2.0
        return wx, wy

    def is_free(self, mx: int, my: int) -> bool:
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return False
        return 0 <= self.data[my, mx] < 50

    def get_inflated_map(self, inflation_m: float) -> np.ndarray:
        inflation_cells = max(1, int(inflation_m / self.resolution))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        occupied_mask = (self.data >= 50).astype(np.uint8)
        inflated_occupied = cv2.dilate(occupied_mask, kernel)
        unknown_mask = (self.data < 0).astype(np.uint8)
        return (inflated_occupied | unknown_mask).astype(bool)


class FrontierDetector:
    def __init__(
        self, 
        min_size: int = 3,
        search_radius: int = 800,
        ignore_radius: float = 1.5,        # 稍微放大死巷黑名單半徑，強迫遠離失敗點
        min_distance_to_robot: float = 0.6  # 稍微調降，允許更靈活的近距離轉折
    ):
        self.min_size = min_size
        self.search_radius = search_radius
        self.ignore_radius = ignore_radius
        self.min_dist = min_distance_to_robot

    def detect(
        self,
        map_data: MapData,
        robot_mx: int,
        robot_my: int,
        robot_yaw: float = 0.0,                    
        current_target: Optional[Tuple[float, float]] = None, 
        visited_list: Optional[List[Tuple[float, float]]] = None,
        unreachable_list: Optional[List[Tuple[float, float]]] = None
    ) -> List[Tuple[float, float]]:
        if visited_list is None:
            visited_list = []
        if unreachable_list is None:
            unreachable_list = []

        ignored_targets = visited_list + unreachable_list
        robot_wx, robot_wy = map_data.map_to_world(robot_mx, robot_my)

        grid = map_data.data
        h, w = map_data.height, map_data.width

        if not (0 <= robot_mx < w and 0 <= robot_my < h):
            return []

        # -------------------------------------------------------------
        # 1. 水滴擴散法 (BFS)
        # -------------------------------------------------------------
        visited_cells = np.zeros((h, w), dtype=bool)
        frontier_points = []

        start_x, start_y = robot_mx, robot_my
        if not (0 <= grid[start_y, start_x] < 50):
            found_free = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = start_x + dx, start_y + dy
                    if 0 <= nx < w and 0 <= ny < h and (0 <= grid[ny, nx] < 50):
                        start_x, start_y = nx, ny
                        found_free = True
                        break
                if found_free:
                    break
            if not found_free:
                return []

        queue = deque([(start_x, start_y, 0.0)])
        visited_cells[start_y, start_x] = True

        directions = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        while queue:
            cx, cy, cur_dist = queue.popleft()
            if cur_dist > self.search_radius:
                continue

            for dx, dy, step_cost in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited_cells[ny, nx]:
                    visited_cells[ny, nx] = True
                    val = grid[ny, nx]

                    if val < 0:
                        frontier_points.append((cx, cy, cur_dist * map_data.resolution))
                    elif 0 <= val < 50:
                        queue.append((nx, ny, cur_dist + step_cost))

        if not frontier_points:
            return []

        # -------------------------------------------------------------
        # 2. 智慧代價評分（優化取樣與黑名單過濾）
        # -------------------------------------------------------------
        candidates = []
        r = 10

        # 改進：動態採樣。如果總點數不多就不跳過，確保狹窄通道不會漏點
        step_sample = 2 if len(frontier_points) < 100 else 4

        for fx, fy, path_dist in frontier_points[::step_sample]:
            wx, wy = map_data.map_to_world(fx, fy)

            # 防線 1：排除車底過近目標
            dist_to_robot = math.hypot(wx - robot_wx, wy - robot_wy)
            if dist_to_robot < self.min_dist:
                continue

            # 防線 2：嚴格排除黑名單（特別是 unreachable 失敗過的點）
            if any(math.hypot(wx - ig_x, wy - ig_y) < self.ignore_radius for ig_x, ig_y in ignored_targets):
                continue

            # 統計周遭未知面積
            y_min, y_max = max(0, fy - r), min(h, fy + r + 1)
            x_min, x_max = max(0, fx - r), min(w, fx + r + 1)
            unknown_gain = np.sum(grid[y_min:y_max, x_min:x_max] < 0)

            if unknown_gain < 3:
                continue

            # 轉向懲罰
            target_angle = math.atan2(wy - robot_wy, wx - robot_wx)
            angle_diff = abs(math.atan2(math.sin(target_angle - robot_yaw), math.cos(target_angle - robot_yaw)))
            heading_penalty = angle_diff * 1.2 

            # 目標黏滯性
            stickiness_bonus = 0.0
            if current_target is not None:
                if math.hypot(wx - current_target[0], wy - current_target[1]) < 1.2:
                    stickiness_bonus = -4.0  # 稍微加深黏滯權重，避免在途中無故甩掉好目標

            cost = path_dist + heading_penalty - (unknown_gain * 0.05) + stickiness_bonus
            candidates.append((wx, wy, cost))

        if not candidates:
            return []

        candidates.sort(key=lambda p: p[2])
        return [(p[0], p[1]) for p in candidates]

    @staticmethod
    def find_nearest_safe_free_goal(
        map_data: MapData,
        fx: float,
        fy: float,
        robot_wx: float,
        robot_wy: float,
        inflated_map: np.ndarray,
        max_search_radius: int = 250, # 稍微擴大安全搜尋範圍
        min_goal_dist: float = 0.5
    ) -> Optional[Tuple[float, float]]:
        start_mx, start_my = map_data.world_to_map(fx, fy)

        if (0 <= start_mx < map_data.width and 
            0 <= start_my < map_data.height and 
            map_data.is_free(start_mx, start_my) and 
            not inflated_map[start_my, start_mx]
        ):
            wx, wy = map_data.map_to_world(start_mx, start_my)
            if math.hypot(wx - robot_wx, wy - robot_wy) >= min_goal_dist:
                return wx, wy

        visited = set()
        queue = deque([(start_mx, start_my)])
        visited.add((start_mx, start_my))

        while queue:
            cx, cy = queue.popleft()

            if abs(cx - start_mx) > max_search_radius or abs(cy - start_my) > max_search_radius:
                continue
            if not (0 <= cx < map_data.width and 0 <= cy < map_data.height):
                continue

            if map_data.is_free(cx, cy) and not inflated_map[cy, cx]:
                wx, wy = map_data.map_to_world(cx, cy)
                if math.hypot(wx - robot_wx, wy - robot_wy) >= min_goal_dist:
                    return wx, wy

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                            (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ncx, ncy = cx + dx, cy + dy
                if (ncx, ncy) not in visited:
                    visited.add((ncx, ncy))
                    queue.append((ncx, ncy))

        return None
