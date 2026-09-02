#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque
from typing import List, Optional, Tuple

import cv2
from nav_msgs.msg import OccupancyGrid
import numpy as np


class MapData:
    """
    負責封裝與處理 OccupancyGrid 地圖資料的類別。
    提供座標轉換 (world <-> map) 以及地圖狀態查詢等基礎功能。
    """
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
    """
    使用膨脹交集法偵測未探索邊界。
    """
    def __init__(
        self, 
        min_size: int = 4,
        search_radius: int = 800,
        ignore_radius: float = 0.8,
        min_distance_to_robot: float = 0.6  # 防卡死：太近的不選
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
        free_mask = ((grid >= 0) & (grid < 50)).astype(np.uint8)
        unknown_mask = (grid < 0).astype(np.uint8)

        # 形態學交集找出邊界線
        kernel = np.ones((5, 5), np.uint8)
        free_dilated = cv2.dilate(free_mask, kernel, iterations=1)
        frontier_raw = cv2.bitwise_and(free_dilated, unknown_mask)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_raw, connectivity=8
        )

        candidates = []
        for lid in range(1, num_labels):
            area = stats[lid, cv2.CC_STAT_AREA]
            if area < self.min_size:
                continue

            cx, cy = int(centroids[lid, 0]), int(centroids[lid, 1])
            dist_cells = math.hypot(cx - robot_mx, cy - robot_my)
            if dist_cells > self.search_radius:
                continue

            wx, wy = map_data.map_to_world(cx, cy)

            # 防線 1：邊界離機器人目前位置太近 (< 0.6m) 直接略過，不抽搐
            dist_to_robot = math.hypot(wx - robot_wx, wy - robot_wy)
            if dist_to_robot < self.min_dist:
                continue

            # 防線 2：黑名單過濾 (已訪問/無法到達)
            is_ignored = any(math.hypot(wx - ig_x, wy - ig_y) < self.ignore_radius for ig_x, ig_y in ignored_targets)
            if not is_ignored:
                candidates.append((wx, wy, dist_cells, area))

        if not candidates:
            return []

        # 優先選擇：距離近且面積大的目標
        candidates.sort(key=lambda p: (p[2], -p[3]))
        return [(p[0], p[1]) for p in candidates]

    @staticmethod
    def find_nearest_safe_free_goal(
        map_data: MapData,
        fx: float,
        fy: float,
        robot_wx: float,
        robot_wy: float,
        inflated_map: np.ndarray,
        max_search_radius: int = 200,
        min_goal_dist: float = 0.5
    ) -> Optional[Tuple[float, float]]:
        """
        尋找安全點，同時加上防線 3：確保安全點不會直接落在機器人腳底下 (< 0.5m)。
        """
        start_mx, start_my = map_data.world_to_map(fx, fy)

        # 檢查原始點是否可用且非腳底
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
                # 防線 3：安全點不能在車底
                if math.hypot(wx - robot_wx, wy - robot_wy) >= min_goal_dist:
                    return wx, wy

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ncx, ncy = cx + dx, cy + dy
                if (ncx, ncy) not in visited:
                    visited.add((ncx, ncy))
                    queue.append((ncx, ncy))

        return None
