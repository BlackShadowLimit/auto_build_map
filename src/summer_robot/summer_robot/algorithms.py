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
    水滴擴散法 (Wavefront BFS) + 資訊增益代價評分 + 雙階段 Fallback 機制。
    天然繞牆，優先探索開闊大廳，全圖探完後自動回頭補掃小角落。
    """
    def __init__(
        self, 
        min_size: int = 4,
        search_radius: int = 800,
        ignore_radius: float = 0.8,
        min_distance_to_robot: float = 0.6  # 防原地抽搐：太近的不選
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
        h, w = map_data.height, map_data.width

        if not (0 <= robot_mx < w and 0 <= robot_my < h):
            return []

        # -------------------------------------------------------------
        # 1. 水滴擴散法 (BFS)：從機器人出發找可連通邊界
        # -------------------------------------------------------------
        visited_cells = np.zeros((h, w), dtype=bool)
        frontier_points = []

        start_x, start_y = robot_mx, robot_my
        # 若機器人目前位置貼牆，周遭 3x3 尋找最近的安全空地當作水滴起點
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
                        # 碰到未知區，記錄此邊界點與真實繞牆步數
                        frontier_points.append((cx, cy, cur_dist * map_data.resolution))
                    elif 0 <= val < 50:
                        # 安全空地，水滴繼續流
                        queue.append((nx, ny, cur_dist + step_cost))

        if not frontier_points:
            return []

        # -------------------------------------------------------------
        # 2. 智慧代價評分 + 雙階段 Fallback 分流
        # -------------------------------------------------------------
        primary_candidates = []   # 第一輪：優先探勘的大未知區域
        fallback_candidates = []  # 第二輪：備用補漏的小碎屑或柱子陰影

        r = 12  # 25x25 的檢測視野

        for fx, fy, path_dist in frontier_points[::4]:
            wx, wy = map_data.map_to_world(fx, fy)

            # 防線 1：排除腳底下 (< 0.6m) 的點
            dist_to_robot = math.hypot(wx - robot_wx, wy - robot_wy)
            if dist_to_robot < self.min_dist:
                continue

            # 防線 2：排除黑名單（已走過或到不了）
            if any(math.hypot(wx - ig_x, wy - ig_y) < self.ignore_radius for ig_x, ig_y in ignored_targets):
                continue

            # 統計該邊界點周遭的未知區域大小 (Information Gain)
            y_min, y_max = max(0, fy - r), min(h, fy + r + 1)
            x_min, x_max = max(0, fx - r), min(w, fx + r + 1)
            local_patch = grid[y_min:y_max, x_min:x_max]
            unknown_gain = np.sum(local_patch < 0)

            # 綜合代價計算：距離成本 - (未知面積 * 權重 0.08)
            cost = path_dist - (unknown_gain * 0.08)

            # 雙階段分流：大開闊區優先，不足 12 格但超過 3 格的列為補漏備用
            if unknown_gain >= 12:
                primary_candidates.append((wx, wy, cost))
            elif unknown_gain >= 3:
                fallback_candidates.append((wx, wy, cost))

        # 優先回傳大空間目標；如果找不到了，自動無縫切換去補掃小細節
        targets = primary_candidates if primary_candidates else fallback_candidates

        if not targets:
            return []

        # 依照 Cost 由低到高排序
        targets.sort(key=lambda p: p[2])
        return [(p[0], p[1]) for p in targets]

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
        防線 3：將 Frontier 往安全自由空地退縮，避免目標在膨脹層內或貼著車底。
        """
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
