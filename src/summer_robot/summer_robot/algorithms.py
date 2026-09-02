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
            dtype=np.int16).reshape(self.height, self.width)


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
        self, min_size: int = 8,
        search_radius: int = 600,
        ignore_radius: float = 1.0
    ):
        self.min_size = min_size
        self.search_radius = search_radius
        self.ignore_radius = ignore_radius
    

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

        grid = map_data.data
        free_mask = ((grid >= 0) & (grid < 50)).astype(np.uint8)
        unknown_mask = (grid < 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
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
            dist_cells = math.sqrt((cx - robot_mx) ** 2 + (cy - robot_my) ** 2)
            if dist_cells > self.search_radius:
                continue

            wx, wy = map_data.map_to_world(cx, cy)
        
            is_ignored = False
            for ig_x, ig_y in ignored_targets:
                if math.sqrt((wx - ig_x)**2 + (wy - ig_y)**2) < self.ignore_radius:
                    is_ignored = True
                    break
        
            if not is_ignored:
                candidates.append((wx, wy, dist_cells, area))

        if not candidates:
            return []

        candidates.sort(key=lambda p: (p[2], -p[3]))
        return [(p[0], p[1]) for p in candidates]


    @staticmethod
    def find_nearest_safe_free_goal(
        map_data: MapData,
        fx: float,
        fy: float,
        inflated_map: np.ndarray,
        max_search_radius: int = 200
    ) -> Optional[Tuple[float, float]]:
        start_mx, start_my = map_data.world_to_map(fx, fy)
        if (0 <= start_mx < map_data.width and 
            0 <= start_my < map_data.height and 
            map_data.is_free(start_mx, start_my) and 
            not inflated_map[start_my, start_mx]
        ):
            return map_data.map_to_world(start_mx, start_my)
        
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
                return map_data.map_to_world(cx, cy)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ncx, ncy = cx + dx, cy + dy
                if (ncx, ncy) not in visited:
                    visited.add((ncx, ncy))
                    queue.append((ncx, ncy))

        return None


