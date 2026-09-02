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
        # 將 1D 資料轉換為 2D 矩陣
        self.data = np.array(
            occupancy_grid.data, 
            dtype=np.int16).reshape(self.height, self.width)

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        """將實際世界座標 (m) 轉換為地圖陣列索引 (grid/pixel)"""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """將地圖陣列索引 (grid/pixel) 轉換為實際世界座標 (m)，取該格子的中心點"""
        wx = mx * self.resolution + self.origin_x + self.resolution / 2.0
        wy = my * self.resolution + self.origin_y + self.resolution / 2.0
        return wx, wy

    def is_free(self, mx: int, my: int) -> bool:
        """檢查特定格子是否為已知空地 (Occupancy值在 0~49 之間)"""
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return False
        return 0 <= self.data[my, mx] < 50

    def get_inflated_map(self, inflation_m: float) -> np.ndarray:
        """
        回傳一張膨脹過的地圖遮罩 (boolean 陣列)。
        將障礙物 (>=50) 膨脹指定的半徑，同時把未知區域 (<0) 也標記為不可行走。
        此遮罩用於導航點的安全篩選。
        """
        # 計算膨脹半徑對應的格子數
        inflation_cells = max(1, int(inflation_m / self.resolution))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        
        # 標記障礙物並進行膨脹操作
        occupied_mask = (self.data >= 50).astype(np.uint8)
        inflated_occupied = cv2.dilate(occupied_mask, kernel)
        
        # 標記未知區域
        unknown_mask = (self.data < 0).astype(np.uint8)

        # 兩者聯集，True 表示危險區域 (膨脹後的障礙物或未知區域)
        return (inflated_occupied | unknown_mask).astype(bool)


class FrontierDetector:
    """
    負責在地圖中尋找未探索邊界 (Frontier) 的類別。
    使用影像處理的方式 (侵蝕膨脹與連通元件分析) 快速找出邊界群集。
    """
    def __init__(
        self, min_size: int = 8,
        search_radius: int = 600,
        ignore_radius: float = 1.0
    ):
        self.min_size = min_size            # 邊界群集的最小面積 (過濾雜訊)
        self.search_radius = search_radius  # 搜尋半徑 (格數)，過遠的邊界不考慮
        self.ignore_radius = ignore_radius  # 黑名單 (已訪問/無法到達) 的忽略半徑 (m)
    
    def detect(
        self,
        map_data: MapData,
        robot_mx: int,
        robot_my: int,
        visited_list: Optional[List[Tuple[float, float]]] = None,
        unreachable_list: Optional[List[Tuple[float, float]]] = None
    ) -> List[Tuple[float, float]]:
        """
        掃描地圖，找出距離機器人最近的有效邊界點列表。
        """
        if visited_list is None:
            visited_list = []
        if unreachable_list is None:
            unreachable_list = []

        # 合併需要忽略的目標
        ignored_targets = visited_list + unreachable_list

        grid = map_data.data
        
        # 產生已知空地 (0~49) 與未知區域 (<0) 的遮罩
        free_mask = ((grid >= 0) & (grid < 50)).astype(np.uint8)
        unknown_mask = (grid < 0).astype(np.uint8)

        # 透過膨脹自由空間並與未知空間取交集，找出交界線
        kernel = np.ones((3, 3), np.uint8)
        free_dilated = cv2.dilate(free_mask, kernel, iterations=1)
        frontier_raw = cv2.bitwise_and(free_dilated, unknown_mask)

        # 尋找連通的邊界區塊
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_raw, connectivity=8
        )

        candidates = []
        # 從 1 開始，跳過背景 (0)
        for lid in range(1, num_labels):
            area = stats[lid, cv2.CC_STAT_AREA]
            if area < self.min_size:
                continue
            
            cx, cy = int(centroids[lid, 0]), int(centroids[lid, 1])
            dist_cells = math.sqrt((cx - robot_mx) ** 2 + (cy - robot_my) ** 2)
            if dist_cells > self.search_radius:
                continue

            wx, wy = map_data.map_to_world(cx, cy)
        
            # 檢查是否落在黑名單的忽略半徑內
            is_ignored = False
            for ig_x, ig_y in ignored_targets:
                if math.sqrt((wx - ig_x)**2 + (wy - ig_y)**2) < self.ignore_radius:
                    is_ignored = True
                    break
        
            if not is_ignored:
                candidates.append((wx, wy, dist_cells, area))

        if not candidates:
            return []

        # 依據距離 (越近越好) 和面積 (越大越好) 排序
        candidates.sort(key=lambda p: (p[2], -p[3]))
        
        # 回傳排序後的世界座標清單
        return [(p[0], p[1]) for p in candidates]

    @staticmethod
    def find_nearest_safe_free_goal(
        map_data: MapData,
        fx: float,
        fy: float,
        inflated_map: np.ndarray,
        max_search_radius: int = 200
    ) -> Optional[Tuple[float, float]]:
        """
        對偵測到的邊界點 (fx, fy) 進行安全檢查。
        若該點本身位於危險區域 (膨脹後的障礙物內)，則使用 BFS 向外擴展，
        尋找鄰近最靠近的安全自由點作為實際導航目標。
        """
        start_mx, start_my = map_data.world_to_map(fx, fy)
        
        # 1. 若原始點本身就安全，直接回傳
        if (0 <= start_mx < map_data.width and 
            0 <= start_my < map_data.height and 
            map_data.is_free(start_mx, start_my) and 
            not inflated_map[start_my, start_mx]
        ):
            return map_data.map_to_world(start_mx, start_my)
        
        # 2. 原點不安全，透過 BFS (廣度優先搜尋) 尋找鄰近的安全點
        visited = set()
        queue = deque([(start_mx, start_my)])
        visited.add((start_mx, start_my))
        
        while queue:
            cx, cy = queue.popleft()
            
            # 若超出最大搜尋範圍則放棄 (避免無窮搜尋)
            if abs(cx - start_mx) > max_search_radius or abs(cy - start_my) > max_search_radius:
                continue
            if not (0 <= cx < map_data.width and 0 <= cy < map_data.height):
                continue

            # 找到第一個安全的自由格子
            if map_data.is_free(cx, cy) and not inflated_map[cy, cx]:
                return map_data.map_to_world(cx, cy)

            # 加入相鄰的 8 個方向 (八連通)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ncx, ncy = cx + dx, cy + dy
                if (ncx, ncy) not in visited:
                    visited.add((ncx, ncy))
                    queue.append((ncx, ncy))

        return None
