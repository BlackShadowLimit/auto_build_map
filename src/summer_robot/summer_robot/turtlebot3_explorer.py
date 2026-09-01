#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurtleBot3 Burger 自主探索系統
=====================================
架構說明：
  1. SLAM  : 使用 Cartographer 建立 OccupancyGrid 地圖
  2. 邊界偵測: 將地圖二值化，利用 OpenCV 找出「已知-未知」邊界 (Frontier)
  3. 路徑規劃: 在 PC 端用 RRT 計算全域路徑，透過 ROS2 DDS 回傳
  4. 地面障礙: USB 攝影機偵測顏色差異 → 估算距離 → 偽裝成 LaserScan 注入地圖
  5. 計算卸載: RRT 計算封裝成 Topic 請求，由 PC 端 Node 承接

部署方式：
  - TurtleBot3 端 : python3 turtlebot3_explorer.py robot
  - PC 端         : python3 turtlebot3_explorer.py pc_server
  環境變數（兩端必須一致）: export ROS_DOMAIN_ID=<同一個數字>

依賴套件:
  pip install opencv-python numpy

ROS2 套件（兩端都需安裝）:
  sudo apt install ros-humble-cartographer ros-humble-cartographer-ros

Cartographer 整合提示：
  在 launch file 中加入 /virtual_obstacles 作為額外的 laser topic，
  或用 topic_tools 合併 /scan 與 /virtual_obstacles 後送入 Cartographer。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import cv2
import math
import time
import random
from typing import List, Tuple, Optional

# ROS2 訊息型別
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# ─────────────────────────────────────────────
# 全域可調參數區（Tunable Parameters）
# 所有超參數集中於此，方便調參
# ─────────────────────────────────────────────

# --- 地圖與探索 ---
MAP_RESOLUTION = 0.05          # 地圖每格解析度 [m/cell]，與 Cartographer 設定一致
FRONTIER_MIN_SIZE = 5          # 最小 Frontier 群集大小（過濾雜訊）[cells]
FRONTIER_SEARCH_RADIUS = 50    # 從機器人位置往外搜尋 Frontier 的最大半徑 [cells]

# --- RRT 路徑規劃 ---
RRT_MAX_ITER = 3000            # RRT 最大迭代次數（越高越容易找到路徑但越慢）
RRT_STEP_SIZE = 0.3            # RRT 每次延伸步長 [m]（越小路徑越精細但越慢）
RRT_GOAL_BIAS = 0.15           # 直接朝目標採樣的機率（0~1，提高可加速收斂）
RRT_GOAL_THRESHOLD = 0.3       # 到達目標的距離閾值 [m]
RRT_COLLISION_CHECK_STEP = 0.05 # 碰撞檢測插值步長 [m]（越小越安全但越慢）
INFLATION_RADIUS = 0.15        # 路徑規劃時障礙物膨脹半徑 [m]
                               # 需 > 機器人半徑 0.105m，建議 0.15~0.25m

# --- 移動控制 ---
MAX_LINEAR_VEL = 0.15          # 最大線速度 [m/s]（Burger 硬體上限 0.22）
MAX_ANGULAR_VEL = 1.0          # 最大角速度 [rad/s]（Burger 硬體上限 2.84）
GOAL_TOLERANCE = 0.15          # 到達路徑點的距離容差 [m]
ANGULAR_KP = 1.5               # 角度控制 P 增益（越大轉彎越積極，過大會震盪）
LINEAR_KP = 0.5                # 線速度控制 P 增益（越大加速越快，過大會超車）
WAYPOINT_SKIP_AHEAD = 3        # 路徑跟隨時最多跳過幾個點（平滑化效果）
LIDAR_SAFE_DISTANCE = 0.30     # LiDAR 前方安全距離 [m]，小於此值觸發緊急停止
LIDAR_FRONT_ANGLE = 30.0       # 前方危險扇形角度 [度]（±N 度內的讀值均納入判斷）

# --- USB 攝影機障礙物偵測 ---
CAMERA_INDEX = 0               # USB 攝影機裝置編號（若有多個相機，調整此值）
CAMERA_WIDTH = 640             # 影像寬度 [px]
CAMERA_HEIGHT = 480            # 影像高度 [px]
CAMERA_FPS = 15                # 攝影機幀率（過高可能超過 USB 頻寬）
CAMERA_FOV_H = 60.0            # 攝影機水平視角 [度]（需依實際鏡頭規格填寫）
CAMERA_HEIGHT_FROM_GROUND = 0.15  # 攝影機離地高度 [m]（實測後填入）
CAMERA_TILT_ANGLE = -15.0     # 攝影機俯角 [度]，負值代表向下看（實測後填入）

# 地面障礙物顏色偵測（HSV 色彩空間，範圍 H:0-179, S:0-255, V:0-255）
# 以下預設為偵測紅色色紙，可根據目標顏色調整
# 使用 HSV 而非 RGB 是因為 HSV 對光照變化更穩健
OBSTACLE_COLOR_LOWER_1 = np.array([0, 100, 100])    # 紅色 Hue 低範圍 1（Hue=0附近）
OBSTACLE_COLOR_UPPER_1 = np.array([10, 255, 255])   # 紅色 Hue 高範圍 1
OBSTACLE_COLOR_LOWER_2 = np.array([160, 100, 100])  # 紅色 Hue 低範圍 2（Hue=180附近，環繞）
OBSTACLE_COLOR_UPPER_2 = np.array([180, 255, 255])  # 紅色 Hue 高範圍 2
OBSTACLE_MIN_AREA = 500        # 最小輪廓面積 [px^2]，過濾過小的雜訊
OBSTACLE_DETECTION_FREQ = 5.0  # 攝影機偵測頻率 [Hz]（越高 CPU 負擔越重）

# 偽裝 LaserScan 參數（注入 Cartographer 用）
FAKE_SCAN_RANGE_MIN = 0.12     # 偽裝掃描最小距離 [m]（與實體 LiDAR 一致）
FAKE_SCAN_RANGE_MAX = 1.5      # 偽裝掃描最大距離 [m]（攝影機偵測有效距離）
FAKE_SCAN_TOPIC = '/virtual_obstacles'  # 虛擬掃描 topic 名稱

# --- 計算卸載 ---
OFFLOAD_ENABLED = True         # True=使用 PC 端計算；False=本機計算（較慢）
PLAN_COOLDOWN = 5.0            # 重新規劃的最小間隔 [s]（避免頻繁重算）


# ─────────────────────────────────────────────
# 輔助資料結構
# ─────────────────────────────────────────────

class RRTNode:
    """RRT 樹的單一節點，儲存座標與父節點"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['RRTNode'] = None

    def distance_to(self, other: 'RRTNode') -> float:
        """計算到另一節點的歐式距離"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class MapData:
    """
    封裝 OccupancyGrid 的地圖工具類別
    提供座標轉換、碰撞查詢、膨脹地圖生成等功能

    OccupancyGrid 數值說明：
      -1  : 未知空間（Cartographer 尚未探索）
       0  : 自由空間（可通行）
      1~100 : 占據機率，>=50 視為障礙物
    """
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.width      = occupancy_grid.info.width
        self.height     = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution
        self.origin_x   = occupancy_grid.info.origin.position.x
        self.origin_y   = occupancy_grid.info.origin.position.y
        # 將 ROS 的 1D list 轉為 2D numpy array（row=y, col=x）
        self.data = np.array(occupancy_grid.data, dtype=np.int8).reshape(
            (self.height, self.width)
        )

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界座標 [m] 轉地圖格座標 [cell]"""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """地圖格座標 [cell] 轉世界座標 [m]（取格子中心點）"""
        wx = mx * self.resolution + self.origin_x + self.resolution / 2.0
        wy = my * self.resolution + self.origin_y + self.resolution / 2.0
        return wx, wy

    def is_free(self, mx: int, my: int) -> bool:
        """判斷地圖格是否為自由空間（占據機率 < 50 且已探索）"""
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return False
        return 0 <= int(self.data[my, mx]) < 50

    def is_unknown(self, mx: int, my: int) -> bool:
        """判斷地圖格是否為未知空間（值 = -1）"""
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return False
        return int(self.data[my, mx]) == -1

    def get_inflated_map(self, inflation_m: float) -> np.ndarray:
        """
        對占據地圖進行形態學膨脹，用於路徑規劃的安全邊距

        膨脹原理：以 inflation_m 為半徑，在所有障礙物周圍建立緩衝區
        使機器人在距障礙物 > inflation_m 的地方規劃路徑，避免刮擦

        回傳：布林陣列（True = 障礙或已膨脹區域，不可通行）
        """
        inflation_cells = int(inflation_m / self.resolution)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        # 將占據格（值 >= 50）標記為 1
        obstacle_map = (self.data >= 50).astype(np.uint8)
        # 形態學膨脹
        inflated = cv2.dilate(obstacle_map, kernel)
        return inflated.astype(bool)


# ─────────────────────────────────────────────
# 模組 1：Frontier 邊界偵測
# ─────────────────────────────────────────────

def detect_frontiers(
    map_data: MapData,
    robot_mx: int,
    robot_my: int
) -> List[Tuple[float, float]]:
    """
    利用影像二值化找出 Frontier（已知自由空間與未知空間的邊界）

    演算法流程（基於 OpenCV 影像處理）：
      Step 1 : 將地圖轉為二值影像
               - 自由空間（0~49）  -> 白色（255）
               - 未知空間（-1）    -> 灰色（128）
               - 障礙空間（50~100）-> 黑色（0）
      Step 2 : 對自由空間做輕微膨脹（3x3 kernel）
               -> 自由空間向外擴張一格，接觸到未知空間邊緣
      Step 3 : 膨脹後的自由空間 AND 未知空間 = Frontier 像素
      Step 4 : 連通元件分析（connectedComponentsWithStats）
               -> 找到各個 Frontier 群集，計算面積與中心
      Step 5 : 過濾太小的群集（FRONTIER_MIN_SIZE），
               按距離由近到遠排序後回傳

    參數:
        map_data  : 當前地圖資料物件
        robot_mx  : 機器人在地圖的 x 格座標
        robot_my  : 機器人在地圖的 y 格座標

    回傳:
        Frontier 中心點世界座標清單 [(wx, wy), ...]，由近到遠排列
        若無 Frontier 則回傳空列表（代表地圖已完整探索）
    """
    # np.int8 的 -1 在比較時需先轉換，避免溢位
    grid_int = map_data.data.astype(np.int16)

    # Step 1: 建立自由空間與未知空間的二值遮罩
    free_mask    = ((grid_int >= 0) & (grid_int < 50)).astype(np.uint8) * 255
    unknown_mask = (grid_int == -1).astype(np.uint8) * 255

    # Step 2: 對自由空間做輕微膨脹（使其觸及未知空間邊緣）
    kernel = np.ones((3, 3), np.uint8)
    free_dilated = cv2.dilate(free_mask, kernel, iterations=1)

    # Step 3: Frontier = 膨脹後的自由空間 AND 未知空間
    frontier_raw = cv2.bitwise_and(free_dilated, unknown_mask)

    # Step 4: 連通元件分析，找到各 Frontier 群集
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        frontier_raw, connectivity=8
    )

    frontier_points = []
    for label_id in range(1, num_labels):  # label_id=0 是背景，跳過
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < FRONTIER_MIN_SIZE:
            continue  # 過濾太小的雜訊 Frontier

        cx = int(centroids[label_id, 0])  # Frontier 群集中心 x [cell]
        cy = int(centroids[label_id, 1])  # Frontier 群集中心 y [cell]

        # 距離篩選（只考慮一定範圍內的 Frontier）
        dist_cells = math.sqrt((cx - robot_mx) ** 2 + (cy - robot_my) ** 2)
        if dist_cells > FRONTIER_SEARCH_RADIUS:
            continue

        wx, wy = map_data.map_to_world(cx, cy)
        frontier_points.append((wx, wy, dist_cells, area))

    if not frontier_points:
        return []

    # Step 5: 依距離由近到遠排序（等距時優先選面積大的）
    frontier_points.sort(key=lambda p: (p[2], -p[3]))

    return [(p[0], p[1]) for p in frontier_points]


# ─────────────────────────────────────────────
# 模組 2：RRT 全域路徑規劃
# ─────────────────────────────────────────────

def rrt_plan(
    start_wx: float, start_wy: float,
    goal_wx: float, goal_wy: float,
    map_data: MapData
) -> Optional[List[Tuple[float, float]]]:
    """
    RRT（Rapidly-exploring Random Tree）全域路徑規劃

    RRT 演算法原理：
      1. 從起點建立一棵樹
      2. 每次隨機採樣一個點（偶爾直接採樣目標點，稱為 goal bias）
      3. 找到樹中最近的節點，往採樣點方向延伸 RRT_STEP_SIZE
      4. 若新節點無碰撞，加入樹中
      5. 重複直到樹中有節點足夠靠近目標

    碰撞檢測：
      使用插值法逐步檢查路徑段上每個 RRT_COLLISION_CHECK_STEP 間距的點
      是否在膨脹地圖的可通行區域內

    參數:
        start_wx, start_wy : 起點世界座標 [m]（機器人當前位置）
        goal_wx,  goal_wy  : 終點世界座標 [m]（Frontier 中心）
        map_data           : 當前地圖資料

    回傳:
        路徑點清單 [(wx, wy), ...]，從起點到終點
        若迭代次數耗盡仍無解，回傳 None

    注意：此函式計算量大（O(N) per iteration），建議在 PC 端執行
    """
    # 取得膨脹地圖用於碰撞檢測（一次性計算，不在迴圈內重複）
    inflated_map = map_data.get_inflated_map(INFLATION_RADIUS)

    def _is_collision_free(x1: float, y1: float, x2: float, y2: float) -> bool:
        """插值法碰撞檢測：沿線段逐點確認是否可通行"""
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        steps = max(1, int(dist / RRT_COLLISION_CHECK_STEP))
        for i in range(steps + 1):
            t = i / steps
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            mx, my = map_data.world_to_map(ix, iy)
            if mx < 0 or mx >= map_data.width or my < 0 or my >= map_data.height:
                return False
            if inflated_map[my, mx]:
                return False
        return True

    def _nearest_node(tree: List[RRTNode], sample: RRTNode) -> RRTNode:
        """在樹中找到歐式距離最近的節點（線性搜尋）"""
        return min(tree, key=lambda n: n.distance_to(sample))

    def _steer(from_node: RRTNode, to_node: RRTNode) -> RRTNode:
        """從 from_node 往 to_node 方向延伸，長度限制為 RRT_STEP_SIZE"""
        dist = from_node.distance_to(to_node)
        if dist <= RRT_STEP_SIZE:
            return RRTNode(to_node.x, to_node.y)
        ratio = RRT_STEP_SIZE / dist
        nx = from_node.x + ratio * (to_node.x - from_node.x)
        ny = from_node.y + ratio * (to_node.y - from_node.y)
        return RRTNode(nx, ny)

    # 計算隨機採樣的地圖邊界
    map_min_x = map_data.origin_x
    map_max_x = map_data.origin_x + map_data.width  * map_data.resolution
    map_min_y = map_data.origin_y
    map_max_y = map_data.origin_y + map_data.height * map_data.resolution

    # 初始化 RRT 樹
    start_node = RRTNode(start_wx, start_wy)
    goal_node  = RRTNode(goal_wx,  goal_wy)
    tree: List[RRTNode] = [start_node]

    for _ in range(RRT_MAX_ITER):
        # 採樣：以 RRT_GOAL_BIAS 機率直接採樣目標（加速收斂）
        if random.random() < RRT_GOAL_BIAS:
            sample = RRTNode(goal_wx, goal_wy)
        else:
            sample = RRTNode(
                random.uniform(map_min_x, map_max_x),
                random.uniform(map_min_y, map_max_y)
            )

        # 找到最近節點並嘗試延伸
        nearest  = _nearest_node(tree, sample)
        new_node = _steer(nearest, sample)

        # 碰撞檢測
        if not _is_collision_free(nearest.x, nearest.y, new_node.x, new_node.y):
            continue

        # 無碰撞，加入樹
        new_node.parent = nearest
        tree.append(new_node)

        # 檢查是否足夠靠近目標
        if new_node.distance_to(goal_node) <= RRT_GOAL_THRESHOLD:
            # 從目標節點回溯路徑
            path = []
            node: Optional[RRTNode] = new_node
            while node is not None:
                path.append((node.x, node.y))
                node = node.parent
            path.reverse()
            return path

    return None  # 超過最大迭代次數，規劃失敗


def smooth_path(
    path: List[Tuple[float, float]],
    map_data: MapData
) -> List[Tuple[float, float]]:
    """
    使用視線（Line of Sight）平滑化 RRT 路徑

    原理（Greedy Shortcutting）：
      從路徑起點出發，嘗試直接連接到路徑上最遠的點
      若直線段無碰撞，跳過中間所有點
      重複此過程直到到達終點

    效果：將 RRT 產生的鋸齒路徑平滑化，減少不必要的轉彎

    參數:
        path     : 原始 RRT 路徑
        map_data : 地圖資料（用於碰撞檢測）

    回傳:
        平滑化後的路徑點清單
    """
    if len(path) <= 2:
        return path

    inflated_map = map_data.get_inflated_map(INFLATION_RADIUS)

    def _has_los(x1: float, y1: float, x2: float, y2: float) -> bool:
        """視線檢測（同碰撞檢測，用於判斷直線可達性）"""
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        steps = max(1, int(dist / RRT_COLLISION_CHECK_STEP))
        for i in range(steps + 1):
            t = i / steps
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            mx, my = map_data.world_to_map(ix, iy)
            if mx < 0 or mx >= map_data.width or my < 0 or my >= map_data.height:
                return False
            if inflated_map[my, mx]:
                return False
        return True

    smoothed = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        # 從最遠的點往回找，尋找可直線到達的最遠路徑點
        furthest_reachable = current_idx + 1
        for j in range(len(path) - 1, current_idx, -1):
            x1, y1 = path[current_idx]
            x2, y2 = path[j]
            if _has_los(x1, y1, x2, y2):
                furthest_reachable = j
                break
        smoothed.append(path[furthest_reachable])
        current_idx = furthest_reachable

    return smoothed


# ─────────────────────────────────────────────
# 模組 3：攝影機地面障礙物偵測
# ─────────────────────────────────────────────

class GroundObstacleDetector:
    """
    使用 USB 攝影機偵測地面顏色障礙物（如色紙、低矮彩色物件）

    偵測流程：
      1. 擷取影像 -> 轉換至 HSV 色彩空間
      2. 建立顏色遮罩（inRange），使用 HSV 提升對光照的穩健性
      3. 形態學去噪（OPEN=去雜訊, CLOSE=填空洞）
      4. 找到輪廓，計算底部中心像素座標
      5. 利用針孔相機幾何模型估算地面距離
      6. 轉換為機器人座標系的相對位置

    距離估算原理（針孔相機投影到地平面）：
      已知：攝影機離地高度 H，俯角 theta
      像素 y 對應仰角 phi（從光軸偏移）
      距離 D = H / tan(|theta| + phi)

      注意：此模型假設地面平坦且障礙物高度約為 0（薄色紙）
      若障礙物有高度，需額外校正
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_camera()

    def _init_camera(self):
        """初始化 USB 攝影機並計算相機內部參數"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"無法開啟攝影機 index={CAMERA_INDEX}\n"
                "請確認 USB 攝影機已連接，或調整 CAMERA_INDEX 參數"
            )

        # 計算焦距（像素單位）
        # 推導：tan(FOV/2) = (影像寬/2) / 焦距_px
        fov_rad = math.radians(CAMERA_FOV_H)
        self.focal_length_px = (CAMERA_WIDTH / 2.0) / math.tan(fov_rad / 2.0)

        # 影像主點（假設在中心，未進行畸變校正）
        self.cx = CAMERA_WIDTH  / 2.0
        self.cy = CAMERA_HEIGHT / 2.0

        # 攝影機俯角（弧度），負值 = 向下看
        self.tilt_rad = math.radians(CAMERA_TILT_ANGLE)

    def detect_obstacles(self, robot_yaw: float) -> List[Tuple[float, float]]:
        """
        執行一次障礙物偵測

        參數:
            robot_yaw : 機器人當前偏航角 [rad]（保留供未來座標轉換使用）

        回傳:
            [(rel_x, rel_y), ...] 障礙物相對機器人前方的位置 [m]
            rel_x > 0 = 前方，rel_y > 0 = 左方，rel_y < 0 = 右方
        """
        ret, frame = self.cap.read()
        if not ret:
            return []

        obstacles = []

        # 轉換為 HSV 色彩空間（H:0-179, S:0-255, V:0-255）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 建立顏色遮罩（合併兩個紅色 Hue 範圍，處理色相環繞問題）
        mask1 = cv2.inRange(hsv, OBSTACLE_COLOR_LOWER_1, OBSTACLE_COLOR_UPPER_1)
        mask2 = cv2.inRange(hsv, OBSTACLE_COLOR_LOWER_2, OBSTACLE_COLOR_UPPER_2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # 形態學處理：先開運算去除雜訊，再閉運算填補空洞
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找到輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < OBSTACLE_MIN_AREA:
                continue  # 輪廓太小，可能是雜訊

            # 取外接矩形
            x_c, y_c, w, h = cv2.boundingRect(contour)
            pixel_x = x_c + w / 2.0   # 水平中心（決定左右角度）
            pixel_y = y_c + h          # 底部邊緣（最接近機器人的障礙邊緣）

            # 估算地面距離
            distance = self._estimate_ground_distance(pixel_x, pixel_y)
            if distance is None:
                continue
            if not (FAKE_SCAN_RANGE_MIN <= distance <= FAKE_SCAN_RANGE_MAX):
                continue  # 距離超出有效範圍

            # 計算水平偏角（相對攝影機正前方）
            angle_horizontal = math.atan2(
                -(pixel_x - self.cx),  # 負號：相機 x 往右，機器人座標往左為正
                self.focal_length_px
            )

            # 轉換為機器人座標系（前方=x軸正方向，左方=y軸正方向）
            rel_x = distance * math.cos(angle_horizontal)
            rel_y = distance * math.sin(angle_horizontal)

            obstacles.append((rel_x, rel_y))

        return obstacles

    def _estimate_ground_distance(self, pixel_x: float, pixel_y: float) -> Optional[float]:
        """
        針孔相機投影到地平面的距離估算

        幾何推導：
          設攝影機俯角為 theta（弧度，負值代表向下）
          像素 (pixel_x, pixel_y) 對應的垂直視角偏移 phi：
            phi = atan((pixel_y - cy) / focal_length_px)

          地面點到攝影機底部的水平距離：
            D = H / tan(|theta| + phi)

          其中 H = CAMERA_HEIGHT_FROM_GROUND

        回傳:
            地面距離 [m]，若幾何不合法（視線平行或朝上）則回傳 None
        """
        # 像素 y 偏移對應的垂直視角（從影像中心計算）
        pixel_angle = math.atan2(pixel_y - self.cy, self.focal_length_px)

        # 總俯角 = 攝影機固定俯角 + 像素對應的俯角偏移
        total_depression_angle = abs(self.tilt_rad) + pixel_angle

        if total_depression_angle <= 0.01:
            # 視線幾乎平行地面，距離趨於無限大
            return None

        distance = CAMERA_HEIGHT_FROM_GROUND / math.tan(total_depression_angle)
        return distance

    def release(self):
        """釋放攝影機資源（節點關閉時呼叫）"""
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ─────────────────────────────────────────────
# 模組 4：偽裝 LaserScan 注入器
# ─────────────────────────────────────────────

def create_virtual_laserscan(
    obstacles_rel: List[Tuple[float, float]],
    node: Node
) -> LaserScan:
    """
    將攝影機偵測到的障礙物轉換為稀疏 LaserScan 訊息

    原理：
      - 建立一個全圓 360 度的 LaserScan（初始全為 inf = 無讀值）
      - 對每個障礙物，計算其相對機器人的角度與距離
      - 在對應的角度索引填入距離值
      - 發布到 FAKE_SCAN_TOPIC，由 Cartographer 的感測器融合機制處理

    Cartographer 整合方式：
      方法 A：在 cartographer launch 中額外加入 /virtual_obstacles 的 sensor 對應
      方法 B：使用 topic_tools MuxLaserScan 合併 /scan 與 /virtual_obstacles

    參數:
        obstacles_rel : 障礙物相對機器人位置清單 [(rel_x, rel_y), ...]
        node          : ROS2 節點（用於取得當前時間戳）

    回傳:
        LaserScan 訊息（可直接發布）
    """
    scan_msg = LaserScan()
    scan_msg.header.stamp    = node.get_clock().now().to_msg()
    scan_msg.header.frame_id = 'base_scan'  # 與 TurtleBot3 LiDAR 使用相同 frame

    # 掃描角度設定（全圓 360 度，每度一個讀值）
    num_readings = 360
    scan_msg.angle_min       = -math.pi
    scan_msg.angle_max       =  math.pi
    scan_msg.angle_increment = (2.0 * math.pi) / num_readings
    scan_msg.time_increment  = 0.0
    scan_msg.scan_time       = 1.0 / OBSTACLE_DETECTION_FREQ
    scan_msg.range_min       = FAKE_SCAN_RANGE_MIN
    scan_msg.range_max       = FAKE_SCAN_RANGE_MAX

    # 初始化所有距離為無效（inf = 無讀值，Cartographer 會忽略）
    ranges = [float('inf')] * num_readings

    for (rel_x, rel_y) in obstacles_rel:
        # 計算障礙物角度（相對機器人朝向，atan2 回傳 [-pi, pi]）
        angle    = math.atan2(rel_y, rel_x)
        distance = math.sqrt(rel_x ** 2 + rel_y ** 2)

        # 將角度對應到陣列索引
        angle_normalized = angle - scan_msg.angle_min  # 轉換到 [0, 2*pi]
        idx = int(angle_normalized / scan_msg.angle_increment) % num_readings

        # 若同一角度有多個障礙物，取最近的（保守策略）
        if distance < ranges[idx]:
            ranges[idx] = distance

    scan_msg.ranges = ranges
    return scan_msg


# ─────────────────────────────────────────────
# 模組 5：LiDAR 即時安全守衛
# ─────────────────────────────────────────────

def check_lidar_obstacle(scan: LaserScan) -> bool:
    """
    即時 LiDAR 障礙物安全檢測（前方扇形區）

    設計原則：
      - 只檢查前方 ±LIDAR_FRONT_ANGLE 度（機器人行進方向）
      - 過濾無效讀值（nan, inf, 超出量程）
      - 任何一個讀值 < LIDAR_SAFE_DISTANCE 即觸發緊急停止

    TurtleBot3 LiDAR（LDS-01/LDS-02）說明：
      - 前方為 angle = 0 弧度
      - 順時針為正角度，逆時針為負角度

    參數:
        scan : 最新的 LaserScan 訊息（TurtleBot3 的 /scan topic）

    回傳:
        True = 前方有障礙物，需緊急停止
        False = 前方安全
    """
    if not scan.ranges:
        return False

    front_angle_rad = math.radians(LIDAR_FRONT_ANGLE)

    for i, r in enumerate(scan.ranges):
        # 過濾無效讀值
        if math.isnan(r) or math.isinf(r):
            continue
        if r < scan.range_min or r > scan.range_max:
            continue

        # 計算此讀值的角度（弧度）
        angle = scan.angle_min + i * scan.angle_increment

        # 只考慮前方扇形
        if abs(angle) < front_angle_rad:
            if r < LIDAR_SAFE_DISTANCE:
                return True  # 前方有障礙物

    return False  # 前方安全


# ─────────────────────────────────────────────
# 模組 6：移動控制器（P 控制路徑跟隨）
# ─────────────────────────────────────────────

class MotionController:
    """
    路徑跟隨控制器（P 控制器）

    控制邏輯：
      1. 確定當前目標路徑點（允許跳過已到達的點）
      2. 計算到目標點的角度誤差
      3. 角速度 = ANGULAR_KP x 角度誤差（P 控制）
      4. 線速度 = LINEAR_KP x 距離 x (1 - |角度誤差| / (pi/2))
         -> 轉彎時自動減速，直線時加速
      5. 速度限制在 [0, MAX_LINEAR_VEL] 和 [-MAX_ANGULAR_VEL, MAX_ANGULAR_VEL]
    """

    def __init__(self, node: Node):
        self.node = node
        self.cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        self.current_waypoint_idx = 0
        self.path: List[Tuple[float, float]] = []

    def set_path(self, path: List[Tuple[float, float]]):
        """設定新路徑並重置路徑點索引"""
        self.path = path
        self.current_waypoint_idx = 0
        self.node.get_logger().info(f"[移動] 設定新路徑，共 {len(path)} 個路徑點")

    def update(self, robot_x: float, robot_y: float, robot_yaw: float) -> bool:
        """
        執行一次控制更新，計算並發布 cmd_vel

        參數:
            robot_x, robot_y : 機器人當前世界座標 [m]
            robot_yaw        : 機器人當前偏航角 [rad]

        回傳:
            True = 已到達路徑終點
            False = 仍在行進中
        """
        if not self.path or self.current_waypoint_idx >= len(self.path):
            self.stop()
            return True  # 路徑已完成

        # 向前看最多 WAYPOINT_SKIP_AHEAD 個點，跳過已到達的
        look_ahead_limit = min(
            self.current_waypoint_idx + WAYPOINT_SKIP_AHEAD,
            len(self.path) - 1
        )
        for i in range(look_ahead_limit, self.current_waypoint_idx, -1):
            wx, wy = self.path[i]
            dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
            if dist <= GOAL_TOLERANCE:
                self.current_waypoint_idx = i + 1
                break

        if self.current_waypoint_idx >= len(self.path):
            self.stop()
            return True  # 到達終點

        # 計算到目標點的方向與距離
        target_x, target_y = self.path[self.current_waypoint_idx]
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist_to_target = math.sqrt(dx ** 2 + dy ** 2)

        if dist_to_target < GOAL_TOLERANCE:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path):
                self.stop()
                return True

        # P 控制計算速度
        target_angle = math.atan2(dy, dx)

        # 角度誤差（標準化到 [-pi, pi]）
        angle_error = target_angle - robot_yaw
        angle_error = (angle_error + math.pi) % (2.0 * math.pi) - math.pi

        # 角速度（P 控制）
        angular_vel = ANGULAR_KP * angle_error
        angular_vel = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular_vel))

        # 線速度（轉彎大時減速，防止甩尾）
        angle_factor = max(0.0, 1.0 - abs(angle_error) / (math.pi / 2))
        linear_vel   = LINEAR_KP * dist_to_target * angle_factor
        linear_vel   = max(0.0, min(MAX_LINEAR_VEL, linear_vel))

        # 發布 cmd_vel
        cmd = Twist()
        cmd.linear.x  = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)

        return False  # 仍在行進中

    def stop(self):
        """發送零速度指令，停止機器人"""
        self.cmd_pub.publish(Twist())

    def emergency_stop(self):
        """緊急停止（碰撞風險）並清除當前路徑"""
        self.stop()
        self.path = []
        self.node.get_logger().warn("[移動] !! 緊急停止！前方偵測到障礙物")


# ─────────────────────────────────────────────
# 機器人端主節點（部署到 TurtleBot3）
# ─────────────────────────────────────────────

class RobotNode(Node):
    """
    TurtleBot3 Burger 自主探索主節點

    訂閱 Topics:
      /map              : Cartographer 輸出的 OccupancyGrid 地圖
      /odom             : 里程計（機器人位姿）
      /scan             : TurtleBot3 LiDAR 掃描資料
      /pc/rrt_path      : PC 端回傳的 RRT 規劃路徑（OFFLOAD 模式）

    發布 Topics:
      /cmd_vel          : 移動速度指令
      /virtual_obstacles: 偽裝 LaserScan（攝影機偵測的地面障礙物）
      /pc/plan_request  : 路徑規劃請求（發送給 PC 端）
      /explorer/status  : 探索狀態字串（供監控）

    計時器:
      10 Hz  : 主控制迴圈（LiDAR 安全守衛 + 路徑跟隨）
      5 Hz   : 攝影機障礙物偵測
      1 Hz   : 探索規劃迴圈（Frontier 偵測 + 路徑請求）
    """

    def __init__(self):
        super().__init__('turtlebot3_explorer')

        # QoS 設定：地圖使用 TransientLocal（後訂閱也能收到最新地圖）
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # 訂閱者
        self.map_sub  = self.create_subscription(OccupancyGrid, '/map',  self._on_map,  map_qos)
        self.odom_sub = self.create_subscription(Odometry,      '/odom', self._on_odom, 10)
        self.scan_sub = self.create_subscription(LaserScan,     '/scan', self._on_scan, 10)
        self.path_sub = self.create_subscription(
            Path, '/pc/rrt_path', self._on_path_received, 10
        )

        # 發布者
        self.virtual_scan_pub = self.create_publisher(LaserScan,   FAKE_SCAN_TOPIC,    10)
        self.plan_request_pub = self.create_publisher(PoseStamped, '/pc/plan_request', 10)
        self.status_pub       = self.create_publisher(String,      '/explorer/status', 10)

        # 狀態變數
        self.current_map: Optional[OccupancyGrid] = None
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.latest_scan: Optional[LaserScan] = None
        self.is_exploring   = False
        self.exploration_done = False
        self.last_plan_time = 0.0

        # 模組初始化
        self.motion_ctrl = MotionController(self)

        # 嘗試初始化攝影機（失敗時降級運行）
        try:
            self.obstacle_detector: Optional[GroundObstacleDetector] = GroundObstacleDetector()
            self.get_logger().info("[初始化] USB 攝影機模組啟動成功")
        except RuntimeError as e:
            self.obstacle_detector = None
            self.get_logger().warn(f"[初始化] 攝影機啟動失敗: {e}")
            self.get_logger().warn("[初始化] 地面障礙物偵測停用，僅使用 LiDAR")

        # 計時器
        self.create_timer(0.10, self._control_loop)                    # 10 Hz
        self.create_timer(1.0 / OBSTACLE_DETECTION_FREQ, self._camera_loop)  # 5 Hz
        self.create_timer(1.00, self._exploration_loop)                # 1 Hz

        self.get_logger().info("=== TurtleBot3 自主探索節點已啟動 ===")
        self.get_logger().info(
            f"計算模式: {'卸載至 PC 端' if OFFLOAD_ENABLED else '本機 RRT'}"
        )

    # Topic 回調

    def _on_map(self, msg: OccupancyGrid):
        """接收 Cartographer 地圖更新"""
        self.current_map = msg

    def _on_odom(self, msg: Odometry):
        """接收里程計，更新機器人位姿（位置 + 偏航角）"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        # 從四元數提取偏航角（Euler Z）
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _on_scan(self, msg: LaserScan):
        """接收 LiDAR 掃描資料（快取最新值）"""
        self.latest_scan = msg

    def _on_path_received(self, msg: Path):
        """接收 PC 端回傳的 RRT 路徑，設定給移動控制器"""
        if not msg.poses:
            self.get_logger().warn("[路徑] PC 端回傳空路徑，將重新規劃")
            self.is_exploring = False
            return

        path_points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.motion_ctrl.set_path(path_points)
        self.is_exploring = True
        self.get_logger().info(f"[路徑] 收到 PC 端路徑，{len(path_points)} 個路徑點")

    # 主要計時器迴圈

    def _control_loop(self):
        """
        主控制迴圈（10 Hz）
        優先順序：① LiDAR 安全守衛 -> ② 路徑跟隨
        """
        # LiDAR 安全守衛：前方有障礙時緊急停止並取消當前路徑
        if self.latest_scan is not None and check_lidar_obstacle(self.latest_scan):
            self.motion_ctrl.emergency_stop()
            self.is_exploring = False
            return

        # 路徑跟隨
        if self.is_exploring:
            reached_goal = self.motion_ctrl.update(
                self.robot_x, self.robot_y, self.robot_yaw
            )
            if reached_goal:
                self.get_logger().info("[移動] 到達目標，請求探索新 Frontier")
                self.is_exploring = False

    def _exploration_loop(self):
        """
        探索規劃迴圈（1 Hz）
        當機器人閒置時（is_exploring=False）：
          1. 偵測 Frontier
          2. 選擇最近的 Frontier 作為目標
          3. 請求 RRT 路徑規劃
        """
        if self.exploration_done or self.is_exploring:
            return
        if self.current_map is None:
            self.get_logger().info("[探索] 等待地圖初始化...")
            return

        # 速率限制
        now = time.time()
        if now - self.last_plan_time < PLAN_COOLDOWN:
            return
        self.last_plan_time = now

        # 偵測 Frontier
        map_data = MapData(self.current_map)
        robot_mx, robot_my = map_data.world_to_map(self.robot_x, self.robot_y)
        frontiers = detect_frontiers(map_data, robot_mx, robot_my)

        if not frontiers:
            self.get_logger().info("[探索] 所有 Frontier 已探索，地圖完整！")
            self.exploration_done = True
            self.motion_ctrl.stop()
            status = String()
            status.data = "EXPLORATION_COMPLETE"
            self.status_pub.publish(status)
            return

        # 選擇最近的 Frontier
        goal_wx, goal_wy = frontiers[0]
        self.get_logger().info(
            f"[探索] 選定 Frontier: ({goal_wx:.2f}, {goal_wy:.2f})，"
            f"共找到 {len(frontiers)} 個候選"
        )

        if OFFLOAD_ENABLED:
            # 卸載模式：發送規劃請求至 PC 端
            req = PoseStamped()
            req.header.stamp    = self.get_clock().now().to_msg()
            req.header.frame_id = 'map'
            req.pose.position.x = goal_wx
            req.pose.position.y = goal_wy
            req.pose.orientation.w = 1.0
            self.plan_request_pub.publish(req)
            self.get_logger().info("[探索] 路徑規劃請求已發送至 PC 端")
        else:
            # 本機模式：直接執行 RRT
            self.get_logger().info("[探索] 本機 RRT 規劃中...")
            path = rrt_plan(self.robot_x, self.robot_y, goal_wx, goal_wy, map_data)
            if path:
                smooth = smooth_path(path, map_data)
                self.motion_ctrl.set_path(smooth)
                self.is_exploring = True
                self.get_logger().info(f"[探索] RRT 規劃成功，{len(smooth)} 個路徑點")
            else:
                self.get_logger().warn("[探索] RRT 規劃失敗，嘗試下一個 Frontier")
                if len(frontiers) > 1:
                    goal_wx, goal_wy = frontiers[1]
                    path = rrt_plan(self.robot_x, self.robot_y, goal_wx, goal_wy, map_data)
                    if path:
                        smooth = smooth_path(path, map_data)
                        self.motion_ctrl.set_path(smooth)
                        self.is_exploring = True

    def _camera_loop(self):
        """
        攝影機偵測迴圈（OBSTACLE_DETECTION_FREQ Hz）
        偵測地面顏色障礙物並發布偽裝 LaserScan
        """
        if self.obstacle_detector is None:
            return

        try:
            obstacles = self.obstacle_detector.detect_obstacles(self.robot_yaw)
        except Exception as e:
            self.get_logger().warn(f"[攝影機] 偵測發生錯誤: {e}")
            return

        if obstacles:
            virtual_scan = create_virtual_laserscan(obstacles, self)
            self.virtual_scan_pub.publish(virtual_scan)
            self.get_logger().debug(f"[攝影機] 發布 {len(obstacles)} 個虛擬障礙物讀值")

    def destroy_node(self):
        """節點關閉時的清理工作"""
        self.motion_ctrl.stop()
        if self.obstacle_detector:
            self.obstacle_detector.release()
        super().destroy_node()


# ─────────────────────────────────────────────
# PC 端計算伺服器節點（部署到電腦，非機器人）
# ─────────────────────────────────────────────

class PCComputeServer(Node):
    """
    PC 端重計算伺服器節點

    職責：
      - 共享同一 ROS2 DDS 網路（設定相同 ROS_DOMAIN_ID）
      - 從 /map 和 /odom 同步地圖與機器人位姿
      - 接收 /pc/plan_request 規劃請求
      - 在 PC 的高效能 CPU 上執行 RRT 規劃
      - 將規劃結果發布至 /pc/rrt_path

    部署方式（PC 端終端機）：
      export ROS_DOMAIN_ID=<與機器人相同的數字>
      python3 turtlebot3_explorer.py pc_server

    注意：PC 與 TurtleBot3 必須在同一個區域網路下
          防火牆需允許 DDS 使用的 UDP 埠（預設 7400-7500）
    """

    def __init__(self):
        super().__init__('pc_compute_server')

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # 訂閱者
        self.map_sub  = self.create_subscription(OccupancyGrid, '/map',  self._on_map,  map_qos)
        self.odom_sub = self.create_subscription(Odometry,      '/odom', self._on_odom, 10)
        self.req_sub  = self.create_subscription(
            PoseStamped, '/pc/plan_request', self._on_plan_request, 10
        )

        # 發布者
        self.path_pub = self.create_publisher(Path, '/pc/rrt_path', 10)

        # 狀態
        self.current_map: Optional[OccupancyGrid] = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.is_computing = False  # 防止同時處理多個請求

        self.get_logger().info("=== PC 端計算伺服器已啟動，等待規劃請求 ===")

    def _on_map(self, msg: OccupancyGrid):
        """同步接收地圖（與機器人端共享同一 topic）"""
        self.current_map = msg

    def _on_odom(self, msg: Odometry):
        """同步接收機器人位姿"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def _on_plan_request(self, msg: PoseStamped):
        """
        接收規劃請求並執行 RRT，完成後發布路徑

        注意：RRT 計算在 ROS 回調中同步執行
              若需要非阻塞，可改用 MultiThreadedExecutor
              搭配 threading.Thread 執行 RRT
        """
        if self.is_computing:
            self.get_logger().warn("[PC] 已有規劃任務進行中，忽略新請求")
            return
        if self.current_map is None:
            self.get_logger().warn("[PC] 尚未收到地圖，無法規劃路徑")
            return

        goal_wx = msg.pose.position.x
        goal_wy = msg.pose.position.y
        self.get_logger().info(
            f"[PC] 開始規劃：起點 ({self.robot_x:.2f}, {self.robot_y:.2f}) "
            f"-> 目標 ({goal_wx:.2f}, {goal_wy:.2f})"
        )

        self.is_computing = True
        try:
            map_data = MapData(self.current_map)
            path = rrt_plan(self.robot_x, self.robot_y, goal_wx, goal_wy, map_data)

            if path:
                smooth = smooth_path(path, map_data)
                self.get_logger().info(
                    f"[PC] 規劃成功！原始 {len(path)} 點 -> 平滑後 {len(smooth)} 點"
                )
                self._publish_path(smooth)
            else:
                self.get_logger().warn("[PC] RRT 規劃失敗（超過最大迭代次數）")
                self._publish_path([])  # 發布空路徑通知機器人端

        except Exception as e:
            self.get_logger().error(f"[PC] 規劃過程發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_computing = False

    def _publish_path(self, path: List[Tuple[float, float]]):
        """將路徑點清單轉換為 nav_msgs/Path 並發布"""
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for wx, wy in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # 無旋轉（RRT 只規劃 xy 平面）
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


# ─────────────────────────────────────────────
# 程式入口
# ─────────────────────────────────────────────

def main():
    """
    程式入口函式

    使用方式：
      機器人端（TurtleBot3 上執行）:
        python3 turtlebot3_explorer.py robot

      PC 端（電腦上執行）:
        python3 turtlebot3_explorer.py pc_server

    環境變數（兩端必須相同）:
      export ROS_DOMAIN_ID=42  # 任意 0-232，確保同一 DDS 域

    前置條件：
      1. Cartographer 已啟動並發布 /map topic
      2. TurtleBot3 相關 driver 已啟動（/scan, /odom, /cmd_vel）
      3. USB 攝影機已連接（若需地面障礙物偵測）
    """
    import sys
    rclpy.init()

    role = sys.argv[1] if len(sys.argv) > 1 else 'robot'

    if role == 'robot':
        node = RobotNode()
        print("[啟動] TurtleBot3 探索節點已啟動")
    elif role == 'pc_server':
        node = PCComputeServer()
        print("[啟動] PC 端計算伺服器已啟動")
    else:
        print(f"未知角色: '{role}'")
        print("使用方式: python3 turtlebot3_explorer.py [robot | pc_server]")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[關閉] 收到中斷信號，正在關閉...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[關閉] 節點已安全關閉")


if __name__ == '__main__':
    main()
