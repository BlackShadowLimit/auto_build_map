#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurtleBot3 Burger Gazebo 自主探索系統 (QRRT + APF + Blacklist)
==========================================================
這是一個基於 ROS 2 的自主探索演算法，結合了：
1. 影像二值化與連通元件分析 (OpenCV) 來尋找 Frontier (未知空間的邊界)
2. 快速擴展隨機樹 (RRT) 進行全域路徑規劃
3. 人工勢場法 (APF) 進行局部避障與路徑跟隨
4. 黑名單機制 (Blacklist) 來避免機器人卡在幽靈邊界 (牆後的死角)
"""

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, qos_profile_sensor_data)
import tf2_ros
from tf2_ros import TransformException

import numpy as np
import cv2
import math
import time
import random
from collections import deque
from typing import List, Tuple, Optional

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# ── Topic 名稱定義 ──────────────────────────────────────────────
TOPIC_MAP          = '/map'             # SLAM 產生的地圖
TOPIC_ODOM         = '/odom'            # 里程計資訊
TOPIC_SCAN         = '/scan'            # 光達掃描資訊
TOPIC_CMDVEL       = '/cmd_vel'         # 機器人速度控制指令
TOPIC_PLAN_REQUEST = '/pc/plan_request' # 傳送至 PC 端的路徑規劃請求
TOPIC_RRT_PATH     = '/pc/rrt_path'     # PC 端回傳的規劃路徑
TOPIC_STATUS       = '/explorer/status' # 探索狀態發佈

MAP_FRAME   = 'map'             # 絕對地圖座標系
ROBOT_FRAME = 'base_footprint'  # 機器人中心座標系

# ─────────────────────────────────────────────────────────────────
# 全域可調參數區
# ─────────────────────────────────────────────────────────────────

# --- Frontier (邊界) 偵測參數 ---
FRONTIER_MIN_SIZE       = 8     # 最小 Frontier 群集大小 [cells] (過濾掉太小的雜訊)
FRONTIER_SEARCH_RADIUS  = 600   # 從機器人往外搜尋的最大半徑 [cells]，600 * 0.05m = 30m (設定夠大才能看遍全圖)
FRONTIER_REACH_DISTANCE = 0.4   # 距 Frontier 質心此距離內即視為「到達」[m] (設定較小以逼迫機器人靠近邊角)

# --- Frontier 黑名單 (防幽靈邊界卡死機制) ---
# 當機器人抵達邊界附近或卡住時，若雷達無法掃到該處(通常是被牆擋住)，將會暫時將此點加入黑名單，強迫前往其他區域。
BLACKLIST_RADIUS   = 1.0    # 黑名單有效半徑 [m]，此半徑內的 Frontier 暫時忽略
BLACKLIST_DURATION = 60.0   # 黑名單持續時間 [s]

# --- RRT 全域路徑規劃參數 ---
RRT_MAX_ITER             = 5000  # 最大迭代次數，避免無解時無窮迴圈
RRT_STEP_SIZE            = 0.25  # 每次樹枝延伸的步長 [m]
RRT_GOAL_BIAS            = 0.20  # 直接朝目標點採樣的機率 (0~1)，提高收斂速度
RRT_GOAL_THRESHOLD       = 0.20  # 距離目標多近視為規劃成功 [m]
RRT_COLLISION_CHECK_STEP = 0.04  # 兩點間碰撞檢查的插值步長 [m]
INFLATION_RADIUS         = 0.25  # 地圖障礙物膨脹半徑 [m]，確保 RRT 規劃出來的路線離牆壁有安全距離

# --- PID 移動控制參數 (結合 APF 使用) ---
MAX_LINEAR_VEL   = 0.18   # 最大前進線速度 [m/s]
MAX_ANGULAR_VEL  = 1.2    # 最大旋轉角速度 [rad/s]
GOAL_TOLERANCE   = 0.15   # 判定到達中途路徑點的容差 [m]
ANGULAR_KP       = 1.8    # 轉向的 P 控制器增益
LINEAR_KP        = 0.5    # 前進的 P 控制器增益
WAYPOINT_SKIP_AHEAD = 4   # 循跡時，往前方幾個路徑點看(Look-ahead)，讓走線更平順

# --- LiDAR 安全急煞守衛 ---
# 保護機制，當雷達偵測到過近的物體時強制煞停。此距離必須小於 INFLATION_RADIUS，避免與演算法互相衝突。
LIDAR_SAFE_DISTANCE = 0.18   # 前方安全距離 [m]
LIDAR_FRONT_ANGLE   = 30.0   # 偵測前方危險的扇形半角 [度] (正前方左右各 30 度)

# --- 計算量卸載 ---
OFFLOAD_ENABLED = False   # 是否將 RRT 計算卸載到 PC 端 (Gazebo 預設在本機計算即可)
PLAN_COOLDOWN   = 2.0     # 重新規劃路線的最小冷卻時間 [s]

# --- 卡住偵測參數 ---
STUCK_INTERVAL  = 6.0    # 偵測卡住的時間區間 [s]
STUCK_THRESHOLD = 0.05   # 在該區間內位移小於此值 [m]，則視為卡住

# --- 急煞升級機制 ---
# 第一次雷達急煞：放棄當前 Frontier，讓 exploration_loop 改選距離更遠的下一個目標。
# 連續急煞達到此閾值：直接將 Frontier 加入黑名單，跳過更遠的目標再繼續探索。
EMERGENCY_BLACKLIST_THRESHOLD = 2  # 連續急煞幾次後才加入黑名單


class RRTNode:
    """RRT 樹節點結構"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['RRTNode'] = None

    def distance_to(self, other: 'RRTNode') -> float:
        """計算與另一節點的直線距離"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class MapData:
    """封裝 OccupancyGrid 地圖資料，提供坐標轉換與碰撞檢查工具"""
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.width      = occupancy_grid.info.width
        self.height     = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution
        self.origin_x   = occupancy_grid.info.origin.position.x
        self.origin_y   = occupancy_grid.info.origin.position.y
        # 將一維陣列轉換為 2D (height x width)，型態轉為 int16 方便處理 -1
        self.data = np.array(occupancy_grid.data, dtype=np.int16).reshape(
            (self.height, self.width)
        )

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        """將真實世界座標 (m) 轉換為地圖陣列索引 (pixels)"""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """將地圖陣列索引 (pixels) 轉換為真實世界座標 (m)"""
        wx = mx * self.resolution + self.origin_x + self.resolution / 2.0
        wy = my * self.resolution + self.origin_y + self.resolution / 2.0
        return wx, wy

    def is_free(self, mx: int, my: int) -> bool:
        """判斷該網格是否為已知且無障礙 (0 <= val < 50)"""
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return False
        return 0 <= self.data[my, mx] < 50

    def get_inflated_map(self, inflation_m: float) -> np.ndarray:
        """使用 OpenCV 將障礙物膨脹，產生安全邊界遮罩，回傳布林陣列"""
        inflation_cells = max(1, int(inflation_m / self.resolution))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_cells + 1, 2 * inflation_cells + 1)
        )
        occupied_mask = (self.data >= 50).astype(np.uint8)
        inflated_occupied = cv2.dilate(occupied_mask, kernel)
        unknown_mask = (self.data < 0).astype(np.uint8)
        # 障礙物與未知空間都視為不可通行
        return (inflated_occupied | unknown_mask).astype(bool)


def find_nearest_safe_free_goal(
    map_data: MapData,
    fx: float, fy: float,
    inflated_map: np.ndarray,
    max_search_radius: int = 40
) -> Optional[Tuple[float, float]]:
    """
    因為 Frontier(未知邊界) 通常落在未知空間，RRT 無法直接連到未知的網格上。
    此函數利用 BFS (廣度優先搜尋)，從 Frontier 出發，尋找最近的一個「安全且已知」的自由網格作為導航終點。
    """
    start_mx, start_my = map_data.world_to_map(fx, fy)

    # 若本身就是安全的自由網格，直接回傳
    if (0 <= start_mx < map_data.width and
        0 <= start_my < map_data.height and
        map_data.is_free(start_mx, start_my) and
            not inflated_map[start_my, start_mx]):
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


def detect_frontiers(
    map_data: MapData,
    robot_mx: int,
    robot_my: int,
    blacklist: List[Tuple[float, float]] = None
) -> List[Tuple[float, float]]:
    """
    影像處理核心：將佔據網格地圖(Occupancy Grid)轉換為影像，並找出 Frontier (邊界)。
    流程：
      1. 找出所有自由空間 (Free) 與 未知空間 (Unknown)
      2. 膨脹自由空間，讓它碰到周圍的格子
      3. 取「膨脹後的自由空間」與「未知空間」的交集 (Bitwise AND) -> 即為 Frontier 邊界線
      4. 找連通塊 (Connected Components)，過濾雜訊，並計算質心
      5. 濾除落在黑名單半徑內的質心，防止卡死
    """
    if blacklist is None:
        blacklist = []

    grid = map_data.data
    free_mask    = ((grid >= 0) & (grid < 50)).astype(np.uint8) * 255
    unknown_mask = (grid < 0).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    free_dilated = cv2.dilate(free_mask, kernel, iterations=1)
    frontier_raw = cv2.bitwise_and(free_dilated, unknown_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        frontier_raw, connectivity=8
    )

    candidates = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < FRONTIER_MIN_SIZE:
            continue

        cx, cy = int(centroids[lid, 0]), int(centroids[lid, 1])
        dist_cells = math.sqrt((cx - robot_mx) ** 2 + (cy - robot_my) ** 2)
        if dist_cells > FRONTIER_SEARCH_RADIUS:
            continue

        wx, wy = map_data.map_to_world(cx, cy)
        
        # 黑名單檢查
        is_blacklisted = False
        for bx, by in blacklist:
            if math.sqrt((wx - bx)**2 + (wy - by)**2) < BLACKLIST_RADIUS:
                is_blacklisted = True
                break
        
        if not is_blacklisted:
            candidates.append((wx, wy, dist_cells, area))

    if not candidates:
        return []

    # 依照距離遠近排序，優先前往最近的未探索區域
    candidates.sort(key=lambda p: (p[2], -p[3]))
    return [(p[0], p[1]) for p in candidates]


def rrt_plan(
    start_wx: float, start_wy: float,
    goal_wx: float, goal_wy: float,
    map_data: MapData,
    inflated_map: np.ndarray
) -> Optional[List[Tuple[float, float]]]:
    """
    快速擴展隨機樹 (Rapidly-exploring Random Tree) 全域路徑規劃
    在充滿障礙物的地圖上隨機採樣，長出一棵避開障礙物的樹，直到連接起點與終點。
    """
    def _collision_free(x1, y1, x2, y2) -> bool:
        """檢查兩點連線間是否會撞到膨脹後的障礙物"""
        dist  = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        steps = max(1, int(dist / RRT_COLLISION_CHECK_STEP))
        for i in range(steps + 1):
            t  = i / steps
            ix = x1 + t*(x2-x1)
            iy = y1 + t*(y2-y1)
            mx, my = map_data.world_to_map(ix, iy)
            if not (0 <= mx < map_data.width and 0 <= my < map_data.height):
                return False
            if inflated_map[my, mx]:
                return False
        return True

    def _nearest(tree, sample):
        return min(tree, key=lambda n: n.distance_to(sample))

    def _steer(from_n, to_n):
        d = from_n.distance_to(to_n)
        if d <= RRT_STEP_SIZE:
            return RRTNode(to_n.x, to_n.y)
        r  = RRT_STEP_SIZE / d
        return RRTNode(from_n.x + r*(to_n.x-from_n.x),
                       from_n.y + r*(to_n.y-from_n.y))

    map_min_x = map_data.origin_x
    map_max_x = map_data.origin_x + map_data.width  * map_data.resolution
    map_min_y = map_data.origin_y
    map_max_y = map_data.origin_y + map_data.height * map_data.resolution

    start = RRTNode(start_wx, start_wy)
    goal  = RRTNode(goal_wx,  goal_wy)
    tree: List[RRTNode] = [start]

    for _ in range(RRT_MAX_ITER):
        if random.random() < RRT_GOAL_BIAS:
            sample = RRTNode(goal_wx, goal_wy)
        else:
            sample = RRTNode(random.uniform(map_min_x, map_max_x),
                             random.uniform(map_min_y, map_max_y))

        nearest  = _nearest(tree, sample)
        new_node = _steer(nearest, sample)

        if not _collision_free(nearest.x, nearest.y, new_node.x, new_node.y):
            continue

        new_node.parent = nearest
        tree.append(new_node)

        if new_node.distance_to(goal) <= RRT_GOAL_THRESHOLD:
            path = []
            node: Optional[RRTNode] = new_node
            while node:
                path.append((node.x, node.y))
                node = node.parent
            path.reverse()
            return path
    return None


def smooth_path(
    path: List[Tuple[float, float]],
    map_data: MapData,
    inflated_map: Optional[np.ndarray] = None
) -> List[Tuple[float, float]]:
    """
    貪婪路徑平滑器 (Greedy Path Smoother)
    RRT 產生的路徑通常是鋸齒狀的，此方法會嘗試截彎取直，拿掉不必要的轉折點。
    """
    if len(path) <= 2:
        return path
    if inflated_map is None:
        inflated_map = map_data.get_inflated_map(INFLATION_RADIUS)

    def _los(x1, y1, x2, y2) -> bool:
        dist  = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        steps = max(1, int(dist / RRT_COLLISION_CHECK_STEP))
        for i in range(steps + 1):
            t  = i / steps
            ix = x1 + t*(x2-x1)
            iy = y1 + t*(y2-y1)
            mx, my = map_data.world_to_map(ix, iy)
            if not (0 <= mx < map_data.width and 0 <= my < map_data.height):
                return False
            if inflated_map[my, mx]:
                return False
        return True

    smoothed   = [path[0]]
    current    = 0
    while current < len(path) - 1:
        furthest = current + 1
        for j in range(len(path)-1, current, -1):
            if _los(path[current][0], path[current][1],
                    path[j][0],       path[j][1]):
                furthest = j
                break
        smoothed.append(path[furthest])
        current = furthest
    return smoothed


def check_lidar_obstacle(scan: LaserScan) -> bool:
    """檢查雷達正前方是否有距離過近的障礙物，用於緊急煞停"""
    if not scan.ranges:
        return False
    front_angle_rad = math.radians(LIDAR_FRONT_ANGLE)
    two_pi = 2.0 * math.pi
    for i, r in enumerate(scan.ranges):
        if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
            continue
        angle = scan.angle_min + i * scan.angle_increment
        if angle < front_angle_rad or angle > (two_pi - front_angle_rad):
            if r < LIDAR_SAFE_DISTANCE:
                return True
    return False

# ─────────────────────────────────────────────────────────────────
# 模組 5：APF (人工勢場法) 移動控制器
# ─────────────────────────────────────────────────────────────────

# --- APF 專用參數 ---
APF_K_ATT        = 1.5    # 目標點引力增益 (越大越急著往目標走)
APF_K_REP        = 0.05   # 障礙物斥力增益 (越大遇到障礙彈開越遠)
APF_MAX_OBS_DIST = 0.40   # 斥力場作用半徑 [m] (雷達距離大於此值的障礙物不產生斥力)
APF_LIDAR_STEP   = 15     # 降採樣雷達資訊的度數 (一圈360度，每15度取一個點，共24點)

class APFMotionController:
    """
    結合 APF (人工勢場法) 與 P (比例) 控制器的運動模型。
    引力 (Attractive Force)：朝向 RRT 規劃的下一個路徑點。
    斥力 (Repulsive Force)：由降採樣的雷達點產生，將機器人推離周圍牆壁。
    合力：兩者向量相加，算出最終的前進方向與速度。
    """
    def __init__(self, node: Node):
        self.node = node
        self.cmd_pub = node.create_publisher(Twist, TOPIC_CMDVEL, 10)
        self.path: List[Tuple[float, float]] = []   # 目前追蹤中的路徑點列表 (世界座標)
        self.waypoint_idx = 0                        # 目前正在追蹤的路徑點索引
        self.latest_scan: Optional[LaserScan] = None # 最新的雷達掃描資料，供斥力計算使用

    def set_path(self, path: List[Tuple[float, float]]):
        """
        設定一條新的全域路徑供 APF 追蹤。
        通常由 _exploration_loop (本機計算) 或 _on_path_received (PC端回傳) 呼叫。
        每次呼叫都會重置路徑索引從頭開始追蹤。
        """
        self.path = path
        self.waypoint_idx = 0
        self.node.get_logger().info(f'[APF移動] 接收新路徑，共 {len(path)} 點')

    def set_scan(self, scan: LaserScan):
        """更新最新的雷達掃描資料。由 _control_loop 每 0.1s 呼叫一次，確保斥力計算使用最新數據。"""
        self.latest_scan = scan


    def update(self, rx: float, ry: float, ryaw: float) -> bool:
        """
        APF 主更新函式，每個控制週期（10Hz）呼叫一次。
        根據當前機器人位姿 (rx, ry, ryaw) 計算合力並輸出 Twist 速度指令。

        Args:
            rx (float): 機器人在地圖座標系下的 X 位置 [m]
            ry (float): 機器人在地圖座標系下的 Y 位置 [m]
            ryaw (float): 機器人當前朝向角 [rad]

        Returns:
            bool: True 代表路徑已走完（呼叫端可重新觸發探索），False 代表仍在追蹤中
        """
        if not self.path or self.waypoint_idx >= len(self.path):
            self.stop()
            return True

        # Look-ahead: 尋找前方幾個可以跳過的點，讓走線更順
        look_ahead = min(self.waypoint_idx + WAYPOINT_SKIP_AHEAD, len(self.path) - 1)
        for i in range(look_ahead, self.waypoint_idx, -1):
            wx, wy = self.path[i]
            if math.sqrt((wx-rx)**2 + (wy-ry)**2) <= GOAL_TOLERANCE:
                self.waypoint_idx = i + 1
                break

        if self.waypoint_idx >= len(self.path):
            self.stop()
            return True

        tx, ty = self.path[self.waypoint_idx]
        
        # 1. 計算引力向量
        dx_world = tx - rx
        dy_world = ty - ry
        dist_to_target = math.sqrt(dx_world**2 + dy_world**2)

        if dist_to_target < GOAL_TOLERANCE:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.path):
                self.stop()
                return True

        target_angle_world = math.atan2(dy_world, dx_world)
        target_angle_robot = (target_angle_world - ryaw + math.pi) % (2*math.pi) - math.pi
        
        f_att_x = APF_K_ATT * dist_to_target * math.cos(target_angle_robot)
        f_att_y = APF_K_ATT * dist_to_target * math.sin(target_angle_robot)

        # 2. 計算斥力向量
        f_rep_x = 0.0
        f_rep_y = 0.0

        if self.latest_scan and self.latest_scan.ranges:
            scan = self.latest_scan
            angle_inc_deg = math.degrees(scan.angle_increment)
            if angle_inc_deg > 0:
                step_idx = max(1, int(APF_LIDAR_STEP / angle_inc_deg))
            else:
                step_idx = 10
            
            # 從光達陣列中取連續且離散的點
            for i in range(0, len(scan.ranges), step_idx):
                r = scan.ranges[i]
                if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                    continue
                
                # 只有過於接近的障礙物才會產生斥力
                if r < APF_MAX_OBS_DIST:
                    angle_rad = scan.angle_min + i * scan.angle_increment
                    angle_robot = (angle_rad + math.pi) % (2*math.pi) - math.pi
                    
                    # 勢場公式：越近斥力呈指數型暴增
                    f_mag = APF_K_REP * (1.0/r - 1.0/APF_MAX_OBS_DIST) / (r**2)
                    f_rep_x += -f_mag * math.cos(angle_robot)
                    f_rep_y += -f_mag * math.sin(angle_robot)

        # 3. 合成最終向量
        f_tot_x = f_att_x + f_rep_x
        f_tot_y = f_att_y + f_rep_y

        theta_tot = math.atan2(f_tot_y, f_tot_x)
        mag_tot = math.sqrt(f_tot_x**2 + f_tot_y**2)

        # 如果方向偏離太多(需要大轉彎)，則 factor 會趨近 0，讓線速度減慢，以利原地轉向
        factor = max(0.0, 1.0 - abs(theta_tot) / (math.pi / 2))
        lin = max(0.0, min(MAX_LINEAR_VEL, LINEAR_KP * mag_tot * factor))
        ang = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, ANGULAR_KP * theta_tot))

        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)
        return False

    def stop(self):
        """發送零速度指令，讓機器人立刻停止（正常停止，不清除路徑）。"""
        self.cmd_pub.publish(Twist())

    def emergency_stop(self):
        """
        緊急停止：立即停止馬達並清空路徑佇列。
        通常在雷達偵測到前方有障礙物、或卡住偵測觸發時呼叫。
        清空路徑後，_control_loop 會在下次迭代中讓 _exploration_loop 重新規劃。
        """
        self.stop()
        self.path = []
        self.node.get_logger().warn('[APF移動] !! 緊急停止')


# ─────────────────────────────────────────────────────────────────
# 機器人端主節點
# ─────────────────────────────────────────────────────────────────

class RobotNode(Node):
    """
    負責機器人的主控制迴圈。包含：
    1. 從 TF 樹讀取位姿
    2. 檢查卡住狀態
    3. 管理黑名單
    4. 呼叫 Frontier 探索並觸發 RRT
    """
    def __init__(self):
        super().__init__('turtlebot3_explorer_v3')
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 地圖 Topic 使用 RELIABLE + TRANSIENT_LOCAL，確保訂閱後能立即收到上一次發佈的地圖快照
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, TOPIC_MAP, self._on_map, map_qos)
        # 感測器資料 (scan/odom) 使用 BestEffort QoS，因為 Cartographer 預設以此 QoS 發佈
        self.scan_sub = self.create_subscription(
            LaserScan, TOPIC_SCAN, self._on_scan, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, TOPIC_ODOM, self._on_odom, qos_profile_sensor_data)
        # 若 OFFLOAD_ENABLED=True，此 subscriber 用於接收 PC 端回傳的路徑
        self.path_sub = self.create_subscription(
            Path, TOPIC_RRT_PATH, self._on_path_received, 10)

        self.plan_request_pub = self.create_publisher(PoseStamped, TOPIC_PLAN_REQUEST, 10)
        self.status_pub       = self.create_publisher(String, TOPIC_STATUS, 10)


        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.tf_ready  = False

        self.current_map: Optional[OccupancyGrid] = None
        self.latest_scan: Optional[LaserScan]     = None

        # 狀態機變數
        self.is_exploring     = False
        self.exploration_done = False
        self.current_frontier: Optional[Tuple[float, float]] = None
        self.last_plan_time   = 0.0
        
        # 黑名單儲存器: list of (x, y, timestamp)
        self.blacklisted_frontiers: List[Tuple[float, float, float]] = []
        
        # 急煞計數器：記錄對每個 Frontier（量化鍵）連續觸發雷達急煞的次數
        # 格式：{(round_x, round_y): count}，超過 EMERGENCY_BLACKLIST_THRESHOLD 才加黑名單
        self._emergency_stop_counts: dict = {}

        self._stuck_last_time = time.time()
        self._stuck_last_pos  = (0.0, 0.0)

        self.motion_ctrl = APFMotionController(self)

        self.create_timer(0.10, self._control_loop)      # 10 Hz 馬達控制
        self.create_timer(1.00, self._exploration_loop)  # 1 Hz 探索規劃

        self.get_logger().info('=== TurtleBot3 自主探索節點 v3 (APF+黑名單) 已啟動 ===')

    def _on_map(self, msg: OccupancyGrid):
        """快取最新地圖資料，供 _exploration_loop 進行 Frontier 偵測與 RRT 規劃使用。"""
        self.current_map = msg

    def _on_scan(self, msg: LaserScan):
        """快取最新雷達掃描資料，供 _control_loop 急煞守衛與 APF 斥力計算使用。"""
        self.latest_scan = msg

    def _on_odom(self, msg: Odometry):
        # 里程計資料在本系統中由 TF 統一處理，此回呼保留供未來擴充使用（例如速度估計）
        pass

    def _on_path_received(self, msg: Path):
        """若開啟 OFFLOAD，接收來自 PC 伺服器的路徑"""
        if not msg.poses:
            self.get_logger().warn('[路徑] PC 端回傳空路徑，重新嘗試')
            self.is_exploring = False
            return
        path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.motion_ctrl.set_path(path)
        self.is_exploring = True
        self.get_logger().info(f'[路徑] 收到 {len(path)} 個路徑點')

    def _update_pose_from_tf(self) -> bool:
        """從 Cartographer 發布的 TF 樹取得機器人在地圖中的絕對坐標"""
        try:
            t = self.tf_buffer.lookup_transform(
                MAP_FRAME, ROBOT_FRAME,
                rclpy.time.Time()
            )
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            q = t.transform.rotation
            self.robot_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            self.tf_ready = True
            return True
        except TransformException:
            return False

    def _control_loop(self):
        """高頻 (10Hz) 迴圈，負責發送速度指令、急煞、以及處理到達 Frontier 的邏輯"""
        # 清理過期的黑名單
        now = time.time()
        self.blacklisted_frontiers = [f for f in self.blacklisted_frontiers if now - f[2] < BLACKLIST_DURATION]

        if not self._update_pose_from_tf():
            return

        # 雷達安全急煞機制（升級版：首次跳過，連續才加黑名單）
        if self.latest_scan is not None:
            self.motion_ctrl.set_scan(self.latest_scan)
            if check_lidar_obstacle(self.latest_scan):
                self.motion_ctrl.emergency_stop()
                self.is_exploring = False

                if self.current_frontier is not None:
                    # 用量化座標 (0.5m 精度) 作為計數器的 key，避免因浮點數差異重複計數
                    key = (round(self.current_frontier[0] * 2) / 2,
                           round(self.current_frontier[1] * 2) / 2)
                    self._emergency_stop_counts[key] = self._emergency_stop_counts.get(key, 0) + 1
                    count = self._emergency_stop_counts[key]

                    if count >= EMERGENCY_BLACKLIST_THRESHOLD:
                        # 連續急煞達閾值：直接加黑名單，exploration_loop 會跳到更遠的 Frontier
                        self.get_logger().warn(
                            f'[急煞升級] Frontier ({self.current_frontier[0]:.2f}, {self.current_frontier[1]:.2f}) '
                            f'連續急煞 {count} 次，加入黑名單'
                        )
                        self.blacklisted_frontiers.append(
                            (self.current_frontier[0], self.current_frontier[1], now)
                        )
                        del self._emergency_stop_counts[key]  # 清除計數，避免黑名單到期後立刻再累積
                    else:
                        # 第一次急煞：只放棄當前 Frontier，讓 exploration_loop 改選下一個更遠的目標
                        self.get_logger().warn(
                            f'[急煞] 放棄當前 Frontier，改選下一個目標 (第 {count} 次，閾值 {EMERGENCY_BLACKLIST_THRESHOLD})'
                        )

                    # 無論如何都清空當前 Frontier，確保 exploration_loop 重新選目標
                    self.current_frontier = None
                return

        if not self.is_exploring:
            return

        # 檢查是否已到達正在前往的 Frontier
        if self.current_frontier is not None:
            dist = math.sqrt(
                (self.robot_x - self.current_frontier[0]) ** 2 +
                (self.robot_y - self.current_frontier[1]) ** 2
            )
            if dist < FRONTIER_REACH_DISTANCE:
                self.get_logger().info(f'[探索] ✓ 成功接近 Frontier ({dist:.2f}m < {FRONTIER_REACH_DISTANCE}m)')
                
                # 即使抵達，也將它加入黑名單一小段時間。
                # 理由：若雷達已經照出該區不是未知空間，它自然不會再次被選上；
                # 若雷達被牆壁擋住照不出來，這招能強制放棄它，打破卡死無限迴圈！
                self.blacklisted_frontiers.append((self.current_frontier[0], self.current_frontier[1], now))
                
                # 成功抵達後清除此點的急煞計數，避免下次重訪時沿用舊計數
                key = (round(self.current_frontier[0] * 2) / 2,
                       round(self.current_frontier[1] * 2) / 2)
                self._emergency_stop_counts.pop(key, None)

                self.motion_ctrl.stop()
                self.is_exploring     = False
                self.current_frontier = None
                return

        # APF 更新馬達指令
        path_done = self.motion_ctrl.update(self.robot_x, self.robot_y, self.robot_yaw)
        if path_done:
            self.is_exploring = False
            self.current_frontier = None

        self._check_stuck()

    def _check_stuck(self):
        """定期檢查機器人是否因為地形或 APF 斥力與引力互相抵銷而卡在原地"""
        now = time.time()
        if now - self._stuck_last_time < STUCK_INTERVAL:
            return

        moved = math.sqrt(
            (self.robot_x - self._stuck_last_pos[0])**2 +
            (self.robot_y - self._stuck_last_pos[1])**2
        )
        if moved < STUCK_THRESHOLD:
            self.get_logger().warn(f'[卡住偵測] {STUCK_INTERVAL} 秒內僅移動 {moved:.2f}m')
            
            # 若判斷卡住，直接將當下的目標邊界送入黑名單，強制改道
            if self.current_frontier is not None:
                self.get_logger().warn('[黑名單] 放棄當前幽靈邊界/卡死點')
                self.blacklisted_frontiers.append((self.current_frontier[0], self.current_frontier[1], now))
                self.current_frontier = None

            self.motion_ctrl.emergency_stop()
            self.is_exploring = False

        self._stuck_last_time = now
        self._stuck_last_pos  = (self.robot_x, self.robot_y)

    def _exploration_loop(self):
        """低頻 (1Hz) 迴圈，負責掃描地圖、找出邊界，並啟動全域規劃"""
        if self.exploration_done or self.is_exploring:
            return
        if not self.current_map or not self.tf_ready:
            return

        now = time.time()
        if now - self.last_plan_time < PLAN_COOLDOWN:
            return
        self.last_plan_time = now

        map_data = MapData(self.current_map)
        robot_mx, robot_my = map_data.world_to_map(self.robot_x, self.robot_y)

        # 找尋邊界，並自動過濾掉黑名單內的座標
        frontiers = detect_frontiers(
            map_data, robot_mx, robot_my, 
            blacklist=[(f[0], f[1]) for f in self.blacklisted_frontiers]
        )

        # 如果連黑名單外的邊界都找不到了，就認定探索完成
        if not frontiers:
            self.get_logger().info('[探索] 找不到有效 Frontier，探索完成！')
            self.exploration_done = True
            self.motion_ctrl.stop()
            self.status_pub.publish(String(data='EXPLORATION_COMPLETE'))
            return

        inflated_map = map_data.get_inflated_map(INFLATION_RADIUS)

        # 依設定決定是在本機算 RRT 還是拋給 PC 算
        if OFFLOAD_ENABLED:
            goal_fx, goal_fy = frontiers[0]
            self.current_frontier = (goal_fx, goal_fy)

            req = PoseStamped()
            req.header.stamp = self.get_clock().now().to_msg()
            req.header.frame_id = MAP_FRAME
            req.pose.position.x = goal_fx
            req.pose.position.y = goal_fy
            req.pose.orientation.w = 1.0
            self.plan_request_pub.publish(req)
            self.get_logger().info(f'[探索] 請求 PC 端規劃至 ({goal_fx:.2f}, {goal_fy:.2f})')
        else:
            for fx, fy in frontiers:
                # 由於邊界在未探索區域，先找周圍的「安全自由格」作為 RRT 終點
                goal = find_nearest_safe_free_goal(map_data, fx, fy, inflated_map)
                if not goal:
                    continue

                gx, gy = goal
                
                # 如果安全自由格就在腳下，代表已經非常近了
                if math.sqrt((gx - self.robot_x)**2 + (gy - self.robot_y)**2) < GOAL_TOLERANCE:
                    self.current_frontier = (fx, fy)
                    self.is_exploring = True
                    return

                # 開始本機 RRT 規劃
                path = rrt_plan(
                    self.robot_x, self.robot_y,
                    gx, gy,
                    map_data, inflated_map
                )
                if not path:
                    continue

                smooth = smooth_path(path, map_data, inflated_map)
                self.motion_ctrl.set_path(smooth)
                self.current_frontier = (fx, fy)
                self.is_exploring = True
                self.get_logger().info(f'[探索] ✓ 本機規劃成功，前往 ({fx:.2f}, {fy:.2f})')
                return

    def destroy_node(self):
        """節點關閉時的清理函式。確保停止馬達後再呼叫父類別銷毀，避免機器人在 shutdown 時繼續移動。"""
        self.motion_ctrl.stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────
# PC 端計算伺服器 (若 OFFLOAD_ENABLED=True 時才需要執行)
# ─────────────────────────────────────────────────────────────────

class PCComputeServer(Node):
    """
    接收 Robot 發來的起終點要求，在 PC 上執行 RRT 運算並回傳。
    實體機時通常會使用此節點，降低 Raspberry Pi 等弱算力開發板的負擔。
    """
    def __init__(self):
        super().__init__('pc_compute_server_v3')
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 與 RobotNode 相同，需使用 TRANSIENT_LOCAL 才能接收到先前發佈的地圖快照
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, TOPIC_MAP, self._on_map, map_qos)
        self.req_sub = self.create_subscription(
            PoseStamped, TOPIC_PLAN_REQUEST, self._on_plan_request, 10)

        self.path_pub = self.create_publisher(Path, TOPIC_RRT_PATH, 10)

        self.current_map: Optional[OccupancyGrid] = None
        self.is_computing = False  # 防止同時接收多個規劃請求，確保運算是序列化的

        self.get_logger().info('=== PC 端伺服器已啟動 ===')

    def _on_map(self, msg: OccupancyGrid):
        """快取地圖，供 _on_plan_request 進行 RRT 規劃使用。"""
        self.current_map = msg

    def _on_plan_request(self, msg: PoseStamped):
        """
        核心：接收 Robot 端的目標點請求，執行以下流程：
          1. 從 TF 樹取得機器人當前位置作為 RRT 起點
          2. 透過 find_nearest_safe_free_goal 將原始 Frontier 座標轉換為可導航的安全目標格
          3. 執行 RRT 全域規劃並平滑路徑
          4. 透過 _publish_path 將結果回傳給 Robot 端
        若目前正在計算中 (is_computing=True)，則忽略新請求避免重疊。
        """
        if self.is_computing or not self.current_map:
            return

        try:
            t = self.tf_buffer.lookup_transform(
                MAP_FRAME, ROBOT_FRAME,
                rclpy.time.Time()
            )
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
        except TransformException:
            self.get_logger().warn('[PC] TF 取得失敗，無法計算路徑')
            return

        self.is_computing = True
        try:
            goal_fx = msg.pose.position.x
            goal_fy = msg.pose.position.y

            self.get_logger().info(f'[PC] 開始為目標 ({goal_fx:.2f}, {goal_fy:.2f}) 規劃路徑...')
            map_data = MapData(self.current_map)
            inflated_map = map_data.get_inflated_map(INFLATION_RADIUS)

            goal = find_nearest_safe_free_goal(map_data, goal_fx, goal_fy, inflated_map)
            if not goal:
                self.get_logger().warn('[PC] 找不到安全目標格，規劃失敗')
                self._publish_path([])
                return

            gx, gy = goal
            path = rrt_plan(robot_x, robot_y, gx, gy, map_data, inflated_map)

            if path:
                smooth = smooth_path(path, map_data, inflated_map)
                self.get_logger().info(f'[PC] 規劃成功: {len(smooth)} 個路徑點')
                self._publish_path(smooth)
            else:
                self.get_logger().warn('[PC] RRT 規劃失敗')
                self._publish_path([])
        finally:
            self.is_computing = False

    def _publish_path(self, path: List[Tuple[float, float]]):
        """
        將世界座標路徑列表打包成 nav_msgs/Path 訊息並發佈。
        若傳入空列表，Robot 端的 _on_path_received 會收到空路徑並重新觸發探索規劃。
        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = MAP_FRAME
        for wx, wy in path:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = wx
            p.pose.position.y = wy
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)

def main():
    import sys
    rclpy.init()
    # 支援指令列參數選擇要啟動 robot 本體還是 pc_server
    role = sys.argv[1] if len(sys.argv) > 1 else 'robot'
    
    if role == 'robot': node = RobotNode()
    elif role == 'pc_server': node = PCComputeServer()
    else: return

    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
