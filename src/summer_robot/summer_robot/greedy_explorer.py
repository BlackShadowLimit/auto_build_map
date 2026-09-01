#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurtleBot3 Burger Gazebo 自主探索系統 v3 (APF + Blacklist)
==========================================================
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


# ── Topic 名稱 ─────────────────────────────────────────────────
TOPIC_MAP          = '/map'
TOPIC_ODOM         = '/odom'
TOPIC_SCAN         = '/scan'
TOPIC_CMDVEL       = '/cmd_vel'
TOPIC_PLAN_REQUEST = '/pc/plan_request'
TOPIC_RRT_PATH     = '/pc/rrt_path'
TOPIC_STATUS       = '/explorer/status'

MAP_FRAME   = 'map'
ROBOT_FRAME = 'base_footprint'

# ─────────────────────────────────────────────────────────────────
# 全域可調參數區
# ─────────────────────────────────────────────────────────────────

# --- Frontier 偵測 ---
FRONTIER_MIN_SIZE       = 8     # 最小 Frontier 群集大小 [cells]
FRONTIER_SEARCH_RADIUS  = 120   # 從機器人往外搜尋的最大半徑 [cells]
FRONTIER_REACH_DISTANCE = 0.4   # [v3縮小] 距 Frontier 質心此距離內視為「到達」[m]

# --- Frontier 黑名單 (防幽靈邊界卡死) ---
BLACKLIST_RADIUS   = 1.0    # 黑名單有效半徑 [m]
BLACKLIST_DURATION = 60.0   # 黑名單持續時間 [s]

# --- RRT 路徑規劃 ---
RRT_MAX_ITER             = 5000  # 最大迭代次數
RRT_STEP_SIZE            = 0.25  # 每次延伸步長 [m]
RRT_GOAL_BIAS            = 0.20  # 直接採樣目標的機率
RRT_GOAL_THRESHOLD       = 0.20  # 到達目標的距離閾值 [m]
RRT_COLLISION_CHECK_STEP = 0.04  # 碰撞插值步長 [m]
INFLATION_RADIUS         = 0.22  # 障礙物膨脹半徑 [m]

# --- 移動控制 ---
MAX_LINEAR_VEL   = 0.18   # 最大線速度 [m/s]
MAX_ANGULAR_VEL  = 1.2    # 最大角速度 [rad/s]
GOAL_TOLERANCE   = 0.15   # 路徑點到達容差 [m]
ANGULAR_KP       = 1.8    # 角度 P 增益
LINEAR_KP        = 0.5    # 線速度 P 增益
WAYPOINT_SKIP_AHEAD = 4   # 路徑跟隨向前看幾個點

# --- LiDAR 安全守衛 ---
LIDAR_SAFE_DISTANCE = 0.25   # 前方安全距離 [m]，小於此值緊急停止
LIDAR_FRONT_ANGLE   = 30.0   # 前方危險扇形半角 [度]

# --- 計算卸載 ---
OFFLOAD_ENABLED = False   # False = 本機計算（單終端機）
PLAN_COOLDOWN   = 2.0     # 重規劃最小間隔 [s]

# --- 卡住偵測 ---
STUCK_INTERVAL  = 6.0    # 偵測時間間隔 [s]
STUCK_THRESHOLD = 0.05   # 間隔內位移量 < 此值視為卡住 [m]


class RRTNode:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['RRTNode'] = None

    def distance_to(self, other: 'RRTNode') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class MapData:
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.width      = occupancy_grid.info.width
        self.height     = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution
        self.origin_x   = occupancy_grid.info.origin.position.x
        self.origin_y   = occupancy_grid.info.origin.position.y
        self.data = np.array(occupancy_grid.data, dtype=np.int16).reshape(
            (self.height, self.width)
        )

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


def find_nearest_safe_free_goal(
    map_data: MapData,
    fx: float, fy: float,
    inflated_map: np.ndarray,
    max_search_radius: int = 40
) -> Optional[Tuple[float, float]]:
    start_mx, start_my = map_data.world_to_map(fx, fy)

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

    candidates.sort(key=lambda p: (p[2], -p[3]))
    return [(p[0], p[1]) for p in candidates]


def rrt_plan(
    start_wx: float, start_wy: float,
    goal_wx: float, goal_wy: float,
    map_data: MapData,
    inflated_map: np.ndarray
) -> Optional[List[Tuple[float, float]]]:
    def _collision_free(x1, y1, x2, y2) -> bool:
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

APF_K_ATT        = 1.5
APF_K_REP        = 0.05
APF_MAX_OBS_DIST = 0.40
APF_LIDAR_STEP   = 15

class APFMotionController:
    def __init__(self, node: Node):
        self.node = node
        self.cmd_pub = node.create_publisher(Twist, TOPIC_CMDVEL, 10)
        self.path: List[Tuple[float, float]] = []
        self.waypoint_idx = 0
        self.latest_scan: Optional[LaserScan] = None

    def set_path(self, path: List[Tuple[float, float]]):
        self.path = path
        self.waypoint_idx = 0
        self.node.get_logger().info(f'[APF移動] 接收新路徑，共 {len(path)} 點')

    def set_scan(self, scan: LaserScan):
        self.latest_scan = scan

    def update(self, rx: float, ry: float, ryaw: float) -> bool:
        if not self.path or self.waypoint_idx >= len(self.path):
            self.stop()
            return True

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

        f_rep_x = 0.0
        f_rep_y = 0.0

        if self.latest_scan and self.latest_scan.ranges:
            scan = self.latest_scan
            angle_inc_deg = math.degrees(scan.angle_increment)
            if angle_inc_deg > 0:
                step_idx = max(1, int(APF_LIDAR_STEP / angle_inc_deg))
            else:
                step_idx = 10
            
            for i in range(0, len(scan.ranges), step_idx):
                r = scan.ranges[i]
                if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                    continue
                
                if r < APF_MAX_OBS_DIST:
                    angle_rad = scan.angle_min + i * scan.angle_increment
                    angle_robot = (angle_rad + math.pi) % (2*math.pi) - math.pi
                    f_mag = APF_K_REP * (1.0/r - 1.0/APF_MAX_OBS_DIST) / (r**2)
                    f_rep_x += -f_mag * math.cos(angle_robot)
                    f_rep_y += -f_mag * math.sin(angle_robot)

        f_tot_x = f_att_x + f_rep_x
        f_tot_y = f_att_y + f_rep_y

        theta_tot = math.atan2(f_tot_y, f_tot_x)
        mag_tot = math.sqrt(f_tot_x**2 + f_tot_y**2)

        factor = max(0.0, 1.0 - abs(theta_tot) / (math.pi / 2))
        lin = max(0.0, min(MAX_LINEAR_VEL, LINEAR_KP * mag_tot * factor))
        ang = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, ANGULAR_KP * theta_tot))

        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)
        return False

    def stop(self):
        self.cmd_pub.publish(Twist())

    def emergency_stop(self):
        self.stop()
        self.path = []
        self.node.get_logger().warn('[APF移動] !! 緊急停止')


# ─────────────────────────────────────────────────────────────────
# 機器人端主節點
# ─────────────────────────────────────────────────────────────────

class RobotNode(Node):
    def __init__(self):
        super().__init__('turtlebot3_explorer_v3')
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, TOPIC_MAP, self._on_map, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, TOPIC_SCAN, self._on_scan, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, TOPIC_ODOM, self._on_odom, qos_profile_sensor_data)
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

        self.is_exploring     = False
        self.exploration_done = False
        self.current_frontier: Optional[Tuple[float, float]] = None
        self.last_plan_time   = 0.0
        
        # [v3] 黑名單
        self.blacklisted_frontiers: List[Tuple[float, float, float]] = []

        self._stuck_last_time = time.time()
        self._stuck_last_pos  = (0.0, 0.0)

        self.motion_ctrl = APFMotionController(self)

        self.create_timer(0.10, self._control_loop)
        self.create_timer(1.00, self._exploration_loop)

        self.get_logger().info('=== TurtleBot3 自主探索節點 v3 (APF+黑名單) 已啟動 ===')

    def _on_map(self, msg: OccupancyGrid):
        self.current_map = msg

    def _on_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def _on_odom(self, msg: Odometry):
        pass

    def _on_path_received(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn('[路徑] PC 端回傳空路徑，重新嘗試')
            self.is_exploring = False
            return
        path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.motion_ctrl.set_path(path)
        self.is_exploring = True
        self.get_logger().info(f'[路徑] 收到 {len(path)} 個路徑點')

    def _update_pose_from_tf(self) -> bool:
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
        # 清理過期黑名單
        now = time.time()
        self.blacklisted_frontiers = [f for f in self.blacklisted_frontiers if now - f[2] < BLACKLIST_DURATION]

        if not self._update_pose_from_tf():
            return

        if self.latest_scan is not None:
            self.motion_ctrl.set_scan(self.latest_scan)
            if check_lidar_obstacle(self.latest_scan):
                self.motion_ctrl.emergency_stop()
                self.is_exploring = False
                return

        if not self.is_exploring:
            return

        if self.current_frontier is not None:
            dist = math.sqrt(
                (self.robot_x - self.current_frontier[0]) ** 2 +
                (self.robot_y - self.current_frontier[1]) ** 2
            )
            if dist < FRONTIER_REACH_DISTANCE:
                self.get_logger().info(f'[探索] ✓ 成功接近 Frontier ({dist:.2f}m < {FRONTIER_REACH_DISTANCE}m)')
                
                # [v3] 將抵達的 Frontier 加入黑名單，強迫尋找下一個
                self.blacklisted_frontiers.append((self.current_frontier[0], self.current_frontier[1], now))
                
                self.motion_ctrl.stop()
                self.is_exploring     = False
                self.current_frontier = None
                return

        path_done = self.motion_ctrl.update(self.robot_x, self.robot_y, self.robot_yaw)
        if path_done:
            self.is_exploring = False
            self.current_frontier = None

        self._check_stuck()

    def _check_stuck(self):
        now = time.time()
        if now - self._stuck_last_time < STUCK_INTERVAL:
            return

        moved = math.sqrt(
            (self.robot_x - self._stuck_last_pos[0])**2 +
            (self.robot_y - self._stuck_last_pos[1])**2
        )
        if moved < STUCK_THRESHOLD:
            self.get_logger().warn(f'[卡住偵測] {STUCK_INTERVAL} 秒內僅移動 {moved:.2f}m')
            
            # [v3] 卡住時將當前 Frontier 加入黑名單
            if self.current_frontier is not None:
                self.get_logger().warn('[黑名單] 放棄當前幽靈邊界')
                self.blacklisted_frontiers.append((self.current_frontier[0], self.current_frontier[1], now))
                self.current_frontier = None

            self.motion_ctrl.emergency_stop()
            self.is_exploring = False

        self._stuck_last_time = now
        self._stuck_last_pos  = (self.robot_x, self.robot_y)

    def _exploration_loop(self):
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

        # [v3] 傳入 blacklist
        frontiers = detect_frontiers(
            map_data, robot_mx, robot_my, 
            blacklist=[(f[0], f[1]) for f in self.blacklisted_frontiers]
        )

        if not frontiers:
            self.get_logger().info('[探索] 找不到有效 Frontier，探索完成！')
            self.exploration_done = True
            self.motion_ctrl.stop()
            self.status_pub.publish(String(data='EXPLORATION_COMPLETE'))
            return

        inflated_map = map_data.get_inflated_map(INFLATION_RADIUS)

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
                goal = find_nearest_safe_free_goal(map_data, fx, fy, inflated_map)
                if not goal:
                    continue

                gx, gy = goal
                
                if math.sqrt((gx - self.robot_x)**2 + (gy - self.robot_y)**2) < GOAL_TOLERANCE:
                    self.current_frontier = (fx, fy)
                    self.is_exploring = True
                    return

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
        self.motion_ctrl.stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────
# PC 端計算伺服器
# ─────────────────────────────────────────────────────────────────

class PCComputeServer(Node):
    def __init__(self):
        super().__init__('pc_compute_server_v3')
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        self.is_computing = False

        self.get_logger().info('=== PC 端伺服器已啟動 ===')

    def _on_map(self, msg: OccupancyGrid):
        self.current_map = msg

    def _on_plan_request(self, msg: PoseStamped):
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
