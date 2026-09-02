#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurtleBot3 極簡自主探索 v4  (Minimal APF Explorer)
====================================================
演算法由四個核心模組組成：

  1. 地圖更新 (SLAM)
     訂閱 /map (OccupancyGrid)，持續接收由 Cartographer 建構的
     2D 佔用網格地圖（已知空地 / 已知障礙 / 未知區域）。

  2. 提取航點 (Frontier Detection)
     掃描地圖矩陣，找出「已知空地」與「未知區域」的交界線。
     計算各連通邊界群集的質心，選擇最近且面積最大的目標。

  3. 導航與避障 (APF — 人工勢場法)
     - 引力：從機器人指向目標的向量，促使機器人前進。
     - 斥力：每個 LiDAR 點作為獨立的「離散斥力點」，
             距離越近斥力越大，足以讓合力方向偏離障礙物。
     - 前進緊急制停：若前方扇形 (±40°) 內有障礙物過近，
             直接將線速度歸零，確保機器人轉向後才前進。
     → 合力方向轉為角速度，輸出 /cmd_vel。

  4. 脫困機制 (Anti-Stuck)
     若機器人在 3 秒內位移 < 4cm，視為卡死；
     強制低速旋轉逃脫，並將目標暫時加入黑名單。
     若同一目標連續觸發 2 次逃脫，永久拉黑 60 秒。

注意事項：
  - 規劃位置 (map_x/map_y) 僅由 TF (map→base_footprint) 更新，
    確保 Frontier 距離判斷在正確座標系下進行。
  - 移動用位置 (robot_x/robot_y) TF 優先，/odom 備援，
    TF 未就緒時機器人等待但不執行規劃，避免過早宣告完成。
  - 所有旋轉速度刻意調低，減少 SLAM 地圖重影（ghosting）。
"""

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import cv2
import math
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException


# =============================================================================
# 全域可調參數
# =============================================================================

# --- Frontier 偵測 ---
FRONTIER_MIN_AREA   = 10    # [cells] 最小連通邊界面積，過濾雜訊
FRONTIER_MAX_DIST   = 60.0  # [m]     只考慮此距離內的 Frontier
FRONTIER_REACH_DIST = 0.45  # [m]     進入此範圍視為「到達目標」

# --- APF：引力 ---
APF_K_ATT     = 1.0   # 引力增益（固定大小，方向為機器人→目標的單位向量）

# --- APF：斥力（每個 LiDAR 點作為獨立離散斥力點）---
APF_K_REP     = 0.30  # 斥力增益，值越大機器人越怕靠近障礙物
APF_REP_RANGE = 0.50  # [m] 斥力影響半徑（此距離以外的障礙物不產生斥力）
APF_REP_ANGLE = 120   # [度] 斥力作用的前方扇形角度（共 ±60°）
APF_LIDAR_STEP = 10   # [度] LiDAR 降採樣步長（10° = 36 個離散斥力點）

# --- APF：前方緊急制停 ---
# 若前方 ±40° 有障礙物進入此距離，線速度歸零，強迫先轉向再前進
APF_SAFETY_DIST = 0.28  # [m]

# --- 速度限制（刻意調低以減少 SLAM ghosting）---
MAX_LINEAR   = 0.12   # [m/s]   前進最大線速度
MAX_ANGULAR  = 0.70   # [rad/s] 旋轉最大角速度
ANGULAR_KP   = 1.2    # 角度誤差比例增益（調低使轉向更平滑）

# --- 脫困機制 ---
STUCK_TIME        = 3.0   # [s]     卡住判定時間窗口
STUCK_THRESHOLD   = 0.04  # [m]     時間窗口內位移小於此值視為卡住
ESCAPE_DURATION   = 2.0   # [s]     旋轉逃脫持續時間（調低減少 SLAM 失真）
ESCAPE_SPEED      = 0.45  # [rad/s] 逃脫旋轉速度（關鍵：必須慢，否則 SLAM 重影）
ESCAPE_BLACKLIST_COUNT = 2  # 同一目標逃脫幾次後加入黑名單
BLACKLIST_DURATION = 60.0   # [s] 黑名單持續時間

# --- ROS 座標系 ---
MAP_FRAME   = 'map'
ROBOT_FRAME = 'base_footprint'


# =============================================================================
# 主節點
# =============================================================================

class MinimalExplorer(Node):
    """
    TurtleBot3 極簡 APF 自主探索節點。

    兩個並行迴圈：
      - _control_loop  (10Hz)：讀取感測器、計算 APF 速度、偵測卡住
      - _planning_loop  (1Hz)：偵測 Frontier、選定目標
    """

    def __init__(self):
        super().__init__('minimal_explorer')
        self.get_logger().info('=== TurtleBot3 極簡 APF 探索節點 v4 啟動 ===')

        # --- 位置（雙來源嚴格分離）-----------------------------------------
        # 規劃用（僅 TF map 座標，確保 Frontier 距離判斷正確）
        self.map_x: float   = None   # None 表示 TF 尚未就緒
        self.map_y: float   = None
        self.map_yaw: float = 0.0

        # 移動用（TF 優先 / /odom 備援）
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.pose_ready = False       # 至少一個位置來源已就緒

        # --- 感測器資料 -------------------------------------------------------
        self.current_map: OccupancyGrid = None
        self.latest_scan: LaserScan     = None

        # --- 目標 Frontier ----------------------------------------------------
        self.target_x: float = None
        self.target_y: float = None

        # --- 探索完成旗標（一旦設為 True 永久停車）---------------------------
        self.exploration_done = False

        # --- 脫困狀態機 -------------------------------------------------------
        self.stuck_check_time  = time.time()
        self.stuck_check_x     = 0.0
        self.stuck_check_y     = 0.0
        self.is_escaping       = False
        self.escape_start_time = 0.0
        self.escape_direction  = 1    # +1 逆時針 / -1 順時針

        # --- 輕量黑名單（記錄 (x, y, timeout_timestamp)）--------------------
        # 若機器人逃脫同一目標 N 次，暫時拉黑該區域
        self.escape_count: dict = {}    # key=(round_x, round_y), value=逃脫次數
        self.blacklist: list    = []    # [(x, y, expire_time), ...]

        # --- TF ---------------------------------------------------------------
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- QoS（地圖需 TRANSIENT_LOCAL 以接收歷史訊息）--------------------
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- 訂閱 -------------------------------------------------------------
        self.create_subscription(OccupancyGrid, '/map',  self._on_map,  map_qos)
        self.create_subscription(LaserScan,     '/scan', self._on_scan, 10)
        self.create_subscription(Odometry,      '/odom', self._on_odom, 10)

        # --- 發布 -------------------------------------------------------------
        self.vel_pub    = self.create_publisher(Twist,  '/cmd_vel',        10)
        self.status_pub = self.create_publisher(String, '/explorer/status', 10)

        # --- 定時器 -----------------------------------------------------------
        self.create_timer(0.1, self._control_loop)   # 10Hz：速度輸出
        self.create_timer(1.0, self._planning_loop)  # 1Hz：目標選擇

        self.get_logger().info(
            f'APF: K_ATT={APF_K_ATT} K_REP={APF_K_REP} '
            f'REP_RANGE={APF_REP_RANGE}m SAFETY={APF_SAFETY_DIST}m\n'
            f'速度: linear={MAX_LINEAR}m/s  angular={MAX_ANGULAR}rad/s  '
            f'escape={ESCAPE_SPEED}rad/s'
        )


    # =========================================================================
    # 回呼：感測器資料接收
    # =========================================================================

    def _on_map(self, msg: OccupancyGrid):
        """接收 SLAM 地圖並儲存。"""
        self.current_map = msg

    def _on_scan(self, msg: LaserScan):
        """接收 LiDAR 掃描資料並儲存。"""
        self.latest_scan = msg

    def _on_odom(self, msg: Odometry):
        """
        接收里程計資料。僅作為位置備援——當 TF 尚未就緒時，
        提供基本移動用位置以讓控制迴圈可以運作。
        ⚠ 里程計從出發點算起，與 map 座標系不同，絕不用於 Frontier 規劃。
        """
        # 優先嘗試 TF；若成功則忽略本次 odom 更新
        if self._try_update_tf():
            return

        # TF 失敗時，用 odom 更新移動用位置
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

        if not self.pose_ready:
            self.get_logger().info('[位姿] /odom 備援就緒（規劃待 TF map→base_footprint）')
            self.pose_ready = True


    # =========================================================================
    # 位姿更新：TF (map 座標系)
    # =========================================================================

    def _try_update_tf(self) -> bool:
        """
        從 TF 取得機器人在 map 座標系下的位姿。
        成功時同步更新：
          - map_x / map_y / map_yaw   → Frontier 規劃使用
          - robot_x / robot_y / robot_yaw → APF 移動使用
        失敗時不修改任何變數，返回 False（TF 初始化慢屬正常）。
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME, ROBOT_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            x   = t.x
            y   = t.y
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw  = math.atan2(siny, cosy)

            first_time = (self.map_x is None)
            self.map_x = self.robot_x = x
            self.map_y = self.robot_y = y
            self.map_yaw = self.robot_yaw = yaw

            if first_time:
                self.get_logger().info(
                    f'[位姿] TF (map→base_footprint) 就緒：({x:.2f}, {y:.2f})')
            if not self.pose_ready:
                self.pose_ready = True
            return True
        except TransformException:
            return False


    # =========================================================================
    # 黑名單工具
    # =========================================================================

    def _purge_expired_blacklist(self):
        """移除已過期的黑名單項目（每次規劃迴圈呼叫一次）。"""
        now = time.time()
        self.blacklist = [(x, y, t) for x, y, t in self.blacklist if t > now]

    def _is_blacklisted(self, wx: float, wy: float) -> bool:
        """判斷某世界座標是否在黑名單半徑 (1.0m) 內。"""
        for bx, by, _ in self.blacklist:
            if math.sqrt((wx - bx) ** 2 + (wy - by) ** 2) < 1.0:
                return True
        return False

    def _blacklist_current_target(self):
        """將當前目標加入黑名單，重置目標。"""
        if self.target_x is not None:
            self.blacklist.append(
                (self.target_x, self.target_y, time.time() + BLACKLIST_DURATION))
            self.get_logger().warn(
                f'[黑名單] ({self.target_x:.2f}, {self.target_y:.2f}) '
                f'拉黑 {BLACKLIST_DURATION:.0f}s')
        self.target_x = None
        self.target_y = None


    # =========================================================================
    # Frontier 偵測
    # =========================================================================

    def _detect_frontiers(self) -> list:
        """
        使用 OpenCV 影像處理偵測所有 Frontier（已知空地 ↔ 未知區域邊界）。

        演算法步驟：
          1. 從 OccupancyGrid 建立兩張遮罩：
               free_mask    — 自由格 (0~49)
               unknown_mask — 未知格 (-1)
          2. 膨脹 free_mask 1 格，使其與相鄰未知格「接觸」。
          3. 取 free_dilated & unknown_mask → Frontier 像素集合。
          4. 連通元件分析：計算各群集質心，過濾面積過小的雜訊。
          5. 過濾超出 FRONTIER_MAX_DIST 及黑名單內的候選。
          6. 依距離由近到遠排序後回傳。

        ⚠ 使用 self.map_x / map_y（TF map 座標）計算距離，
          絕不使用 odom 備援座標。

        回傳：
          [(world_x, world_y, distance_m), ...]
        """
        m    = self.current_map
        data = np.array(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        res  = m.info.resolution          # [m/cell]
        ox   = m.info.origin.position.x   # 地圖左下角 x 座標
        oy   = m.info.origin.position.y   # 地圖左下角 y 座標

        # 機器人在地圖格座標系中的位置（使用 TF map 座標）
        robot_mx   = int((self.map_x - ox) / res)
        robot_my   = int((self.map_y - oy) / res)
        max_cells  = FRONTIER_MAX_DIST / res  # 搜索半徑換算成格數

        # 建立遮罩
        free_mask    = ((data >= 0) & (data < 50)).astype(np.uint8) * 255
        unknown_mask = (data < 0).astype(np.uint8) * 255

        # 計算 Frontier 像素：膨脹自由空間，取與未知空間的交集
        kernel       = np.ones((3, 3), np.uint8)
        free_dilated = cv2.dilate(free_mask, kernel, iterations=1)
        frontier_px  = cv2.bitwise_and(free_dilated, unknown_mask)

        # 連通元件分析
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_px, connectivity=8)

        candidates = []
        for lid in range(1, num_labels):   # 0 為背景，略過
            if stats[lid, cv2.CC_STAT_AREA] < FRONTIER_MIN_AREA:
                continue   # 面積太小，雜訊

            # 質心格座標
            cx = int(centroids[lid, 0])
            cy = int(centroids[lid, 1])

            # 格距離過濾
            dist_cells = math.sqrt((cx - robot_mx) ** 2 + (cy - robot_my) ** 2)
            if dist_cells > max_cells:
                continue

            # 轉換為世界座標
            wx = ox + cx * res
            wy = oy + cy * res

            # 黑名單過濾
            if self._is_blacklisted(wx, wy):
                continue

            candidates.append((wx, wy, dist_cells * res))

        # 由近到遠排序（貪心策略：優先探索近處）
        candidates.sort(key=lambda p: p[2])
        return candidates


    # =========================================================================
    # APF 速度計算
    # =========================================================================

    def _apf_velocity(self) -> tuple:
        """
        人工勢場法 (APF)，計算並回傳 (linear_x, angular_z)。

        引力（Attractive Force）：
          方向為機器人→目標，固定單位大小 × APF_K_ATT。
          保持機器人向目標推進的基礎動力。

        斥力（Repulsive Force — 離散斥力點）：
          將每個 LiDAR 測量點視為獨立的斥力源。
          距離 d < APF_REP_RANGE 且在前方 ±(APF_REP_ANGLE/2)° 扇形內時：
            magnitude = K_REP × (1/d − 1/d₀) / d²
          斥力方向為「障礙物→機器人」（即遠離障礙物方向）。
          距離越近斥力越大（∝ 1/d²），足以推開機器人。

        合力→速度：
          1. 合力方向 = atan2(fy, fx) 即理想前進方向。
          2. 角度誤差 = 理想方向 − 當前朝向，轉為角速度。
          3. 線速度 × 轉向縮減因子（大角度誤差時自動減速）。
          4. 線速度 × 前方緊急制停因子（前方 ±40° 有障礙物過近時歸零）。

        回傳：
          (linear_x [m/s], angular_z [rad/s])
        """
        # ── 引力 ────────────────────────────────────────────────────────────
        dx  = self.target_x - self.robot_x
        dy  = self.target_y - self.robot_y
        d2t = math.sqrt(dx * dx + dy * dy)
        if d2t < 1e-6:
            return 0.0, 0.0  # 已在目標點

        att_x = APF_K_ATT * (dx / d2t)
        att_y = APF_K_ATT * (dy / d2t)

        # ── 斥力（離散 LiDAR 點）────────────────────────────────────────────
        rep_x, rep_y   = 0.0, 0.0
        min_front_dist = float('inf')   # 追蹤前方最近障礙物距離
        half_rep_angle = math.radians(APF_REP_ANGLE / 2)   # 斥力扇形半角
        safety_angle   = math.radians(40)                   # 緊急制停扇形半角

        if self.latest_scan is not None:
            scan = self.latest_scan
            n    = len(scan.ranges)

            for i in range(0, 360, APF_LIDAR_STEP):
                if i >= n:
                    continue

                d = scan.ranges[i]
                if not math.isfinite(d) or d <= 0.01:
                    continue

                # 機器人座標系下的角度（前方 = 0）
                angle_robot = scan.angle_min + i * scan.angle_increment

                # 追蹤前方 ±40° 內的最近障礙物（供緊急制停使用）
                if abs(angle_robot) < safety_angle:
                    min_front_dist = min(min_front_dist, d)

                # 斥力扇形過濾（只在前方 ±N° 內的障礙物產生斥力）
                if abs(angle_robot) > half_rep_angle:
                    continue
                if d > APF_REP_RANGE:
                    continue

                # 計算離散斥力點的力大小
                # 公式：K_REP × (1/d − 1/d₀) / d²
                # d 夾緊下限 0.05m 避免數值爆炸
                d_safe   = max(d, 0.05)
                magnitude = APF_K_REP * (1.0 / d_safe - 1.0 / APF_REP_RANGE) / (d_safe * d_safe)

                # 斥力方向：障礙物→機器人（即對 LiDAR 方向取反）
                angle_world = self.robot_yaw + angle_robot
                rep_x += magnitude * (-math.cos(angle_world))
                rep_y += magnitude * (-math.sin(angle_world))

        # ── 合力 → 期望朝向 → 角速度 ──────────────────────────────────────
        total_x = att_x + rep_x
        total_y = att_y + rep_y

        desired_yaw = math.atan2(total_y, total_x)
        angle_err   = math.atan2(
            math.sin(desired_yaw - self.robot_yaw),
            math.cos(desired_yaw - self.robot_yaw)
        )

        angular_z = max(-MAX_ANGULAR, min(MAX_ANGULAR, ANGULAR_KP * angle_err))

        # ── 線速度：轉向縮減 × 前方緊急制停 ──────────────────────────────
        # 轉向縮減：角度誤差 > 60° 時線速度降為 0（轉向完成才前進）
        turn_factor = max(0.0, 1.0 - abs(angle_err) / math.radians(60))

        # 緊急制停：前方障礙物進入 APF_SAFETY_DIST 範圍時線速度線性降為 0
        if min_front_dist < APF_SAFETY_DIST:
            safety_factor = max(0.0, min_front_dist / APF_SAFETY_DIST)
        else:
            safety_factor = 1.0

        linear_x = MAX_LINEAR * turn_factor * safety_factor

        return linear_x, angular_z


    # =========================================================================
    # 脫困機制
    # =========================================================================

    def _check_and_escape(self) -> bool:
        """
        偵測機器人是否卡住，若是則啟動旋轉逃脫。

        逃脫流程：
          1. 偵測 STUCK_TIME 秒內位移 < STUCK_THRESHOLD → 判定卡住
          2. 統計同一目標的逃脫次數
          3. 次數 ≥ ESCAPE_BLACKLIST_COUNT → 將該目標加入黑名單
          4. 開始以 ESCAPE_SPEED（低速）旋轉 ESCAPE_DURATION 秒
             （低速旋轉是減少 SLAM ghosting 的關鍵）

        回傳 True 代表目前正在逃脫，控制迴圈不應發其他速度指令。
        """
        now = time.time()

        # 正在執行旋轉逃脫中
        if self.is_escaping:
            if now - self.escape_start_time < ESCAPE_DURATION:
                twist = Twist()
                twist.angular.z = float(ESCAPE_SPEED * self.escape_direction)
                self.vel_pub.publish(twist)
                return True
            # 逃脫結束：重置計時器與起始位置
            self.is_escaping      = False
            self.stuck_check_time = now
            self.stuck_check_x    = self.robot_x
            self.stuck_check_y    = self.robot_y
            self.get_logger().info('[脫困] 旋轉逃脫結束，恢復正常導航')
            return False

        # 定期偵測是否卡住
        if now - self.stuck_check_time < STUCK_TIME:
            return False

        dist = math.sqrt(
            (self.robot_x - self.stuck_check_x) ** 2 +
            (self.robot_y - self.stuck_check_y) ** 2
        )

        if dist >= STUCK_THRESHOLD:
            # 位移足夠，更新起始點
            self.stuck_check_time = now
            self.stuck_check_x    = self.robot_x
            self.stuck_check_y    = self.robot_y
            return False

        # 卡住！統計次數並決定是否拉黑
        self.get_logger().warn(
            f'[脫困] 偵測到卡住！{STUCK_TIME:.0f}s 位移={dist:.3f}m')

        if self.target_x is not None:
            # 以 0.5m 精度量化目標座標作為計數 key
            key = (round(self.target_x * 2) / 2, round(self.target_y * 2) / 2)
            self.escape_count[key] = self.escape_count.get(key, 0) + 1

            if self.escape_count[key] >= ESCAPE_BLACKLIST_COUNT:
                self.get_logger().warn(
                    f'[脫困] 目標 {key} 逃脫 {self.escape_count[key]} 次，拉黑！')
                del self.escape_count[key]
                self._blacklist_current_target()
            else:
                # 尚未達到拉黑次數，清除目標讓規劃器重選
                self.target_x = None
                self.target_y = None

        # 啟動旋轉逃脫（交替左右方向避免原地打轉）
        self.is_escaping       = True
        self.escape_start_time = now
        self.escape_direction  = 1 if (int(now) % 2 == 0) else -1
        return True


    # =========================================================================
    # 規劃迴圈 (1Hz)
    # =========================================================================

    def _planning_loop(self):
        """
        每秒執行一次，負責偵測 Frontier 並選定下一個導航目標。

        流程：
          1. 清除過期黑名單。
          2. 確認 TF map 座標已就緒（否則只有移動能力，無法規劃）。
          3. 若當前目標未到達，維持現有目標。
          4. 偵測 Frontier，選擇最近的非黑名單候選。
          5. 若無 Frontier 且黑名單為空 → 宣告探索完成。
          6. 若無 Frontier 但黑名單非空 → 等待黑名單過期或地圖更新。
        """
        if self.exploration_done:
            return  # 探索已結束，不再執行

        # 1. 清理過期黑名單
        self._purge_expired_blacklist()

        # 2. 確認地圖與 TF 就緒
        if self.current_map is None:
            self.get_logger().warn('[規劃] 等待地圖 (/map)...')
            return

        self._try_update_tf()   # 嘗試更新 TF（非阻塞）

        if self.map_x is None:
            self.get_logger().warn('[規劃] 等待 TF (map→base_footprint)，暫不規劃...')
            return

        # 3. 若當前目標仍有效，維持不重選
        if self.target_x is not None:
            dist = math.sqrt(
                (self.target_x - self.map_x) ** 2 +
                (self.target_y - self.map_y) ** 2
            )
            if dist > FRONTIER_REACH_DIST:
                return  # 繼續前往

            # 到達目標
            self.get_logger().info(
                f'[規劃] ✓ 到達目標 ({self.target_x:.2f}, {self.target_y:.2f})'
                f'  dist={dist:.2f}m，重新選擇...')
            self.target_x = None
            self.target_y = None

        # 4. 偵測 Frontier
        frontiers = self._detect_frontiers()
        self.get_logger().info(
            f'[規劃] 機器人 map=({self.map_x:.2f},{self.map_y:.2f})  '
            f'找到 {len(frontiers)} 個 Frontier  黑名單={len(self.blacklist)}')

        # 5. 無 Frontier 的處理
        if not frontiers:
            if self.blacklist:
                self.get_logger().info(
                    '[規劃] 所有 Frontier 在黑名單中，等待解除...')
                return
            # 黑名單也空了，真的探索完成
            self.get_logger().info('[規劃] 🎉 找不到 Frontier，探索完成！')
            self.exploration_done = True
            self.status_pub.publish(String(data='EXPLORATION_COMPLETE'))
            self.vel_pub.publish(Twist())   # 停車
            return

        # 6. 選最近的 Frontier 作為新目標
        tx, ty, dist = frontiers[0]
        self.target_x = tx
        self.target_y = ty
        self.get_logger().info(
            f'[規劃] → 新目標: ({tx:.2f}, {ty:.2f})  距離={dist:.2f}m')


    # =========================================================================
    # 控制迴圈 (10Hz)
    # =========================================================================

    def _control_loop(self):
        """
        每 0.1 秒執行一次，輸出 /cmd_vel 速度指令。

        優先順序：
          1. 探索已完成 → 永久停車
          2. 位置來源未就緒 → 等待
          3. 脫困逃脫中 → 旋轉逃脫（低速）
          4. 無目標 → 停車等待規劃
          5. 正常 APF 導航 → 計算引力 + 離散斥力，輸出速度
        """
        # 1. 探索完成 → 永久停車
        if self.exploration_done:
            self.vel_pub.publish(Twist())
            return

        # 2. 嘗試更新 TF（非阻塞）
        self._try_update_tf()
        if not self.pose_ready:
            return   # 任何位置來源皆未就緒

        # 3. 脫困優先
        if self._check_and_escape():
            return

        # 4. 無目標 → 停車等待規劃迴圈選定目標
        if self.target_x is None:
            self.vel_pub.publish(Twist())
            return

        # 5. APF 計算速度並發布
        linear_x, angular_z = self._apf_velocity()
        twist = Twist()
        twist.linear.x  = float(linear_x)
        twist.angular.z = float(angular_z)
        self.vel_pub.publish(twist)


# =============================================================================
# 程式入口
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = MinimalExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 確保程式退出時停車
        node.vel_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
