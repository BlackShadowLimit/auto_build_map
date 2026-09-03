#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Tuple

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException

from summer_robot.algorithms import FrontierDetector, MapData
from summer_robot.controller import Nav2Controller


class ExplorerNode(Node):
    def __init__(self):
        super().__init__("explorer_node")
        self.get_logger().info("Launching Nav2 auto explore node")

        # 宣告參數與預設值
        self.declare_parameter("min_size", 3)          # 配合新版演算法微調預設值
        self.declare_parameter("search_radius", 800)
        self.declare_parameter("ignore_radius", 1.5)   # 放大死巷黑名單半徑
        self.declare_parameter("inflation_radius", 0.22)
        self.declare_parameter("decision_rate", 1.0)

        min_size = self.get_parameter("min_size").value
        search_radius = self.get_parameter("search_radius").value
        ignore_radius = self.get_parameter("ignore_radius").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        decision_rate = self.get_parameter("decision_rate").value

        self.detector = FrontierDetector(
            min_size=min_size,
            search_radius=search_radius,
            ignore_radius=ignore_radius
        )

        self.current_map_data: Optional[MapData] = None
        self.visited_list: List[Tuple[float, float]] = []
        self.unreachable_list: List[Tuple[float, float]] = []
        self.current_target: Optional[Tuple[float, float]] = None  # 新增：追蹤當前目標
        self.exploration_done = False
        
        # 新增：連續失敗計數與上限
        self.consecutive_failures = 0
        self.max_failures_to_stop = 3

        # 給予 10 次容錯緩衝 (約 10 秒)，避免開機地圖未成形直接退出
        self.no_frontier_count = 0
        self.max_no_frontier_retries = 10

        self.controller = Nav2Controller(self, on_finish_callback=self._on_navigation_finished)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)
        self.status_pub = self.create_publisher(String, "/explorer/status", 10)

        self.create_timer(1.0 / decision_rate, self._step)

    def _on_map(self, msg: OccupancyGrid):
        self.current_map_data = MapData(msg)

    def _get_robot_pose(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        # 修改：同步回傳車體朝向 (Yaw)，讓演算法產生轉向懲罰
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            # Quaternion 轉 Euler Yaw
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return x, y, yaw
        except TransformException:
            return None, None, None

    def _on_navigation_finished(self, success: bool, goal: Tuple[float, float]):
        self.current_target = None  # 導航結束，清除目標鎖定
        
        if success:
            self.visited_list.append(goal)
            self.consecutive_failures = 0  # 成功到達，計數器歸零
        else:
            self.unreachable_list.append(goal)
            self.consecutive_failures += 1 # 失敗，累加計數器
            self.get_logger().warn(f"[Explore] Nav failed ({self.consecutive_failures}/{self.max_failures_to_stop}), blacklisting target: {goal}")
            
            # 新增：限制黑名單長度，避免無效點堆積吃效能或永久鎖死
            if len(self.unreachable_list) > 30:
                self.unreachable_list.pop(0)

    def _step(self):
        if self.exploration_done or self.controller.is_navigating:
            return

        # 1. 達標終止判斷：如果連續失敗達上限，代表跳躍無效，結束掃描
        if self.consecutive_failures >= self.max_failures_to_stop:
            self.get_logger().info(f"Nav failed {self.max_failures_to_stop} times consecutively. Force stopping exploration!")
            self.exploration_done = True
            self.status_pub.publish(String(data="EXPLORATION_COMPLETE"))
            return

        if self.current_map_data is None:
            self.get_logger().warn("[Explore] waiting for /map building")
            return 

        rx, ry, ryaw = self._get_robot_pose()
        if rx is None:
            self.get_logger().warn("[Explore] wait for TF (map -> base_footprint)...")
            return

        rmx, rmy = self.current_map_data.world_to_map(rx, ry)

        # 印出目前地圖概況
        grid = self.current_map_data.data
        free_cells = np.sum((grid >= 0) & (grid < 50))
        unknown_cells = np.sum(grid < 0)
        self.get_logger().info(f"[Map] Size: {self.current_map_data.width}x{self.current_map_data.height} | Free: {free_cells} | Unknown: {unknown_cells}")

        # 2. 動態跳躍：根據失敗次數，暫時放大 detector 的黑名單半徑
        original_ignore_radius = self.detector.ignore_radius
        if self.consecutive_failures == 1:
            self.detector.ignore_radius = 1.5  # 第一次失敗，嘗試中距離跳躍
        elif self.consecutive_failures >= 2:
            self.detector.ignore_radius = 3.0  # 第二次以上失敗，嘗試遠距離跳躍

        # 修改：傳入 robot_yaw 與 current_target
        candidates = self.detector.detect(
            self.current_map_data,
            rmx,
            rmy,
            robot_yaw=ryaw,
            current_target=self.current_target,
            visited_list=self.visited_list,
            unreachable_list=self.unreachable_list
        )
        
        # 恢復原始設定
        self.detector.ignore_radius = original_ignore_radius

        if not candidates:
            self.no_frontier_count += 1
            self.get_logger().info(f"[Explore] No frontier detected ({self.no_frontier_count}/{self.max_no_frontier_retries}), waiting...")
            if self.no_frontier_count >= self.max_no_frontier_retries:
                self.get_logger().info("No Unknown area confirmed. Exploration Complete!")
                self.exploration_done = True
                self.status_pub.publish(String(data="EXPLORATION_COMPLETE"))
            return

        # 只要抓到邊界就重設計數器
        self.no_frontier_count = 0

        inflated_map = self.current_map_data.get_inflated_map(inflation_m=self.inflation_radius)
        safe_target: Optional[Tuple[float, float]] = None

        for fx, fy in candidates:
            safe_target = FrontierDetector.find_nearest_safe_free_goal(
                self.current_map_data, fx, fy, rx, ry, inflated_map
            )
            if safe_target is not None:
                break

            self.unreachable_list.append((fx, fy))

        if safe_target is None:
            self.get_logger().warn("[Explore] Candidate has no safe point to navigate, waiting map to update...")
            return

        # 紀錄鎖定目標並發送給 Nav2
        self.current_target = safe_target
        self.controller.send_goal(safe_target[0], safe_target[1])


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
