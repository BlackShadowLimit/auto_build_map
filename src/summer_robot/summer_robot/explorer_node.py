#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        self.declare_parameter("min_size", 4)
        self.declare_parameter("search_radius", 800)
        self.declare_parameter("ignore_radius", 0.5)
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
        self.exploration_done = False

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

    def _get_robot_pose(self) -> Tuple[Optional[float], Optional[float]]:
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            return tf.transform.translation.x, tf.transform.translation.y
        except TransformException:
            return None, None

    def _on_navigation_finished(self, success: bool, goal: Tuple[float, float]):
        if success:
            self.visited_list.append(goal)
        else:
            self.unreachable_list.append(goal)

    def _step(self):
        if self.exploration_done or self.controller.is_navigating:
            return

        if self.current_map_data is None:
            self.get_logger().warn("[Explore] waiting for /map building")
            return 

        rx, ry = self._get_robot_pose()
        if rx is None:
            self.get_logger().warn("[Explore] wait for TF (map -> base_footprint)...")
            return

        rmx, rmy = self.current_map_data.world_to_map(rx, ry)

        # 印出目前地圖概況
        grid = self.current_map_data.data
        free_cells = np.sum((grid >= 0) & (grid < 50))
        unknown_cells = np.sum(grid < 0)
        self.get_logger().info(f"[Map] Size: {self.current_map_data.width}x{self.current_map_data.height} | Free: {free_cells} | Unknown: {unknown_cells}")

        candidates = self.detector.detect(
            self.current_map_data,
            rmx,
            rmy,
            visited_list=self.visited_list,
            unreachable_list=self.unreachable_list
        )

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
