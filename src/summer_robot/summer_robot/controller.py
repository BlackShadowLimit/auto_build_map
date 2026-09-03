#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Callable, Optional, Tuple

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class Nav2Controller:
    """
    負責與 Nav2 的 navigate_to_pose Action Server 溝通的控制器。
    """
    def __init__(
        self,
        node: Node,
        on_finish_callback: Optional[Callable[[bool, Tuple[float, float]], None]] = None
    ):
        self.node = node
        self.on_finish_callback = on_finish_callback
        self.nav_client = ActionClient(node, NavigateToPose, "navigate_to_pose")

        self.is_navigating = False
        self.current_goal: Optional[Tuple[float, float]] = None

    def send_goal(self, x: float, y: float, force: bool = False) -> bool:
        if self.is_navigating and self.current_goal and not force:
            # 如果新目標距離原本的目標不到 0.5 公尺，就不重複發送，避免打斷導航跟脫困
            dist = math.hypot(x - self.current_goal[0], y - self.current_goal[1])
            if dist < 0.5:
                return False

        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().error('Nav2 Action Server (navigate_to_pose) is not online')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.w = 1.0

        self.current_goal = (x, y)
        self.is_navigating = True
        self.node.get_logger().info(f"[Explore] Navigate to: ({x:.2f}, {y:.2f})")

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn(f"[Explore] Refuse target: {self.current_goal}")
            failed_goal = self.current_goal
            self.is_navigating = False
            self.current_goal = None
            if self.on_finish_callback and failed_goal:
                self.on_finish_callback(False, failed_goal)
            return

        self.goal_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._on_navigation_finished)

    def cancel_goal(self):
        if self.is_navigating and hasattr(self, 'goal_handle') and self.goal_handle:
            self.node.get_logger().warn("[Explore] Proactively cancelling current goal due to stuck detection!")
            self.goal_handle.cancel_goal_async()
            # do not clear state here; wait for _on_navigation_finished to handle the cancellation result

    def _on_navigation_finished(self, future):
        if hasattr(self, '_get_result_future') and future != self._get_result_future:
            # 這是被新目標覆蓋（Preempted）的舊目標回呼，直接忽略，不當作失敗
            return

        status = future.result().status
        success = (status == GoalStatus.STATUS_SUCCEEDED)
        finished_goal = self.current_goal

        if success:
            self.node.get_logger().info(f"[Explore] Arrived at: {self.current_goal}")
        else:
            self.node.get_logger().warn(f"[Explore] Navigation unsuccessful (Status: {status}), Noted as unreachable")

        self.is_navigating = False
        self.current_goal = None

        if self.on_finish_callback and finished_goal:
            self.on_finish_callback(success, finished_goal)
