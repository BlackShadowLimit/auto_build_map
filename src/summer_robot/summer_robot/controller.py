from typing import Callable, Optional, Tuple

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class Nav2Controller:
    """
    負責與 Nav2 的 navigate_to_pose Action Server 溝通的控制器。
    封裝了發送目標點、等待回應以及處理導航結果的非同步邏輯。
    """
    def __init__(
        self,
        node: Node,
        on_finish_callback: Optional[Callable[[bool, Tuple[float, float]], None]] = None
    ):
        self.node = node
        # 當導航結束 (成功或失敗) 時觸發的回呼函式
        self.on_finish_callback = on_finish_callback
        
        # 建立連接到 Nav2 的 Action Client
        self.nav_client = ActionClient(node, NavigateToPose, "navigate_to_pose")

        self.is_navigating = False
        self.current_goal: Optional[Tuple[float, float]] = None

    def send_goal(self, x: float, y: float) -> bool:
        """
        發送目標點 (x, y) 給 Nav2 進行導航。
        回傳 True 表示成功發送請求，回傳 False 表示 Action Server 沒上線。
        """
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().error('Nav2 Action Server (navigate_to_pose) is not online')
            return False

        # 準備 Goal 訊息
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        # 預設朝向設定為 w=1.0 (不旋轉)
        goal_msg.pose.pose.orientation.w = 1.0

        self.current_goal = (x, y)
        self.is_navigating = True
        self.node.get_logger().info(f"[Explore] Navigate to: ({x:.2f}, {y:.2f})")

        # 非同步發送 Goal
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future):
        """處理 Action Server 對於接受或拒絕 Goal 的回應"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn(f"[Explore] Refuse target: {self.current_goal}")
            failed_goal = self.current_goal
            self.is_navigating = False
            self.current_goal = None
            
            # 若被拒絕，視為導航失敗，呼叫回呼
            if self.on_finish_callback and failed_goal:
                self.on_finish_callback(False, failed_goal)
            return

        # 若接受，則繼續等待導航執行結果
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_navigation_finished)

    def _on_navigation_finished(self, future):
        """處理導航最終結果 (到達目的地或失敗)"""
        status = future.result().status
        success = (status == GoalStatus.STATUS_SUCCEEDED)
        finished_goal = self.current_goal

        if success:
            self.node.get_logger().info(f"[Explore] Arrived to: {self.current_goal}")
        else:
            self.node.get_logger().warn(f"[Explore] Navigation unsuccessful (Status: {status}), Noted as unreachable")

        self.is_navigating = False
        self.current_goal = None

        # 呼叫回呼，傳遞結果與目標座標
        if self.on_finish_callback and finished_goal:
            self.on_finish_callback(success, finished_goal)
