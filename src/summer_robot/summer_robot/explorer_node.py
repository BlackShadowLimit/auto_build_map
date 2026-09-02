from typing import List, Optional, Tuple

from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException

from algorithms import FrontierDetector, MapData
from controller import Nav2Controller


class ExplorerNode(Node):
    """
    自主探索的 ROS 2 節點 (Main Node)。
    結合地圖資料 (MapData)、邊界偵測 (FrontierDetector) 與導航控制 (Nav2Controller)。
    """
    def __init__(self):
        super().__init__("explorer_node")
        self.get_logger().info("Launching Nav2 auto explore node")

        # 宣告可調參數
        self.declare_parameter("min_size", 10)
        self.declare_parameter("search_radius", 600)
        self.declare_parameter("ignore_radius", 1.0)
        self.declare_parameter("inflation_radius", 0.25)
        self.declare_parameter("decision_rate", 1.0)

        # 讀取參數
        min_size = self.get_parameter("min_size").value
        search_radius = self.get_parameter("search_radius").value
        ignore_radius = self.get_parameter("ignore_radius").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        decision_rate = self.get_parameter("decision_rate").value

        # 初始化邊界偵測器
        self.detector = FrontierDetector(
            min_size=min_size,
            search_radius=search_radius,
            ignore_radius=ignore_radius
        )
        
        self.current_map_data: Optional[MapData] = None
        
        # 紀錄已訪問及無法到達的黑名單
        self.visited_list: List[Tuple[float, float]] = []
        self.unreachable_list: List[Tuple[float, float]] = []
        self.exploration_done = False

        # 初始化 Nav2 導航控制器，並註冊導航結束的回呼函式
        self.controller = Nav2Controller(self, on_finish_callback=self._on_navigation_finished)

        # 設定 TF 監聽器，用於取得機器人當前位置
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 設定 QoS，確保能收到過去發布的地圖訊息 (TRANSIENT_LOCAL)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)
        
        # 狀態發布器
        self.status_pub = self.create_publisher(String, "/explorer/status", 10)

        # 定時執行決策迴圈
        self.create_timer(1.0 / decision_rate, self._step)

    def _on_map(self, msg: OccupancyGrid):
        """接收 /map 更新地圖資料"""
        self.current_map_data = MapData(msg)

    def _get_robot_pose(self) -> Tuple[Optional[float], Optional[float]]:
        """取得機器人在 map 座標系下的位置 (x, y)"""
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            return tf.transform.translation.x, tf.transform.translation.y
        except TransformException:
            return None, None

    def _on_navigation_finished(self, success: bool, goal: Tuple[float, float]):
        """導航結束的回呼函式：將目標點分別加入成功或失敗的黑名單"""
        if success:
            self.visited_list.append(goal)
        else:
            self.unreachable_list.append(goal)

    def _step(self):
        """
        探索決策主迴圈。
        如果正在導航則跳過；若導航閒置，則尋找下一個邊界並發送導航目標。
        """
        if self.exploration_done or self.controller.is_navigating:
            return

        # 等待地圖建置
        if self.current_map_data is None:
            self.get_logger().warn("[Explore] waiting for /map building")
            return 

        # 等待 TF 取得機器人位置
        rx, ry = self._get_robot_pose()
        if rx is None:
            self.get_logger().warn("Explore wait for TF (map -> base_footprint)...")
            return

        # 轉換為地圖格座標，並進行邊界偵測
        rmx, rmy = self.current_map_data.world_to_map(rx, ry)
        candidates = self.detector.detect(
            self.current_map_data,
            rmx,
            rmy,
            visited_list=self.visited_list,
            unreachable_list=self.unreachable_list
        )

        # 如果找不到任何有效邊界，判定探索完成
        if not candidates:
            self.get_logger().info("No Unknown area. Complete!")
            self.exploration_done = True
            self.status_pub.publish(String(data="EXPLORATION_COMPLETE"))
            return

        # 產生膨脹地圖以利安全點檢查
        inflated_map = self.current_map_data.get_inflated_map(inflation_m=self.inflation_radius)
        safe_target: Optional[Tuple[float, float]] = None

        # 從最近的候選邊界開始找尋附近的安全點
        for fx, fy in candidates:
            safe_target = FrontierDetector.find_nearest_safe_free_goal(
                self.current_map_data, fx, fy, inflated_map
            )
            # 找到安全點即跳出迴圈
            if safe_target is not None:
                break

            # 如果該邊界附近沒有安全點，將其加入無法到達的黑名單
            self.unreachable_list.append((fx, fy))

        if safe_target is None:
            self.get_logger().warn("[Explore] candidate has no safe point to navigate, waiting map to update...")
            return

        # 傳送安全目標點給 Nav2
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
