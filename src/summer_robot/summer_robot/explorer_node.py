from typing import Optional, Tuple

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
    def __init__(self):
        super().__init__("explorer_node")
        self.get_logger().info("Launching Nav2 auto explor node")

        self.declare_parameter("min_size", 10)
        self.declare_parameter("search_radius", 600)
        self.declare_parameter("ignore_radius", 1.0)
        self.declare_parameter("inflation_radius", 0.25)
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
        self.visited_list: list[Tuple[float, float]] = []
        self.unreachable_list: list[Tuple[float, float]] = []
        self.exploration_done = False

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
            self.get_logger().warn("[Explore] wait for /map building")
            return 

        rx, ry = self._get_robot_pose()
        if rx is None:
            self.get_logger().warn("Explore wait for TF (map -> basefootpring)...")
            return

        rmx, rmy = self.current_map_data.world_to_map(rx, ry)
        candidates = self.detector.detect(
            self.current_map_data,
            rmx,
            rmy,
            visited_list=self.visited_list,
            unreachable_list=self.unreachable_list
        )

        if not candidates:
            self.get_logger().info("No Unknown area. Complete!")
            self.exploration_done = True
            self.status_pub.publish(String(data="EXPLORATION_COMPLETE"))
            return

        inflated_map = self.current_map_data.get_inflated_map(inflation_m=self.inflation_radius)
        safe_target: Optional[Tuple[float, float]] = None

        for fx, fy in candidates:
            safe_target = FrontierDetector.find_nearest_safe_free_goal(
                self.current_map_data, fx, fy, inflated_map
            )
            if safe_target is not None:
                break

            self.unreachable_list.append((fx, fy))

        if safe_target is None:
            self.get_logger().warn("[Explorer] candidate has no safe point to navigate, waiting map to update...")
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
