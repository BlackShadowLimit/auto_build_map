from collections import deque
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import rclpy
import numpy as np

class GreedyExplorer(Node):
    def __init__(self):
        super().__init__(
            'greedy_explorer',
            automatically_declare_parameters_from_overrides=True,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.is_navigating = False

        ##### Camera 預留位置
        self.latest_camera_frame = None
        #####

        self.get_logger().info('Greedy Explorer launching...')


    ##### Camera 預留位置
    def camera_callback(self, msg):
        self.latest_camera_frame = msg

    def process_vision(self, target_wx, target_wy, grid, msg_info):
        """
        出發之前先用影像辨識檢查前方的障礙物在哪
        """ 
        if self.latest_camera_frame is None:
            return True, target_wx, target_wy
        return True, target_wx, target_wy
    #####

    def parse_map(self, msg: OccupancyGrid):
        if len(msg.data) == 0:
            return None, None, None

        width = msg.info.width
        height = msg.info.height
        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))

        return width, height, grid
    def world_to_grid(self, wx, wy, info):
        gx = int((wx - info.origin.position.x) / info.resolution)
        gy = int((wy - info.origin.position.y) / info.resolution)
        return gx, gy
        
    def grid_to_world(self, gx, gy, info):
        wx = info.origin.position.x + (gx + 0.5) * info.resolution
        wy = info.origin.position.y + (gy + 0.5) * info.resolution
        return wx, wy

    def get_robot_grid_pose(self, msg: OccupancyGrid):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )

            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
            grid_x, grid_y = self.world_to_grid(robot_x, robot_y, msg.info)

            return grid_x, grid_y
        except Exception as e:
            self.get_logger().error(f"TF error: {e}")
            return None, None

    def find_frontier(self, grid, width, height, start_x, start_y):
        frontier_candidates = deque([(start_x, start_y)])
        visited_cells = set([(start_x, start_y)])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while frontier_candidates:
            cx, cy = frontier_candidates.popleft()

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx] == -1:
                        return (cx, cy)
                    elif grid[ny, nx] == 0 and (nx, ny) not in visited_cells:
                        visited_cells.add((nx, ny))
                        frontier_candidates.append((nx, ny))

        return None

    def send_nav_goal(self, wx, wy):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Nav2 Action Server is NOT ready")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = wx
        goal_msg.pose.pose.position.y = wy
        goal_msg.pose.pose.orientation.w = 1.0

        self.is_navigating = True
        self.get_logger().info(f"Navigating to {wx:.2f}, {wy:.2f}")

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 refuse navigation")
            self.is_navigating = False
            return

        self.get_logger().info("Nav2 accept the target, moving...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
      self.get_logger().info("Arrived")
      self.is_navigating = False

    def map_callback(self, msg: OccupancyGrid):
        if self.is_navigating:
            return

        width, height, grid = self.parse_map(msg)
        if grid is None:
            return

        start_x, start_y = self.get_robot_grid_pose(msg)
        if start_x is None or start_y is None:
            self.get_logger().warn("Waiting transforming map")
            return
        if not (0 <= start_x < width and 0 <= start_y < height):
            return

        target = self.find_frontier(grid, width, height, start_x, start_y)

        if target:
            self.get_logger().info(f"Finding point. The nearest unknown point at: {target}")
            
            raw_wx, raw_wy = self.grid_to_world(target[0], target[1], msg.info)

            ##### 影像辨識預位置
            allowed, final_wx, final_wy = self.process_vision(
                raw_wx, raw_wy, grid, msg.info
            )
            #####

            if allowed:
                self.send_nav_goal(final_wx, final_wy)
        else:
            self.get_logger().info("There's no unknown area in the map")

def main(args=None):
    rclpy.init(args=args)
    node = GreedyExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
