import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 強制寫入實體車環境變數 (免去手動 export)
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    os.environ['LDS_MODEL'] = 'lds-01'  # 若光達是新款，請改為 'lds-02'

    pkg_summer_robot = get_package_share_directory('summer_robot')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    turtlebot3_bringup_dir = get_package_share_directory('turtlebot3_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false') # 實體車必須為 false

    nav2_params_file = os.path.join(pkg_summer_robot, 'config', 'nav2_params.yaml')
    explorer_params_file = os.path.join(pkg_summer_robot, 'config', 'explorer_params.yaml')

    # 2. 啟動實體車底盤與光達驅動 (負責發布 odom 與 scan)
    turtlebot3_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(turtlebot3_bringup_dir, 'launch', 'robot.launch.py'))
    )

    # 3. 靜態發布相機座標 (取代自訂 URDF，避免跟原廠 TF 樹打架)
    # 參數對應：x y z yaw pitch roll frame_id child_frame_id
    # 這裡的 pitch 0.95 (約 54 度) 請與 ground_scanner_node.py 裡的設定保持一致
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0.08', '0.0', '0.12', '0', '0.95', '0', 'base_link', 'camera_link']
    )

    # 4. 延遲 4 秒：啟動 Cartographer 建圖 (等待 odom 與 TF 樹就緒)
    cartographer_launch = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )

    # 5. 延遲 8 秒：啟動地面視覺避障節點
    ground_scanner_launch = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='summer_robot',
                executable='ground_scanner_node',
                name='ground_scanner_node',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )

    # 6. 延遲 14 秒：啟動 Nav2 導航 (確保地圖與 costmap 已經開始發布)
    nav2_launch = TimerAction(
        period=14.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'params_file': nav2_params_file
                }.items()
            )
        ]
    )

    # 7. 延遲 22 秒：啟動自主探索節點 (確保 Nav2 的 Action Server 完全上線)
    explorer_node = TimerAction(
        period=22.0,
        actions=[
            Node(
                package='summer_robot',
                executable='explorer_node',
                name='explorer_node',
                output='screen',
                parameters=[explorer_params_file, {'use_sim_time': use_sim_time}]
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        turtlebot3_bringup,
        camera_tf_publisher,
        cartographer_launch,
        ground_scanner_launch,
        nav2_launch,
        explorer_node
    ])
