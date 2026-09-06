import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_summer_robot = get_package_share_directory('summer_robot')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    nav2_params_file = os.path.join(pkg_summer_robot, 'config', 'nav2_params.yaml')
    explorer_params_file = os.path.join(pkg_summer_robot, 'config', 'explorer_params.yaml')

    # 1. 靜態發布相機座標 (車上的 Bringup 只發布了 base_link，這裡補上 camera_link)
    # 這裡的 pitch 0.52 必須對應魚眼視覺節點的 pitch
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0.08', '0.0', '0.10', '0', '0.52', '0', 'base_link', 'camera_link']
    )

    # 2. 啟動 Cartographer 建圖
    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 3. 延遲 3 秒：啟動魚眼視覺避障雷達節點 (接收來自車上發布的影像)
    ground_scanner_launch = TimerAction(
        period=3.0,
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

    # 4. 延遲 10 秒：啟動 Nav2 導航
    nav2_launch = TimerAction(
        period=10.0,
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

    # 5. 延遲 18 秒：啟動自主探索節點
    explorer_node = TimerAction(
        period=18.0,
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
        camera_tf_publisher,
        cartographer_launch,
        ground_scanner_launch,
        nav2_launch,
        explorer_node
    ])
