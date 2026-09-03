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
    turtlebot3_bringup_dir = get_package_share_directory('turtlebot3_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    nav2_params_file = os.path.join(pkg_summer_robot, 'config', 'nav2_params.yaml')
    explorer_params_file = os.path.join(pkg_summer_robot, 'config', 'explorer_params.yaml')

    # 1. 啟動實體車底盤與光達驅動 (確保 /scan 和 /odom 正常發布)
    turtlebot3_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(turtlebot3_bringup_dir, 'launch', 'robot.launch.py'))
    )

    # 2. 啟動 Cartographer 建圖 (官方內建會開一個 RViz，不要另外手動加 rviz_node)
    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 3. 地面視覺避障雷達節點 (延遲啟動，給相機驅動一點準備時間)
    ground_scanner_launch = TimerAction(
        period=5.0,
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

    # 4. 延遲 10 秒啟動 Nav2 導航
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

    # 5. 延遲 16 秒啟動自主探索節點
    explorer_node = TimerAction(
        period=16.0,
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
        cartographer_launch,
        ground_scanner_launch,
        nav2_launch,
        explorer_node
    ])
