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
    
    # 實體車專用 RViz 設定檔（直接調用官方 Cartographer 與 Nav2 的複合檢視）
    rviz_config_file = os.path.join(tb3_cartographer_dir, 'rviz', 'tb3_cartographer.rviz')

    # 1. 啟動 Cartographer 建圖 (實體車接收真實 /scan 與 /odom)
    cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 2. 地面視覺避障雷達節點 (處理實體相機影像並發布 /camera_scan)
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

    # 3. 延遲 8 秒啟動 Nav2 導航
    nav2_launch = TimerAction(
        period=8.0,
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

    # 4. 延遲 14 秒啟動自主探索節點
    explorer_node = TimerAction(
        period=14.0,
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

    # 5. 即時監測視窗：啟動 RViz2 觀看即時建圖與導航狀態
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        cartographer_launch,
        ground_scanner_launch,
        nav2_launch,
        explorer_node,
        rviz_node  # <--- 確保隨時能看到即時建圖結果
    ])
