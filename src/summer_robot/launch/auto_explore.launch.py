import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_summer_robot = get_package_share_directory('summer_robot')
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 出生點設定：直接放在房子正中央大廳開闊處
    spawn_x = LaunchConfiguration('x_pose', default='-1.0')
    spawn_y = LaunchConfiguration('y_pose', default='1.0')

    nav2_params_file = os.path.join(pkg_summer_robot, 'config', 'nav2_params.yaml')
    explorer_params_file = os.path.join(pkg_summer_robot, 'config', 'explorer_params.yaml')

    # 1. 啟動 Gazebo House 並指定中央出生座標
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_gazebo_dir, 'launch', 'turtlebot3_house.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'x_pose': spawn_x,
            'y_pose': spawn_y
        }.items()
    )

    # 2. 延遲 4 秒啟動 Cartographer 建圖（等 Gazebo TF 與雷達出光）
    cartographer_launch = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )

    # 3. 延遲 9 秒啟動 Nav2（等 Cartographer 發布 /map 與 TF）
    nav2_launch = TimerAction(
        period=9.0,
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

    # 4. 延遲 14 秒啟動自訂探索節點
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

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='-1.0'),
        DeclareLaunchArgument('y_pose', default_value='1.0'),
        gazebo_launch,
        cartographer_launch,
        nav2_launch,
        explorer_node
    ])
