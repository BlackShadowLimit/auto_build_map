import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_summer_robot = get_package_share_directory('summer_robot')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    spawn_x = LaunchConfiguration('x_pose', default='-1.0')
    spawn_y = LaunchConfiguration('y_pose', default='1.0')

    nav2_params_file = os.path.join(pkg_summer_robot, 'config', 'nav2_params.yaml')
    explorer_params_file = os.path.join(pkg_summer_robot, 'config', 'explorer_params.yaml')
    
    # 關鍵：指定讀取自訂的帶相機 URDF
    custom_urdf_file = os.path.join(pkg_summer_robot, 'urdf', 'turtlebot3_burger.urdf')
    robot_description = Command(['xacro ', custom_urdf_file])

    # 1. 啟動 Gazebo 伺服器與客戶端載入 House 地圖
    world_file = os.path.join(tb3_gazebo_dir, 'worlds', 'turtlebot3_house.world')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world_file}.items()
    )

    # 2. 發布帶相機的 robot_description 與 TF 樹
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_description}]
    )

    # 3. 將帶相機的 Burger 模型生成到 Gazebo 中
    spawn_burger = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3_burger',
            '-topic', 'robot_description',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', '0.01'
        ],
        output='screen'
    )

    # 4. 延遲 4 秒啟動 Cartographer 建圖
    cartographer_launch = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )

    # 5. 【新增】：延遲 6 秒啟動地面視覺避障雷達節點（待相機出圖後進行地面校準）
    ground_scanner_launch = TimerAction(
        period=6.0,
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

    # 6. 延遲 15 秒啟動 Nav2
    nav2_launch = TimerAction(
        period=15.0,
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

    # 7. 延遲 22 秒啟動探索節點
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
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='-1.0'),
        DeclareLaunchArgument('y_pose', default_value='1.0'),
        gazebo_launch,
        robot_state_publisher,
        spawn_burger,
        cartographer_launch,
        ground_scanner_launch,  # <--- 已加入
        nav2_launch,
        explorer_node
    ])
