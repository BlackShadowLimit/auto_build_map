import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 強制寫入實體車環境變數
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    os.environ['LDS_MODEL'] = 'lds-02'  # 確認你的光達型號是 lds-02

    pkg_summer_robot = get_package_share_directory('summer_robot')
    tb3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    turtlebot3_bringup_dir = get_package_share_directory('turtlebot3_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    nav2_params_file = os.path.join(pkg_summer_robot, 'config', 'nav2_params.yaml')
    explorer_params_file = os.path.join(pkg_summer_robot, 'config', 'explorer_params.yaml')

    # 2. 啟動實體車底盤與光達驅動 (負責發布 odom 與 scan)
    turtlebot3_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(turtlebot3_bringup_dir, 'launch', 'robot.launch.py'))
    )

    # 3. 啟動實體相機硬體驅動 (負責抓取 USB 相機畫面並發布 /camera/image_raw)
    camera_driver_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera_node',
        parameters=[{
            'image_size': [640, 480],
            'camera_frame_id': 'camera_link'
        }],
        remappings=[
            ('/image_raw', '/camera/image_raw')  # 將預設話題重新對應到你的掃描節點
        ]
    )

    # 4. 靜態發布相機座標
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0.08', '0.0', '0.10', '0', '0.52', '0', 'base_link', 'camera_link'] # pitch 對應 0.52
    )

    # 5. 延遲 4 秒：啟動 Cartographer 建圖
    cartographer_launch = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(tb3_cartographer_dir, 'launch', 'cartographer.launch.py')),
                launch_arguments={'use_sim_time': use_sim_time}.items()
            )
        ]
    )

    # 6. 延遲 8 秒：啟動魚眼視覺避障雷達節點
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

    # 7. 延遲 14 秒：啟動 Nav2 導航
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

    # 8. 延遲 22 秒：啟動自主探索節點
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
        camera_driver_node,
        camera_tf_publisher,
        cartographer_launch,
        ground_scanner_launch,
        nav2_launch,
        explorer_node
    ])
