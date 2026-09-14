from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pressedfinal',
            executable='cctv_camera_publisher',
            name='cctv_camera_publisher',
            output='screen'
        ),

        Node(
            package='pressedfinal',
            executable='initiating_report',
            name='initiating_report',
            output='screen'
        ),
    ])