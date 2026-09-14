from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pressedfinal',
            executable='beep_infinite',
            name='beep_infinite',
            output='screen'
        ),

        Node(
            package='pressedfinal',
            executable='yolo_detection',
            name='yolo_detection',
            output='screen'
        ),

        Node(
            package='pressedfinal',
            executable='report',
            name='report',
            output='screen'
        ),
    ])