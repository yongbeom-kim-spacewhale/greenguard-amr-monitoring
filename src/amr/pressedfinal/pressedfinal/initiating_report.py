#!/usr/bin/env python3

import subprocess
import rclpy

from rclpy.node import Node
from std_msgs.msg import Bool


FLAG_TOPIC = '/detection/cctv/human/flag'


class LaunchReportNode(Node):

    def __init__(self):
        super().__init__('launch_report_node')

        self.launched = False

        self.create_subscription(
            Bool,
            FLAG_TOPIC,
            self.flag_callback,
            10
        )

        self.get_logger().info('Waiting for /detection/cctv/human/flag...')

    def flag_callback(self, msg):
        if self.launched:
            return

        if msg.data:
            self.launched = True

            self.get_logger().info(
                'Human detected. Launching report.launch.py...'
            )

            subprocess.Popen([
                'ros2',
                'launch',
                'pressedfinal',
                'report.launch.py'
            ])

            self.get_logger().info('Launch complete. Exiting.')

            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = LaunchReportNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()