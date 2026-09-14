#!/usr/bin/env python3

import time
import subprocess
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


RESERVE_INTERVAL_TOPIC_NAME = '/robot3/reserve'
RESERVE_TIME_TOPIC_NAME = '/robot3/reserve_time'

# 실제 실행할 launch 명령어
# 패키지명이 필요하면 아래처럼 바꿔야 함:
# ['ros2', 'launch', 'pressedfinal', 'patrol.launch.py']
LAUNCH_COMMAND = ['ros2', 'launch', 'patrol.launch.py']


def get_time_text_from_timestamp(timestamp):
    return datetime.fromtimestamp(
        timestamp
    ).astimezone().strftime('%Y-%m-%d %H:%M %Z')


class ReserveTimeCalculatorNode(Node):

    def __init__(self):
        super().__init__('reserve_time_calculator_node')

        self.reserve_subscriber = self.create_subscription(
            String,
            RESERVE_INTERVAL_TOPIC_NAME,
            self.reserve_callback,
            10
        )

        self.reserve_time_publisher = self.create_publisher(
            String,
            RESERVE_TIME_TOPIC_NAME,
            10
        )

        # patrol.launch.py 실행 프로세스 저장용
        self.patrol_process = None

        self.get_logger().info(
            f'Subscribe: {RESERVE_INTERVAL_TOPIC_NAME}'
        )

        self.get_logger().info(
            f'Publish: {RESERVE_TIME_TOPIC_NAME}'
        )

    def reserve_callback(self, msg):
        reserve_text = msg.data.strip()

        try:
            interval_hours, interval_minutes = self.parse_reserve_text(
                reserve_text
            )

        except ValueError:
            self.get_logger().error(
                f'Invalid reserve data: "{reserve_text}" '
                f'expected format is HH:MM, example: 00:01'
            )
            return

        interval_sec = (interval_hours * 60 + interval_minutes) * 60

        if interval_sec <= 0:
            self.get_logger().error(
                'Reserve interval must be greater than 0 minutes'
            )
            return

        now_timestamp = time.time()
        next_publish_timestamp = now_timestamp + interval_sec

        next_publish_time_text = get_time_text_from_timestamp(
            next_publish_timestamp
        )

        publish_msg = String()
        publish_msg.data = next_publish_time_text

        self.reserve_time_publisher.publish(publish_msg)

        self.get_logger().info(
            f'Received interval: {interval_hours:02d}:{interval_minutes:02d}'
        )

        self.get_logger().info(
            f'Published next reserve time: {publish_msg.data}'
        )

        # 예약 시간이 갱신될 때마다 launch 실행 시도
        self.start_patrol_launch_if_not_running()

    def start_patrol_launch_if_not_running(self):
        if self.patrol_process is not None:
            if self.patrol_process.poll() is None:
                self.get_logger().info(
                    'patrol.launch.py is already running. Skip launch.'
                )
                return

        self.get_logger().info(
            f'Start launch: {" ".join(LAUNCH_COMMAND)}'
        )

        try:
            self.patrol_process = subprocess.Popen(
                LAUNCH_COMMAND
            )

            self.get_logger().info(
                'patrol.launch.py started.'
            )

        except Exception as e:
            self.get_logger().error(
                f'Failed to start patrol.launch.py: {e}'
            )

    def parse_reserve_text(self, reserve_text):
        split_text = reserve_text.split(':')

        if len(split_text) != 2:
            raise ValueError

        interval_hours = int(split_text[0])
        interval_minutes = int(split_text[1])

        if not (0 <= interval_hours <= 23):
            raise ValueError

        if not (0 <= interval_minutes <= 59):
            raise ValueError

        return interval_hours, interval_minutes


def main(args=None):
    rclpy.init(args=args)

    node = ReserveTimeCalculatorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()