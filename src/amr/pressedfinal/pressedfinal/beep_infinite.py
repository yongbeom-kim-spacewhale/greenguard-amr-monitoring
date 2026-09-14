#!/usr/bin/env python3

# ROS2 Python 클라이언트 라이브러리
import rclpy
from rclpy.node import Node

# 경고 여부(True/False)를 수신하기 위한 메시지
from std_msgs.msg import Bool

# TurtleBot4(Create3)의 스피커로 음계를 재생하기 위한 메시지
from irobot_create_msgs.msg import AudioNote, AudioNoteVector

# 각 음표의 재생 시간을 지정하기 위한 Duration 메시지
from builtin_interfaces.msg import Duration


# =========================
# Topic 설정
# =========================

# 경고 상태(True/False)를 구독하는 토픽
WARNING_TOPIC = '/robot3/warning'

# 로봇 스피커에 음계를 전송하는 토픽
AUDIO_TOPIC = '/robot3/cmd_audio'

# 마지막으로 True를 받은 후 이 시간(초) 이상 추가 True가 들어오지 않으면
# 경고를 자동으로 OFF 처리한다.
WARNING_TIMEOUT = 1.0


class BeepNode(Node):
    """
    경고음 재생 노드.

    역할
    1. /robot3/warning 토픽을 구독한다.
    2. True가 들어오면 일정 주기로 경고음을 계속 재생한다.
    3. False가 들어오면 즉시 경고음을 중지한다.
    4. 일정 시간(True 미수신) 동안 업데이트가 없으면
       자동으로 Warning을 OFF 처리하여 안전하게 종료한다.
    """

    def __init__(self):
        # ROS2 노드 이름 설정
        super().__init__('beep_node')

        # AudioNoteVector 메시지를 발행하는 Publisher 생성
        self.pub = self.create_publisher(
            AudioNoteVector,
            AUDIO_TOPIC,
            10
        )

        # 경고 토픽 구독
        # Bool(True/False)을 받아 warning_callback()을 실행한다.
        self.create_subscription(
            Bool,
            WARNING_TOPIC,
            self.warning_callback,
            10
        )

        # 현재 경고 상태
        # True이면 삐뽀음을 재생해야 하는 상태
        self.warning = False

        # 마지막으로 True를 수신한 ROS 시간
        # 아직 True를 받은 적이 없으면 None
        self.last_true_time = None

        # 0.5초마다 상태를 확인하는 Timer 생성
        self.timer = self.create_timer(
            0.5,
            self.timer_callback
        )

    def warning_callback(self, msg):
        """
        /robot3/warning 토픽을 수신했을 때 호출된다.
        """

        # 현재 ROS 시간을 얻는다.
        now = self.get_clock().now()

        # -------------------------
        # True를 수신한 경우
        # -------------------------
        if msg.data is True:

            # 마지막 True 수신 시간을 갱신한다.
            self.last_true_time = now

            # 기존에 OFF였다면 ON으로 변경
            if not self.warning:
                self.warning = True
                self.get_logger().info('Warning ON')

        # -------------------------
        # False를 수신한 경우
        # -------------------------
        else:

            # 현재 Warning 상태였다면 OFF 처리
            if self.warning:
                self.warning = False
                self.get_logger().info('Warning OFF')

                # 현재 재생 중인 음계를 중단
                self.stop_beep()

            # 마지막 True 시간도 초기화
            self.last_true_time = None

    def timer_callback(self):
        """
        0.5초마다 호출되는 타이머.

        최근 True 수신 여부를 확인하여
        계속 삐뽀음을 낼지,
        자동 OFF 처리할지를 결정한다.
        """

        # 현재 Audio Topic을 구독하는 장치가 없으면
        # 불필요한 메시지 발행을 하지 않는다.
        if self.pub.get_subscription_count() == 0:
            return

        # 아직 True를 받은 적이 없으면 아무 작업도 하지 않는다.
        if self.last_true_time is None:
            return

        # 현재 시간
        now = self.get_clock().now()

        # 마지막 True 이후 경과 시간(초)
        elapsed = (now - self.last_true_time).nanoseconds / 1e9

        # 일정 시간 이상 True가 다시 들어오지 않으면
        # 자동으로 Warning OFF 처리
        if elapsed > WARNING_TIMEOUT:

            if self.warning:
                self.warning = False
                self.get_logger().info('Warning timeout -> OFF')

                # 현재 재생 중인 음계 정지
                self.stop_beep()

            return

        # 최근에도 True가 들어오고 있는 상태라면
        # 계속 경고음을 재생한다.
        if self.warning:
            self.publish_beep()

    def publish_beep(self):
        """
        삐뽀삐뽀 경고음을 AudioNoteVector 형태로 발행한다.
        """

        # 새로운 음계 메시지 생성
        msg = AudioNoteVector()

        # 기존 음계를 덮어쓰도록 설정
        # True이면 기존 음계 뒤에 추가된다.
        msg.append = False

        # (주파수, 초, 나노초)
        # 삐(784Hz) - 뽀(523Hz)를 반복하는 음계
        notes_data = [
            (784, 0, 100000000),  # 삐
            (523, 0, 100000000),  # 뽀
            (784, 0, 100000000),  # 삐
            (523, 0, 100000000),  # 뽀
            (784, 0, 100000000),  # 삐
            (523, 0, 100000000),  # 뽀
            (784, 0, 100000000),  # 삐
            (523, 0, 100000000),  # 뽀
            (784, 0, 100000000),  # 삐
            (523, 0, 100000000),  # 뽀
        ]

        # 음표 데이터를 AudioNote 메시지로 변환
        for freq, sec, nanosec in notes_data:

            note = AudioNote()

            # 음 높이(Hz)
            note.frequency = freq

            # 해당 음표의 재생 시간
            note.max_runtime = Duration(
                sec=sec,
                nanosec=nanosec
            )

            # AudioNoteVector에 추가
            msg.notes.append(note)

        # 완성된 음계를 발행
        self.pub.publish(msg)

    def stop_beep(self):
        """
        빈 AudioNoteVector를 발행하여
        현재 재생 중인 음계를 중지한다.
        """

        msg = AudioNoteVector()

        # 기존 음계를 모두 덮어쓰도록 설정
        msg.append = False

        # 빈 음계를 발행하면 Create3 스피커 재생이 종료된다.
        self.pub.publish(msg)


def main(args=None):
    """
    프로그램 시작 함수.
    """

    # ROS2 초기화
    rclpy.init(args=args)

    # 노드 생성
    node = BeepNode()

    try:
        # 노드 실행
        rclpy.spin(node)

    finally:
        # 종료 시 노드 제거
        node.destroy_node()

        # ROS2 종료
        rclpy.shutdown()


# 직접 실행될 경우 main() 실행
if __name__ == '__main__':
    main()