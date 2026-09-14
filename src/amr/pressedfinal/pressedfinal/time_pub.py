#!/usr/bin/env python3

# 시간 정보를 얻기 위해 사용하는 모듈
import time
from datetime import datetime

# Tkinter는 Python 기본 GUI 라이브러리
import tkinter as tk
from tkinter import ttk, messagebox

# ROS2 Python 클라이언트 라이브러리
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


# 반복 간격 정보를 발행할 ROS2 토픽 이름
# 예: "00:01" 형태의 문자열을 /robot3/reserve 토픽으로 발행
RESERVE_INTERVAL_TOPIC_NAME = '/robot3/reserve'


def get_current_time_text_without_sec():
    """
    현재 노트북의 시간을 문자열로 반환하는 함수

    초 단위는 표시하지 않고,
    'YYYY-MM-DD HH:MM 시간대' 형식으로 반환한다.

    예:
        2026-06-26 13:40 KST
    """

    # 현재 시간을 timestamp 기준으로 가져온 뒤,
    # 노트북의 로컬 시간대로 변환한다.
    now = datetime.fromtimestamp(time.time()).astimezone()

    # 날짜와 시간을 초 없이 문자열로 변환
    time_text = now.strftime('%Y-%m-%d %H:%M')

    # 현재 시간대 정보 추출
    timezone_text = now.strftime('%Z')

    return f'{time_text} {timezone_text}'


def get_interval_text(interval_hours, interval_minutes):
    """
    사용자가 입력한 시간, 분 값을 ROS 토픽으로 보낼 문자열로 변환하는 함수

    입력:
        interval_hours   : 반복 간격의 시간 값
        interval_minutes : 반복 간격의 분 값

    출력:
        'HH:MM' 형식의 문자열

    예:
        0시간 1분  -> '00:01'
        2시간 5분  -> '02:05'
    """

    return f'{interval_hours:02d}:{interval_minutes:02d}'


class ReserveIntervalPublisherNode(Node):
    """
    ROS2 토픽으로 반복 간격 정보를 발행하는 노드 클래스

    역할:
        - /robot3/reserve 토픽에 String 메시지를 발행한다.
        - GUI에서 버튼을 누르면 사용자가 입력한 반복 간격을 발행한다.
    """

    def __init__(self):
        """
        ROS2 노드를 초기화하고 Publisher를 생성한다.
        """

        # ROS2 노드 이름 설정
        super().__init__('reserve_interval_publisher_ui_node')

        # String 타입 메시지를 발행하는 Publisher 생성
        # 큐 크기 10은 메시지를 임시로 저장할 수 있는 버퍼 크기
        self.reserve_interval_publisher = self.create_publisher(
            String,
            RESERVE_INTERVAL_TOPIC_NAME,
            10
        )

    def publish_interval_info(self, interval_hours, interval_minutes):
        """
        반복 간격 정보를 ROS2 토픽으로 발행하는 함수

        입력받은 시간, 분 값을 'HH:MM' 형식으로 변환한 뒤
        std_msgs/String 메시지에 담아 발행한다.
        """

        # 시간, 분 값을 'HH:MM' 문자열로 변환
        interval_text = get_interval_text(
            interval_hours,
            interval_minutes
        )

        # ROS2 String 메시지 객체 생성
        msg = String()

        # 메시지 데이터에 반복 간격 문자열 저장
        msg.data = interval_text

        # /robot3/reserve 토픽으로 메시지 발행
        self.reserve_interval_publisher.publish(msg)

        # 터미널에 발행한 메시지 로그 출력
        self.get_logger().info(
            f'Published to {RESERVE_INTERVAL_TOPIC_NAME}: {msg.data}'
        )


class ReserveIntervalUI(tk.Tk):
    """
    반복 간격을 입력하고 ROS2 토픽으로 발행하는 GUI 클래스

    Tkinter의 Tk 클래스를 상속받아 GUI 창을 구성한다.

    역할:
        - 현재 노트북 시각 표시
        - 반복 간격의 시간, 분 입력
        - 버튼 클릭 시 ROS2 토픽 발행 요청
    """

    def __init__(self, ros_node):
        """
        GUI 창을 초기화하는 생성자

        ros_node:
            실제 ROS2 토픽 발행을 담당하는 ReserveIntervalPublisherNode 객체
        """

        # Tkinter 기본 창 초기화
        super().__init__()

        # GUI에서 사용할 ROS2 노드 저장
        self.ros_node = ros_node

        # 창 제목 설정
        # 빈 문자열이므로 현재는 제목이 표시되지 않음
        self.title('')

        # 창의 초기 크기 설정
        self.geometry('460x220')

        # 창 크기 조절 허용
        self.resizable(True, True)

        # GUI 위젯 생성
        self.create_widgets()

        # 0.2초 후부터 현재 시각 표시를 반복적으로 갱신
        self.after(200, self.update_ui_time)

    def create_widgets(self):
        """
        GUI 화면에 배치될 위젯들을 생성하는 함수

        구성:
            1. 현재 노트북 시각 표시 라벨
            2. 반복 간격 입력 영역
            3. 시간 입력 Spinbox
            4. 분 입력 Spinbox
            5. 토픽 발행 버튼
        """

        # 현재 노트북 시각을 표시할 라벨
        self.local_time_label = ttk.Label(
            self,
            text='현재 노트북 시각: -',
            font=('Arial', 12)
        )
        self.local_time_label.pack(pady=12)

        # 반복 간격 설정 영역을 감싸는 LabelFrame
        interval_frame = ttk.LabelFrame(
            self,
            text='반복 간격 설정'
        )
        interval_frame.pack(padx=20, pady=15, fill='x')

        # 시간, 분 입력 위젯들을 가로로 배치하기 위한 내부 Frame
        interval_inner_frame = ttk.Frame(interval_frame)
        interval_inner_frame.pack(pady=12)

        # 시간 입력값을 저장하는 Tkinter 문자열 변수
        # 기본값은 00시간
        self.interval_hour_var = tk.StringVar(value='00')

        # 분 입력값을 저장하는 Tkinter 문자열 변수
        # 기본값은 01분
        self.interval_minute_var = tk.StringVar(value='00')

        # 반복 간격의 '시간'을 입력하는 Spinbox
        # 0부터 23까지 입력 가능
        self.interval_hour_spin = ttk.Spinbox(
            interval_inner_frame,
            from_=0,
            to=23,
            width=4,
            textvariable=self.interval_hour_var,
            format='%02.0f'
        )

        # 반복 간격의 '분'을 입력하는 Spinbox
        # 0부터 59까지 입력 가능
        self.interval_minute_spin = ttk.Spinbox(
            interval_inner_frame,
            from_=0,
            to=59,
            width=4,
            textvariable=self.interval_minute_var,
            format='%02.0f'
        )

        # 시간 Spinbox 배치
        self.interval_hour_spin.pack(side='left')

        # 시간 단위 표시 라벨
        ttk.Label(
            interval_inner_frame,
            text=' 시간 '
        ).pack(side='left')

        # 분 Spinbox 배치
        self.interval_minute_spin.pack(side='left')

        # 분 단위 표시 라벨
        ttk.Label(
            interval_inner_frame,
            text=' 분'
        ).pack(side='left')

        # 버튼을 담을 Frame
        publish_interval_button_frame = ttk.Frame(self)
        publish_interval_button_frame.pack(pady=12)

        # 반복 간격 정보를 ROS2 토픽으로 발행하는 버튼
        # 버튼 클릭 시 on_publish_interval_button 함수 실행
        self.publish_interval_button = ttk.Button(
            publish_interval_button_frame,
            text='반복 간격 토픽 발행',
            command=self.on_publish_interval_button
        )
        self.publish_interval_button.pack(padx=8)

    def update_ui_time(self):
        """
        GUI에 표시되는 현재 노트북 시각을 갱신하는 함수

        self.after()를 이용해 0.2초마다 자기 자신을 다시 호출한다.
        따라서 GUI가 실행되는 동안 현재 시간이 계속 갱신된다.
        """

        # 현재 노트북 시각 문자열 생성
        now_text = get_current_time_text_without_sec()

        # 라벨의 텍스트를 현재 시각으로 변경
        self.local_time_label.config(
            text=f'현재 노트북 시각: {now_text}'
        )

        # 200ms 후 다시 update_ui_time 함수를 호출
        self.after(200, self.update_ui_time)

    def get_interval_input(self):
        """
        사용자가 입력한 반복 간격 값을 읽고 검증하는 함수

        검증 조건:
            - 시간은 0 이상 23 이하
            - 분은 0 이상 59 이하
            - 0시간 0분은 허용하지 않음

        반환:
            정상 입력이면 (interval_hours, interval_minutes)
            잘못된 입력이면 (None, None)
        """

        try:
            # Spinbox에 입력된 문자열 값을 정수로 변환
            interval_hours = int(self.interval_hour_var.get())
            interval_minutes = int(self.interval_minute_var.get())

            # 시간 범위 검사
            if not (0 <= interval_hours <= 23):
                raise ValueError

            # 분 범위 검사
            if not (0 <= interval_minutes <= 59):
                raise ValueError

            # 반복 간격이 0시간 0분이면 의미가 없으므로 오류 처리
            if interval_hours == 0 and interval_minutes == 0:
                raise ValueError

        except ValueError:
            # 입력값이 숫자가 아니거나 범위를 벗어난 경우 오류 메시지 출력
            messagebox.showerror(
                '입력 오류',
                '반복 간격은 0시간 1분 이상으로 입력해야 합니다.'
            )
            return None, None

        # 검증을 통과한 시간, 분 값을 반환
        return interval_hours, interval_minutes

    def on_publish_interval_button(self):
        """
        '반복 간격 토픽 발행' 버튼을 눌렀을 때 실행되는 콜백 함수

        동작 순서:
            1. 사용자가 입력한 반복 간격을 읽는다.
            2. 입력값이 올바른지 검사한다.
            3. 올바르면 ROS2 토픽으로 발행한다.
        """

        # GUI에서 입력한 시간, 분 값을 가져온다.
        interval_hours, interval_minutes = self.get_interval_input()

        # 입력 오류가 있으면 발행하지 않고 함수 종료
        if interval_hours is None:
            return

        # ROS2 노드를 통해 반복 간격 정보 발행
        self.ros_node.publish_interval_info(
            interval_hours=interval_hours,
            interval_minutes=interval_minutes
        )


def main(args=None):
    """
    프로그램의 시작점

    동작 순서:
        1. ROS2 초기화
        2. ROS2 Publisher 노드 생성
        3. ROS2 executor 생성
        4. ROS2 spin을 별도 스레드에서 실행
        5. Tkinter GUI 실행
        6. GUI 종료 시 ROS2 자원 정리
    """

    # ROS2 통신을 사용하기 위해 반드시 먼저 초기화해야 한다.
    rclpy.init(args=args)

    # 반복 간격을 발행하는 ROS2 노드 생성
    ros_node = ReserveIntervalPublisherNode()

    # ROS2 콜백 처리를 담당하는 executor 생성
    executor = SingleThreadedExecutor()

    # executor에 ROS2 노드 등록
    executor.add_node(ros_node)

    # GUI와 ROS2 spin을 동시에 실행하기 위해 threading 사용
    # Tkinter의 mainloop와 ROS2의 spin은 둘 다 계속 실행되는 루프이므로
    # 하나의 스레드에서 같이 실행하면 프로그램이 멈춘 것처럼 동작할 수 있다.
    import threading

    # ROS2 executor.spin()을 별도 스레드에서 실행
    # daemon=True로 설정하면 메인 프로그램 종료 시 스레드도 함께 종료된다.
    ros_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )
    ros_thread.start()

    # Tkinter GUI 객체 생성
    app = ReserveIntervalUI(ros_node)

    try:
        # GUI 이벤트 루프 실행
        # 사용자가 창을 닫기 전까지 계속 실행된다.
        app.mainloop()

    finally:
        # GUI가 종료되면 ROS2 관련 자원을 안전하게 정리한다.

        # executor 종료
        executor.shutdown()

        # ROS2 노드 제거
        ros_node.destroy_node()

        # ROS2 종료
        rclpy.shutdown()


# 이 파일을 직접 실행했을 때만 main() 함수 실행
# 다른 파일에서 import할 경우에는 자동 실행되지 않는다.
if __name__ == '__main__':
    main()