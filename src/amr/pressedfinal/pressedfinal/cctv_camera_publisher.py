# OpenCV: 웹캠 영상 입력, 화면 출력, 바운딩 박스 그리기에 사용
import cv2

# NumPy: 영상 데이터 처리를 위한 기본 배열 라이브러리
import numpy as np

# ROS2 Python 클라이언트 라이브러리
import rclpy
from rclpy.node import Node

# 압축 이미지 메시지 타입
from sensor_msgs.msg import CompressedImage

# 사람 탐지 여부 True/False 발행용 메시지 타입
from std_msgs.msg import Bool

# OpenCV 이미지와 ROS 이미지 메시지 변환용
from cv_bridge import CvBridge

# YOLO 객체 탐지 모델
from ultralytics import YOLO


# YOLO 탐지 신뢰도 임계값
# confidence가 이 값보다 낮으면 탐지 결과를 무시한다.
CONF_THRESH = 0.7


class CctvCameraPublisher(Node):
    """
    CCTV 웹캠 기반 사람 탐지 노드.

    역할
    1. 웹캠 영상을 읽는다.
    2. YOLOv8n 모델로 사람(person)을 탐지한다.
    3. 탐지 결과가 표시된 이미지를 압축 이미지 토픽으로 발행한다.
    4. 사람 탐지 여부를 Bool 토픽으로 발행한다.
    5. OpenCV 창으로 탐지 화면을 출력한다.
    """

    def __init__(self):
        # ROS2 노드 이름 설정
        super().__init__('cctv_camera_publisher')

        # YOLOv8 nano 모델 로드
        # COCO pretrained 모델이므로 person 클래스 탐지가 가능하다.
        self.model = YOLO("yolov8n.pt")

        # 웹캠 장치 열기
        # 2번 카메라 인덱스를 사용한다.
        self.cap = cv2.VideoCapture(2)

        # 웹캠이 정상적으로 열리지 않았을 경우 예외 발생
        if not self.cap.isOpened():
            raise RuntimeError("CCTV 웹캠을 열 수 없습니다.")

        # 웹캠 프레임 가로 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        # 웹캠 프레임 세로 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 웹캠 FPS 설정
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # OpenCV 이미지와 ROS 이미지 메시지 변환을 담당하는 브릿지
        self.bridge = CvBridge()

        # 사람 탐지 결과가 그려진 압축 이미지를 발행하는 Publisher
        self.image_pub = self.create_publisher(
            CompressedImage,
            '/detection/cctv/human',
            10
        )

        # 사람 탐지 여부를 True/False로 발행하는 Publisher
        self.flag_pub = self.create_publisher(
            Bool,
            '/detection/cctv/human/flag',
            10
        )

        # 이전 탐지 상태 저장용 변수였으나 현재는 사용하지 않음
        # 상태가 바뀔 때만 publish하려는 용도로 작성했던 흔적
        # self.prev_person_detected = False

        # 0.3초마다 웹캠 프레임을 읽고 YOLO 추론을 수행한다.
        # YOLO는 0.1초보다 0.3초 정도가 더 안정적
        self.timer = self.create_timer(0.3, self.timer_callback)

        # OpenCV 시각화 창 사용 여부
        self.show_window = True

        # 화면 출력이 활성화되어 있으면 창 생성
        if self.show_window:
            cv2.namedWindow("CCTV Person Detection", cv2.WINDOW_NORMAL)

    def timer_callback(self):
        """
        타이머 주기마다 실행되는 메인 처리 함수.

        처리 흐름
        1. 웹캠에서 프레임을 읽는다.
        2. YOLO로 사람을 탐지한다.
        3. 탐지된 사람에 바운딩 박스와 라벨을 그린다.
        4. 결과 이미지를 ROS 토픽으로 발행한다.
        5. 사람 탐지 여부 Bool 값을 발행한다.
        6. 화면에 결과 영상을 표시한다.
        """

        # 웹캠에서 프레임 1장을 읽는다.
        ret, frame = self.cap.read()

        # 프레임 읽기에 실패한 경우 경고 로그 출력 후 종료
        if not ret or frame is None:
            self.get_logger().warn("CCTV 웹캠 읽기 실패")
            return

        # 화면 출력 및 publish용 프레임 복사본
        # 원본 frame은 YOLO 입력으로 사용하고,
        # display_frame에는 탐지 결과를 그린다.
        display_frame = frame.copy()

        # 현재 프레임에서 사람이 탐지되었는지 여부
        person_detected = False

        try:
            # YOLO 추론 실행
            # conf: 모델 내부 confidence 필터
            # verbose=False: 추론 로그 출력 억제
            results = self.model(
                frame,
                conf=CONF_THRESH,
                verbose=False
            )

            # YOLO 결과는 프레임 단위 result 목록으로 반환된다.
            for result in results:

                # 탐지 박스가 없으면 다음 결과로 넘어간다.
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                # 탐지된 각 박스를 순회
                for box in result.boxes:

                    # 탐지 클래스 ID
                    cls_id = int(box.cls[0])

                    # 탐지 confidence 값
                    confidence = float(box.conf[0])

                    # confidence가 임계값보다 낮으면 무시
                    if confidence < CONF_THRESH:
                        continue

                    # 클래스 ID를 클래스 이름으로 변환
                    class_name = self.model.names[cls_id]

                    # person 클래스가 아니면 무시
                    if class_name != "person":
                        continue

                    # person이 하나라도 탐지되면 True로 설정
                    person_detected = True

                    # 바운딩 박스 좌표 추출
                    # xyxy 형식: 왼쪽 위 x, 왼쪽 위 y, 오른쪽 아래 x, 오른쪽 아래 y
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 화면에 표시할 라벨 문자열
                    label = f"{class_name} {confidence:.2f}"

                    # 탐지된 사람 영역에 사각형 표시
                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    # 바운딩 박스 위에 클래스명과 confidence 표시
                    cv2.putText(
                        display_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        # YOLO 추론 과정에서 예외가 발생한 경우
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 중 오류 발생: {e}")
            return

        try:
            # OpenCV BGR 이미지를 ROS CompressedImage 메시지로 변환
            image_msg = self.bridge.cv2_to_compressed_imgmsg(
                display_frame,
                dst_format='jpg'
            )

            # 탐지 결과 이미지 publish
            self.image_pub.publish(image_msg)

        # 이미지 변환 또는 publish 중 예외 처리
        except Exception as e:
            self.get_logger().error(f"이미지 publish 중 오류 발생: {e}")

        # 탐지 상태가 바뀔 때만 publish하려던 코드였으나 현재는 비활성화
        # if person_detected != self.prev_person_detected:

        # Bool 메시지 생성
        flag_msg = Bool()

        # 현재 프레임에서 사람 탐지 여부 저장
        flag_msg.data = person_detected

        # 사람 탐지 여부 publish
        self.flag_pub.publish(flag_msg)

        # 이전 상태 저장용 코드였으나 현재는 비활성화
        # self.prev_person_detected = person_detected

        # OpenCV 창으로 결과 영상 표시
        if self.show_window:
            cv2.imshow("CCTV Person Detection", display_frame)

            # waitKey(1)을 호출해야 OpenCV 창이 정상적으로 갱신된다.
            cv2.waitKey(1)

    def destroy_node(self):
        """
        노드 종료 시 자원 정리 함수.

        웹캠 장치를 해제하고,
        OpenCV 창을 닫은 뒤,
        ROS2 노드를 정상적으로 제거한다.
        """

        # 웹캠 객체가 존재하면 장치 해제
        if self.cap is not None:
            self.cap.release()

        # 생성된 모든 OpenCV 창 닫기
        cv2.destroyAllWindows()

        # 부모 클래스의 destroy_node() 호출
        super().destroy_node()


def main(args=None):
    """
    프로그램 시작 함수.
    """

    # ROS2 초기화
    rclpy.init(args=args)

    # 예외 처리와 종료 처리를 위해 node를 먼저 None으로 초기화
    node = None

    try:
        # CCTV 사람 탐지 노드 생성
        node = CctvCameraPublisher()

        # 노드 실행
        rclpy.spin(node)

    # Ctrl+C 종료 처리
    except KeyboardInterrupt:
        pass

    # 그 외 실행 중 발생한 예외 처리
    except Exception as e:
        print(f"노드 실행 중 오류 발생: {e}")

    finally:
        # 노드가 생성된 경우 정상 종료 처리
        if node is not None:
            node.destroy_node()

        # ROS2가 아직 동작 중이면 shutdown
        if rclpy.ok():
            rclpy.shutdown()


# 이 파일을 직접 실행했을 때 main() 실행
if __name__ == '__main__':
    main()