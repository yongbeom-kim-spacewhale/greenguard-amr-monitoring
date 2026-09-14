import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO


CONF_THRESH = 0.8

class RobotYoloPublisher(Node):

    def __init__(self):
        super().__init__('robot_yolo_publisher')

        # ================= YOLO 모델 로드 =================
        # COCO pretrained 모델 사용 → person 클래스 탐지
        self.robot_model = YOLO("yolov8n.pt")

        # OpenCV ↔ ROS CompressedImage 변환
        self.bridge = CvBridge()

        # ================= Publisher 생성 =================

        # robot3 사람 탐지 결과 이미지 publish
        self.robot3_result_image_pub = self.create_publisher(
            CompressedImage,
            '/robot3/oakd/rgb/image_raw/compressed/person',
            10
        )

        # robot8 사람 탐지 결과 이미지 publish
        self.robot8_result_image_pub = self.create_publisher(
            CompressedImage,
            '/robot8/oakd/rgb/image_raw/compressed/person',
            10
        )

        # robot3 사람 감지 flag publish
        self.flag_robot3_pub = self.create_publisher(
            Bool,
            '/detection/robot3/flag',
            10
        )

        # robot8 사람 감지 flag publish
        self.flag_robot8_pub = self.create_publisher(
            Bool,
            '/detection/robot8/flag',
            10
        )

        # 초기 flag 값 False 전송
        init_msg = Bool()
        init_msg.data = False
        self.flag_robot3_pub.publish(init_msg)
        self.flag_robot8_pub.publish(init_msg)

        # 최신 이미지 프레임 저장 변수
        self.robot3_frame = None
        self.robot8_frame = None

        # 이전 사람 감지 상태 저장
        # 감지 상태가 바뀔 때만 flag를 publish하기 위해 사용
        self.prev_robot3_person_detected = False
        self.prev_robot8_person_detected = False

        # ================= Subscriber 생성 =================

        # robot3 OAK-D RGB compressed 이미지 구독
        self.sub_image_robot3 = self.create_subscription(
            CompressedImage,
            '/robot3/oakd/rgb/image_raw/compressed',
            self.robot3_image_callback,
            10
        )

        # robot8 OAK-D RGB compressed 이미지 구독
        self.sub_image_robot8 = self.create_subscription(
            CompressedImage,
            '/robot8/oakd/rgb/image_raw/compressed',
            self.robot8_image_callback,
            10
        )

        # 디버깅용 화면 확인이 필요할 때만 사용
        # cv2.namedWindow("Robot3 YOLO", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Robot8 YOLO", cv2.WINDOW_NORMAL)

        # 약 30FPS 주기로 robot3 YOLO 추론
        # 두 로봇을 동시에 추론하므로 PC/GPU 부하가 크면 조절 가능
        self.robot3_timer = self.create_timer(0.03, self.robot3_yolo_callback)

        # 약 30FPS 주기로 robot8 YOLO 추론
        # 시스템 과부하 발생 시 robot3/robot8 중 하나만 실행하거나 주기 증가 필요
        self.robot8_timer = self.create_timer(0.03, self.robot8_yolo_callback)

    # ================= CompressedImage → OpenCV 이미지 변환 =================
    def decode_compressed_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_img is None:
            return None

        # YOLO 입력 크기를 일정하게 맞추기 위해 640x480으로 resize
        return cv2.resize(
            cv_img,
            (640, 480),
            interpolation=cv2.INTER_AREA
        )

    # robot3 이미지 콜백
    # 최신 robot3 프레임을 저장
    def robot3_image_callback(self, msg):
        try:
            frame = self.decode_compressed_image(msg)
            if frame is not None:
                self.robot3_frame = frame

        except Exception as e:
            self.get_logger().error(f"robot3 이미지 변환 중 오류 발생: {e}")

    # robot8 이미지 콜백
    # 최신 robot8 프레임을 저장
    def robot8_image_callback(self, msg):
        try:
            frame = self.decode_compressed_image(msg)
            if frame is not None:
                self.robot8_frame = frame

        except Exception as e:
            self.get_logger().error(f"robot8 이미지 변환 중 오류 발생: {e}")

    # ================= YOLO 추론 함수 =================
    # 입력 프레임에서 person 클래스만 탐지
    # 사람 감지 시 박스가 그려진 이미지와 person_detected=True 반환
    def run_yolo(self, frame):
        display_frame = frame.copy()
        person_detected = False

        results = self.robot_model(
            frame,
            conf=CONF_THRESH,
            verbose=False
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # confidence가 낮은 객체는 무시
                if confidence < CONF_THRESH:
                    continue

                class_name = self.robot_model.names[cls_id]

                # person 클래스만 사용
                if class_name != "person":
                    continue

                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        return display_frame, person_detected

    # ================= robot3 YOLO 콜백 =================
    def robot3_yolo_callback(self):
        if self.robot3_frame is None:
            return

        frame = self.robot3_frame.copy()

        try:
            display_frame, person_detected = self.run_yolo(frame)

        except Exception as e:
            self.get_logger().error(f"robot3 YOLO 추론 중 오류 발생: {e}")
            return

        # 사람 감지 상태가 바뀐 경우에만 flag publish
        if person_detected != self.prev_robot3_person_detected:
            flag_msg = Bool()
            flag_msg.data = person_detected
            self.flag_robot3_pub.publish(flag_msg)
            self.prev_robot3_person_detected = person_detected

        # 사람 감지 시 박스 이미지 전송, 감지 안 됐을 때는 원본 이미지 전송
        result_msg = self.bridge.cv2_to_compressed_imgmsg(
            display_frame if person_detected else frame,
            dst_format='jpg'
        )

        self.robot3_result_image_pub.publish(result_msg)

        # 디버깅용 화면 확인
        # cv2.imshow("Robot3 YOLO", display_frame)
        # cv2.waitKey(1)

    # ================= robot8 YOLO 콜백 =================
    def robot8_yolo_callback(self):
        if self.robot8_frame is None:
            return

        frame = self.robot8_frame.copy()

        try:
            display_frame, person_detected = self.run_yolo(frame)

        except Exception as e:
            self.get_logger().error(f"robot8 YOLO 추론 중 오류 발생: {e}")
            return

        # 사람 감지 상태가 바뀐 경우에만 flag publish
        if person_detected != self.prev_robot8_person_detected:
            flag_msg = Bool()
            flag_msg.data = person_detected
            self.flag_robot8_pub.publish(flag_msg)
            self.prev_robot8_person_detected = person_detected

        # 사람 감지 시 박스 이미지 전송, 감지 안 됐을 때는 원본 이미지 전송
        result_msg = self.bridge.cv2_to_compressed_imgmsg(
            display_frame if person_detected else frame,
            dst_format='jpg'
        )

        self.robot8_result_image_pub.publish(result_msg)

        # # 디버깅용 화면 확인
        # cv2.imshow("Robot8 YOLO", display_frame)
        # cv2.waitKey(1)

    # 노드 종료 시 OpenCV 창 정리
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = RobotYoloPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()