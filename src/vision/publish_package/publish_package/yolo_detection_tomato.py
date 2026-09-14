import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage

from std_msgs.msg import Bool

from cv_bridge import CvBridge
from ultralytics import YOLO

CONF_THRESH = 0.8

YOLO_PATH = '/home/rokey/test_ws/src/publish_package/models/best_yolov8n_web.pt'

class TomatoYoloPublisher(Node):

    def __init__(self):
        super().__init__('tomato_yolo_publisher')

        ### =================== YOLO 모델 불러오기 =================================          
        self.webcam_model = YOLO(YOLO_PATH)
        
        ### ============== 웹캠 열기 ===========================
        #(2번 카메라)
        self.cap = cv2.VideoCapture(2)

        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다.")

        # 웹캠 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        ### =================================================

        # OpenCV ↔ ROS Image 변환
        self.bridge = CvBridge()

        # ============== Publisher 생성 ======================
    
        # 터틀봇 위 웹캠 원본 이미지 Publish 
        self.raw_image_pub = self.create_publisher(
            CompressedImage,
            '/detection/tomato/raw',
            10
        )

        # 터틀봇 위 웹캠 비정상 이미지 publish
        self.rotten_image_pub = self.create_publisher(
        CompressedImage,
        '/detection/tomato/rotten',
        10
        )

        # 터틀봇 위 웹캠 토마토 감지 flag 전송
        self.flag_web_pub = self.create_publisher(
        Bool,
        '/detection/tomato/flag',
        10
        )
        init_flag_msg = Bool()
        init_flag_msg.data = False
        self.flag_web_pub.publish(init_flag_msg)

        self.webcam_frame = None      

        ### =========================== 사람 감지 로직 쓰려면 주석 풀기 =======================================
        # cv2.namedWindow("Web YOLO", cv2.WINDOW_NORMAL)
 
        # self.frame = None
        # self.person_detected = False

        # 약 30FPS 주기로 웹캠 프레임을 읽어 raw 이미지 publish
        # 시스템 모니터 서버 측이나 현재 pc 과부하 심하면 조절 가능
        self.webcam_image_timer = self.create_timer(0.03, self.webcam_image_callback)

        # 약 30FPS 주기로 YOLO 추론 및 rotten 이미지 publish
        # TurtleBot 이동 중 rotten 감지 순간을 놓치지 않기 위해 주기를 짧게 설정
        self.webcam_yolo_timer = self.create_timer(0.03, self.webcam_yolo_callback)
        
   
    ### ========================== webcam 이미지 콜백 ==================================

    # 웹캠에서 최신 프레임을 읽고 raw 이미지 토픽으로 publish
    def webcam_image_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("웹캠 읽기 실패")
            return
        
        self.webcam_frame = frame.copy()

        raw_msg = self.bridge.cv2_to_compressed_imgmsg(
        frame,
        dst_format='jpg'
        )

        self.raw_image_pub.publish(raw_msg)

    
    # =============== webcam YOLO 추론 콜백 =============================

    # 최신 웹캠 프레임에 YOLO를 적용하여 rotten 토마토를 감지
    # rotten 감지 시 박스가 그려진 이미지를 publish하고 flag=True 전송      
    def webcam_yolo_callback(self):
       
        if self.webcam_frame is None:
            return
        
        frame = self.webcam_frame.copy()

        ### ====== # todo: 사람 감지 기능과 통합할 경우 활성화====
        # if self.person_detected:
        #     flag_msg = Bool()
        #     flag_msg.data = False
        #     self.flag_web_pub.publish(flag_msg)

        #     # 사람이 보일 때도 rotten 토픽에는 원본 이미지 유지
        #     rotten_msg = self.bridge.cv2_to_compressed_imgmsg(
        #         frame,
        #         dst_format='jpg'
        #     )
        #     self.rotten_image_pub.publish(rotten_msg)
        #     return
        ### =========================================================
        
        results = self.webcam_model(frame, conf=CONF_THRESH, verbose=False)
                 
        # YOLO 박스가 그려진 화면 생성
        display_frame = frame.copy()
        flag=False

        # 객체 추론 부분
        for result in results:

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.webcam_model.names[cls_id]

                # confidence가 너무 낮으면 무시
                if confidence < CONF_THRESH:
                    continue

                # rotten만 라벨링
                if class_name == 'rotten':
                    flag = True

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    label = f"{class_name} {confidence:.2f}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
        # # 웹으로 보낼 이미지 미리 확인
        # cv2.imshow("yolo detection", display_frame)
        # cv2.waitKey(1)
      
        ## =========== 침입자 감지 했을 때 토마토 탐지X flag =================
        # if self.person_detected:
        #     flag = False

        flag_msg = Bool()
        flag_msg.data = flag
        self.flag_web_pub.publish(flag_msg)

        # rotten 감지됐을 때 라벨링 이미지 전송
        if flag:
            # rotten 발견 → 박스 이미지
            rotten_msg = self.bridge.cv2_to_compressed_imgmsg(
                display_frame,
                dst_format='jpg'
            )
        else:
            # rotten 없음 → 원본 이미지
            rotten_msg = self.bridge.cv2_to_compressed_imgmsg(
                frame,
                dst_format='jpg'
            )
        self.rotten_image_pub.publish(rotten_msg)

    # 종료 시 카메라 해제
    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):

    rclpy.init(args=args)

    node = TomatoYoloPublisher()

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