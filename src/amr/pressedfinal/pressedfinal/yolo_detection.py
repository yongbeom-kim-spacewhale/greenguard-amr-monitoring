#!/usr/bin/env python3

import os
import signal
import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from ultralytics import YOLO


# =========================
# YOLO 및 탐지 설정
# =========================

TARGET_CLASS_ID = 0
DETECT_SECONDS = 0.5


class YoloCompressedViewer(Node):

    def __init__(self):
        super().__init__('yolo_compressed_viewer')

        self.model = YOLO('yolov8n.pt')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/robot3/oakd/rgb/image_raw/compressed',
            self.callback,
            10
        )

        self.center_pub = self.create_publisher(
            Point,
            '/robot3/detected_bottom',
            10
        )

        self.detect_start_time = None
        self.is_detected = False

        self.get_logger().info("YOLO compressed image viewer started")

    def publish_detected_center(self, center_x, center_y):
        msg = Point()
        msg.x = float(center_x)
        msg.y = float(center_y)
        msg.z = 0.0

        self.center_pub.publish(msg)

    def publish_not_detected(self):
        self.publish_detected_center(-1.0, -1.0)

    def get_box_center(self, xyxy):
        x1, y1, x2, y2 = xyxy

        center_x = (x1 + x2) / 2.0
        center_y = y2

        return center_x, center_y

    def update_detection_state(self, detected_now):
        now = self.get_clock().now().nanoseconds / 1e9

        if detected_now:
            if self.detect_start_time is None:
                self.detect_start_time = now

            elapsed = now - self.detect_start_time

            if elapsed >= DETECT_SECONDS:
                self.is_detected = True

        else:
            self.detect_start_time = None
            self.is_detected = False

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.update_detection_state(False)
            self.publish_not_detected()
            return

        results = self.model(frame, verbose=False, conf=0.8)

        result = results[0]
        boxes = result.boxes

        detected_now = False
        target_center = None

        if boxes is not None and len(boxes) > 0:
            target_indices = []

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])

                if cls_id == TARGET_CLASS_ID:
                    target_indices.append(i)

            if len(target_indices) > 0:
                best_idx = max(
                    target_indices,
                    key=lambda i: boxes[i].xyxy[0][3].item()
                )

                detected_now = True

                xyxy = boxes[best_idx].xyxy[0].cpu().numpy()
                target_center = self.get_box_center(xyxy)

                keep_indices = np.array([best_idx])

                result.boxes = boxes[keep_indices]

            else:
                result.boxes = boxes[[]]

        self.update_detection_state(detected_now)

        if self.is_detected and target_center is not None:
            center_x, center_y = target_center
            self.publish_detected_center(center_x, center_y)
        else:
            self.publish_not_detected()

        annotated_frame = result.plot()

        status_text = f"is_detected: {self.is_detected}"

        cv2.putText(
            annotated_frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if self.is_detected else (0, 0, 255),
            2
        )

        if self.is_detected and target_center is not None:
            center_x, center_y = target_center

            cv2.circle(
                annotated_frame,
                (int(center_x), int(center_y)),
                6,
                (255, 0, 0),
                -1
            )

            center_text = f"bottom: ({int(center_x)}, {int(center_y)})"

            cv2.putText(
                annotated_frame,
                center_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        cv2.imshow("TurtleBot4 YOLO Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            self.get_logger().info("C key pressed. Shutting down...")
            cv2.destroyAllWindows()
            os.kill(os.getpid(), signal.SIGINT)


def main(args=None):
    rclpy.init(args=args)

    node = YoloCompressedViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()