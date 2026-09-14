#!/usr/bin/env python3

import math
import threading
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterValue, ParameterType

from std_msgs.msg import Bool
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import (
    Twist,
    Point,
    PointStamped,
    PoseStamped,
    Quaternion,
    PoseWithCovarianceStamped
)

from irobot_create_msgs.action import Dock

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import cv2
import numpy as np

from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator
)

from nav2_simple_commander.robot_navigator import TaskResult


ROBOT_NS = '/robot3'

AMCL_TOPIC = f'{ROBOT_NS}/amcl_pose'
DETECTED_BOTTOM_TOPIC = f'{ROBOT_NS}/detected_bottom'
CMD_VEL_TOPIC = f'{ROBOT_NS}/cmd_vel'
DOCK_ACTION = f'{ROBOT_NS}/dock'
WARNING_TOPIC = f'{ROBOT_NS}/warning'
CONTROLLER_SERVER_NAME = f'{ROBOT_NS}/controller_server'

# 평상시 / 추적 / 고홈 속도
NORMAL_MAX_VEL_X = 0.30
NORMAL_MAX_VEL_THETA = 1.00

# 이동 구간 속도
PATROL_MAX_VEL_X = 0.15
PATROL_MAX_VEL_THETA = 1.00

RGB_TOPIC = f'{ROBOT_NS}/oakd/rgb/image_raw/compressed'
DEPTH_TOPIC = f'{ROBOT_NS}/oakd/stereo/image_raw/compressedDepth'
CAMERA_INFO_TOPIC = f'{ROBOT_NS}/oakd/rgb/camera_info'

STOP_PUBLISH_HZ = 50.0
STOP_PERIOD = 1.0 / STOP_PUBLISH_HZ
WARNING_PUBLISH_HZ = 10.0

STOP_DISTANCE = 0.4
PROCESS_PERIOD = 0.5
DISPLAY_PERIOD = 0.1

LOST_DETECT_SECONDS = 1.0
SEARCH_ANGULAR_SPEED = 0.2
SEARCH_TURNS = 1.0
SEARCH_MAX_ROTATION = SEARCH_TURNS * 2.0 * math.pi


class DetectMonitorNode(Node):

    def __init__(self):
        super().__init__('detect_monitor_node')

        self.detected_requested = False
        self.stop_requested = False

        self.current_x = None
        self.current_y = None

        self.warning_state = False

        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.warning_pub = self.create_publisher(Bool, WARNING_TOPIC, 10)
        self.dock_client = ActionClient(self, Dock, DOCK_ACTION)

        self.warning_timer = self.create_timer(
            1.0 / WARNING_PUBLISH_HZ,
            self.warning_timer_callback
        )

        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            AMCL_TOPIC,
            self.amcl_callback,
            amcl_qos
        )

        self.create_subscription(
            Point,
            DETECTED_BOTTOM_TOPIC,
            self.detected_bottom_callback,
            10
        )

    def amcl_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def warning_timer_callback(self):
        msg = Bool()
        msg.data = self.warning_state
        self.warning_pub.publish(msg)

    def set_warning(self, value):
        if self.warning_state != value:
            self.get_logger().info(f'{WARNING_TOPIC}: {value}')

        self.warning_state = value

        msg = Bool()
        msg.data = value
        self.warning_pub.publish(msg)

    def detected_bottom_callback(self, msg):
        if msg.x != -1.0 or msg.y != -1.0:
            if not self.detected_requested:
                self.get_logger().warn(
                    f'Human detected: ({msg.x:.1f}, {msg.y:.1f})'
                )
                self.set_warning(True)

            self.detected_requested = True

    def reset_detection(self):
        self.detected_requested = False

    def is_detected_requested(self):
        return self.detected_requested

    def has_pose(self):
        return self.current_x is not None and self.current_y is not None

    def get_pose_xy(self):
        return self.current_x, self.current_y

    def publish_zero_cmd(self):
        self.cmd_vel_pub.publish(Twist())

    def force_stop_robot(self, duration_sec=1.0):
        count = int(duration_sec * STOP_PUBLISH_HZ)

        for _ in range(count):
            self.publish_zero_cmd()
            time.sleep(STOP_PERIOD)


def spin_detect_node(executor, node):
    while rclpy.ok() and not node.stop_requested:
        executor.spin_once(timeout_sec=0.01)


def get_current_xy_once():
    result = subprocess.run(
        ['ros2', 'topic', 'echo', AMCL_TOPIC, '--once'],
        capture_output=True,
        text=True
    )

    output = result.stdout.splitlines()

    x = None
    y = None

    for i, line in enumerate(output):
        stripped = line.strip()

        if stripped.startswith('position:'):
            for j in range(i + 1, min(i + 5, len(output))):
                pos_line = output[j].strip()

                if pos_line.startswith('x:'):
                    x = float(pos_line.split(':')[1].strip())
                elif pos_line.startswith('y:'):
                    y = float(pos_line.split(':')[1].strip())

        if x is not None and y is not None:
            break

    return x, y


def wait_for_current_xy(navigator, detect_node):
    while rclpy.ok():
        if detect_node.is_detected_requested():
            return None, None

        navigator.info(f'Reading current pose from {AMCL_TOPIC}...')

        x, y = get_current_xy_once()

        if x is not None and y is not None:
            navigator.info(f'Current Position: x={x:.3f}, y={y:.3f}')
            return x, y

        navigator.info('AMCL pose not received yet. Waiting...')
        time.sleep(0.5)

    return None, None


def wait_for_amcl_pose(control_node):
    while rclpy.ok():
        if control_node.has_pose():
            return control_node.get_pose_xy()

        control_node.get_logger().info('Waiting for AMCL pose...')
        time.sleep(0.5)

    return None



def make_double_param(name, value):
    param = ParameterMsg()
    param.name = name

    param.value = ParameterValue()
    param.value.type = ParameterType.PARAMETER_DOUBLE
    param.value.double_value = float(value)

    return param


def set_nav2_speed(control_node, linear_x, angular_z):
    service_name = f'{CONTROLLER_SERVER_NAME}/set_parameters'

    client = control_node.create_client(
        SetParameters,
        service_name
    )

    control_node.get_logger().info(
        f'Waiting for parameter service: {service_name}'
    )

    if not client.wait_for_service(timeout_sec=3.0):
        control_node.get_logger().warn(
            'controller_server set_parameters service not available. Speed change skipped.'
        )
        return False

    request = SetParameters.Request()
    request.parameters = [
        make_double_param('FollowPath.max_vel_x', linear_x),
        make_double_param('FollowPath.max_speed_xy', linear_x),
        make_double_param('FollowPath.max_vel_theta', angular_z),
    ]

    future = client.call_async(request)

    while rclpy.ok() and not future.done():
        time.sleep(0.02)

    response = future.result()

    if response is None:
        control_node.get_logger().warn(
            'Speed parameter service returned None.'
        )
        return False

    for result in response.results:
        if not result.successful:
            control_node.get_logger().warn(
                f'Speed parameter rejected: {result.reason}'
            )
            return False

    control_node.get_logger().info(
        f'Speed changed: '
        f'max_vel_x={linear_x}, '
        f'max_speed_xy={linear_x}, '
        f'max_vel_theta={angular_z}'
    )

    return True


def distance_xy(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def build_point_data():
    point_data = {
        'point1': {
            'xy': [-2.8649349212646484, -0.10677920281887054],
            'direction': TurtleBot4Directions.SOUTH
        },
        'point2': {
            'xy': [-2.5, 1.05],
            'direction': TurtleBot4Directions.WEST
        },
        'point3': {
            'xy': [-4.667, 1.275],
            'direction': TurtleBot4Directions.WEST
        },
        'point4': {
            'xy': [-2.557, 4.094],
            'direction': TurtleBot4Directions.EAST
        },
        'point5': {
            'xy': [-4.740, 4.403],
            'direction': TurtleBot4Directions.NORTH
        },
        'point24_mid': {
            'xy': [-2.434, 3.028],
            'direction': TurtleBot4Directions.SOUTH
        },
        'point35_mid': {
            'xy': [-4.700, 2.830],
            'direction': TurtleBot4Directions.NORTH
        },
    }

    target_xy = [
        (point_data['point24_mid']['xy'][0] + point_data['point35_mid']['xy'][0]) / 2.0,
        (point_data['point24_mid']['xy'][1] + point_data['point35_mid']['xy'][1]) / 2.0,
    ]

    point_data['target'] = {
        'xy': target_xy,
        'direction': TurtleBot4Directions.SOUTH
    }

    return point_data


def make_route(nearest_name):
    route_names = [nearest_name]

    if nearest_name in ['point2', 'point4']:
        route_names += ['point24_mid', 'target']

    elif nearest_name in ['point3', 'point5']:
        route_names += ['point35_mid', 'target']

    elif nearest_name == 'point1':
        route_names += ['point2', 'point24_mid', 'target']

    elif nearest_name in ['point24_mid', 'point35_mid']:
        route_names += ['target']

    return route_names


def cancel_and_stop(navigator, detect_node):
    set_nav2_speed(
        detect_node,
        NORMAL_MAX_VEL_X,
        NORMAL_MAX_VEL_THETA
    )

    try:
        navigator.cancelTask()
    except Exception:
        pass

    detect_node.force_stop_robot(1.0)


def go_to_pose(navigator, detect_node, pose, name):
    if detect_node.is_detected_requested():
        cancel_and_stop(navigator, detect_node)
        return 'TRACKING'

    navigator.info(f'Going to {name}...')
    navigator.startToPose(pose)

    while rclpy.ok():
        if detect_node.is_detected_requested():
            navigator.info('Human detected. Cancel current goal and start tracking.')
            cancel_and_stop(navigator, detect_node)
            return 'TRACKING'

        if navigator.isTaskComplete():
            result = navigator.getResult()
            navigator.info(f'{name} navigation result: {result}')

            if result == TaskResult.SUCCEEDED:
                return 'SUCCEEDED'

            return 'FAILED'

        time.sleep(0.02)

    return 'FAILED'


def go_to_pose_simple(navigator, pose, name):
    navigator.info(f'Going to {name}')
    navigator.startToPose(pose)

    while rclpy.ok():
        if navigator.isTaskComplete():
            result = navigator.getResult()
            navigator.info(f'{name} navigation result: {result}')

            if result == TaskResult.SUCCEEDED:
                return True

            return False

        time.sleep(0.02)

    return False


def dock_robot(control_node):
    control_node.get_logger().info('Waiting for dock action server...')

    if not control_node.dock_client.wait_for_server(timeout_sec=5.0):
        control_node.get_logger().warn('Dock action server not available.')
        return False

    goal_msg = Dock.Goal()
    send_future = control_node.dock_client.send_goal_async(goal_msg)

    while rclpy.ok() and not send_future.done():
        time.sleep(0.02)

    goal_handle = send_future.result()

    if not goal_handle.accepted:
        control_node.get_logger().warn('Dock goal rejected.')
        return False

    control_node.get_logger().info('Dock goal accepted.')

    result_future = goal_handle.get_result_async()

    while rclpy.ok() and not result_future.done():
        time.sleep(0.02)

    control_node.get_logger().info('Dock finished.')
    return True


def run_simple_gohome(navigator, control_node):
    current_pose = wait_for_amcl_pose(control_node)

    if current_pose is None:
        navigator.info('Gohome failed: AMCL pose unavailable.')
        return False

    current_x, current_y = current_pose

    point_xy = {
        '1': [-2.8649349212646484, -0.10677920281887054],
        '2': [-2.5, 1.05],
        '3': [-4.667, 1.275],
        '4': [-2.557, 4.094],
        '5': [-4.740, 4.403],
        '24_mid': [-2.434, 3.028],
        '35_mid': [-4.700, 2.830],
        '0': [-0.1, -0.1],
    }

    point_pose = {
        '1': navigator.getPoseStamped(point_xy['1'], TurtleBot4Directions.NORTH),
        '2': navigator.getPoseStamped(point_xy['2'], TurtleBot4Directions.EAST),
        '3': navigator.getPoseStamped(point_xy['3'], TurtleBot4Directions.NORTH),
        '4': navigator.getPoseStamped(point_xy['4'], TurtleBot4Directions.EAST),
        '5': navigator.getPoseStamped(point_xy['5'], TurtleBot4Directions.EAST),
        '24_mid': navigator.getPoseStamped(point_xy['24_mid'], TurtleBot4Directions.EAST),
        '35_mid': navigator.getPoseStamped(point_xy['35_mid'], TurtleBot4Directions.EAST),
        '0': navigator.getPoseStamped(point_xy['0'], TurtleBot4Directions.NORTH),
    }

    candidate_names = ['1', '2', '3', '4', '5', '24_mid', '35_mid']

    nearest_name = None
    nearest_dist = None

    for name in candidate_names:
        px, py = point_xy[name]
        dist = math.hypot(current_x - px, current_y - py)

        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_name = name

    route_table = {
        '1': ['1', '0'],
        '2': ['2', '1', '0'],
        '3': ['3', '2', '1', '0'],
        '4': ['4', '2', '1', '0'],
        '5': ['5', '3', '2', '1', '0'],
        '24_mid': ['24_mid', '2', '1', '0'],
        '35_mid': ['35_mid', '3', '2', '1', '0'],
    }

    route = route_table[nearest_name]

    navigator.info(
        f'Gohome current pose: x={current_x:.3f}, y={current_y:.3f}'
    )
    navigator.info(
        f'Gohome nearest point: {nearest_name}, distance={nearest_dist:.3f}m'
    )
    navigator.info(f'Gohome route: {" -> ".join(route)}')

    for name in route:
        if not go_to_pose_simple(navigator, point_pose[name], name):
            navigator.info('Gohome route failed.')
            return False

    control_node.force_stop_robot(1.0)

    dock_success = dock_robot(control_node)

    if dock_success:
        navigator.info('Gohome complete. Docked.')
        return True

    navigator.info('Dock failed.')
    return False


def one_turn_search(navigator, detect_node, search_name='search'):
    navigator.info(f'{search_name}: start one-turn search.')

    detect_node.reset_detection()

    try:
        navigator.cancelTask()
    except Exception:
        pass

    detect_node.force_stop_robot(0.5)

    start_time = time.time()
    max_search_time = SEARCH_MAX_ROTATION / abs(SEARCH_ANGULAR_SPEED)

    while rclpy.ok():
        if detect_node.is_detected_requested():
            navigator.info(f'{search_name}: human detected during search.')
            cancel_and_stop(navigator, detect_node)
            return 'TRACKING'

        elapsed = time.time() - start_time

        if elapsed >= max_search_time:
            break

        msg = Twist()
        msg.angular.z = SEARCH_ANGULAR_SPEED
        detect_node.cmd_vel_pub.publish(msg)

        time.sleep(STOP_PERIOD)

    detect_node.force_stop_robot(1.0)
    navigator.info(f'{search_name}: search finished. Human not found.')

    return 'GOHOME'


class TrackingModeNode(Node):

    def __init__(self):
        super().__init__('tracking_mode_node')

        self.K = None
        self.lock = threading.Lock()

        self.rgb_image = None
        self.depth_image = None
        self.camera_frame = None
        self.detected_bottom = None
        self.is_detected = False
        self.display_image = None

        self.goal_sent = False
        self.stopped = False
        self.follow_mode = False
        self.current_distance = None

        self.last_detected_bottom = None
        self.last_seen_time = None

        self.searching = False
        self.search_start_time = None
        self.search_direction = 0.0

        self.tracking_finished = False
        self.finish_reason = None

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = TurtleBot4Navigator()
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)
        self.create_subscription(CompressedImage, RGB_TOPIC, self.rgb_callback, 10)
        self.create_subscription(CompressedImage, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(Point, DETECTED_BOTTOM_TOPIC, self.bottom_callback, 10)

        self.gui_thread_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()

        self.get_logger().info('Tracking mode started.')
        self.get_logger().info('TF Tree 안정화 시작. 5초 후 추적을 시작합니다.')

        self.start_timer = self.create_timer(5.0, self.start_transform)

    def start_transform(self):
        self.get_logger().info('TF Tree 안정화 완료. 최초 goal 전송 및 거리 감시 시작.')
        self.timer = self.create_timer(PROCESS_PERIOD, self.process_bottom)
        self.display_timer = self.create_timer(DISPLAY_PERIOD, self.update_display)
        self.start_timer.cancel()

    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)

        if not self.logged_intrinsics:
            self.get_logger().info(
                f'Camera intrinsics received: '
                f'fx={self.K[0, 0]:.2f}, '
                f'fy={self.K[1, 1]:.2f}, '
                f'cx={self.K[0, 2]:.2f}, '
                f'cy={self.K[1, 2]:.2f}'
            )
            self.logged_intrinsics = True

    def rgb_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if rgb is None or rgb.size == 0:
            return

        with self.lock:
            self.rgb_image = rgb

        if not self.logged_rgb_shape:
            self.get_logger().info(f'RGB image decoded: {rgb.shape}')
            self.logged_rgb_shape = True

    def depth_callback(self, msg):
        if len(msg.data) <= 12:
            return

        depth_data = np.frombuffer(msg.data[12:], np.uint8)
        depth = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)

        if depth is None or depth.size == 0:
            return

        with self.lock:
            self.depth_image = depth
            self.camera_frame = msg.header.frame_id

        if not self.logged_depth_shape:
            self.get_logger().info(
                f'CompressedDepth image decoded: {depth.shape}, '
                f'dtype={depth.dtype}, frame={msg.header.frame_id}'
            )
            self.logged_depth_shape = True

    def bottom_callback(self, msg):
        with self.lock:
            if msg.x == -1.0 and msg.y == -1.0:
                self.is_detected = False
                self.detected_bottom = None
                self.current_distance = None
                return

            bottom = (int(msg.x), int(msg.y))

            self.is_detected = True
            self.detected_bottom = bottom
            self.last_detected_bottom = bottom
            self.last_seen_time = self.get_clock().now()
            self.current_distance = None

            if self.searching:
                self.get_logger().info(
                    'Target detected during one-turn search. Resume tracking.'
                )

                self.goal_sent = False
                self.stopped = False
                self.follow_mode = False

            self.searching = False
            self.search_start_time = None
            self.search_direction = 0.0

    def make_goal_from_bottom(self, x, y, z, frame_id, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        Z = z

        pt_camera = PointStamped()
        pt_camera.header.stamp = Time().to_msg()
        pt_camera.header.frame_id = frame_id
        pt_camera.point.x = X
        pt_camera.point.y = Y
        pt_camera.point.z = Z

        pt_map = self.tf_buffer.transform(
            pt_camera,
            'map',
            timeout=Duration(seconds=1.0)
        )

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = pt_map.point.x
        goal_pose.pose.position.y = pt_map.point.y
        goal_pose.pose.position.z = 0.0

        yaw = 0.0
        goal_pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

        return goal_pose, pt_map

    def stop_robot(self):
        try:
            self.navigator.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'cancelTask failed: {e}')

        self.cmd_vel_pub.publish(Twist())

        self.stopped = True
        self.follow_mode = True

        self.get_logger().info(
            'Stop distance reached. Navigation cancelled and robot stopped. Follow mode enabled.'
        )

    def start_lost_target_search(self, last_bottom, image_width):
        try:
            self.navigator.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'Lost target cancelTask failed: {e}')

        self.cmd_vel_pub.publish(Twist())

        last_x, _ = last_bottom
        image_center_x = image_width / 2.0

        if last_x < image_center_x:
            self.search_direction = 1.0
            direction_text = 'LEFT'
        else:
            self.search_direction = -1.0
            direction_text = 'RIGHT'

        self.search_start_time = self.get_clock().now()
        self.searching = True
        self.stopped = True
        self.current_distance = None

        self.goal_sent = False

        self.get_logger().warn(
            f'Target lost over {LOST_DETECT_SECONDS:.1f}s. '
            f'Cancel tracking goal, stop robot, and start {SEARCH_TURNS:.1f} turn search to {direction_text}.'
        )

    def update_lost_target_search(self):
        if self.search_start_time is None:
            return

        now = self.get_clock().now()
        elapsed_search = (now - self.search_start_time).nanoseconds / 1e9
        rotated_angle = elapsed_search * abs(SEARCH_ANGULAR_SPEED)

        if rotated_angle >= SEARCH_MAX_ROTATION:
            self.finish_tracking_go_home()
            return

        msg = Twist()
        msg.angular.z = self.search_direction * SEARCH_ANGULAR_SPEED
        self.cmd_vel_pub.publish(msg)

    def finish_tracking_go_home(self):
        try:
            self.navigator.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'Finish tracking cancelTask failed: {e}')

        self.cmd_vel_pub.publish(Twist())

        self.get_logger().warn(
            f'Target not found during {SEARCH_TURNS:.1f} turn search. '
            'Tracking finished. Start gohome.'
        )

        self.gui_thread_stop.set()
        self.tracking_finished = True
        self.finish_reason = 'GOHOME'

    def process_bottom(self):
        if self.tracking_finished:
            return

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            bottom = self.detected_bottom
            last_bottom = self.last_detected_bottom
            last_seen_time = self.last_seen_time
            frame_id = self.camera_frame
            K = self.K.copy() if self.K is not None else None
            is_detected = self.is_detected
            searching = self.searching

        if searching:
            if is_detected and bottom is not None:
                try:
                    self.navigator.cancelTask()
                except Exception:
                    pass

                self.cmd_vel_pub.publish(Twist())

                self.searching = False
                self.search_start_time = None
                self.search_direction = 0.0
                self.stopped = False
                self.follow_mode = False
                self.goal_sent = False

                self.get_logger().info(
                    'Target detected again. Stop one-turn search and resume first tracking goal.'
                )
            else:
                self.update_lost_target_search()
                return

        if not is_detected or bottom is None:
            self.current_distance = None

            if last_bottom is None or last_seen_time is None:
                self.cmd_vel_pub.publish(Twist())
                return

            elapsed_lost = (
                self.get_clock().now() - last_seen_time
            ).nanoseconds / 1e9

            if elapsed_lost < LOST_DETECT_SECONDS:
                self.cmd_vel_pub.publish(Twist())
                return

            if rgb is not None:
                image_width = rgb.shape[1]
            elif depth is not None:
                image_width = depth.shape[1]
            else:
                self.cmd_vel_pub.publish(Twist())
                return

            self.start_lost_target_search(last_bottom, image_width)
            return

        if depth is None or frame_id is None or K is None:
            return

        x, y = bottom
        h, w = depth.shape[:2]

        if x < 0 or x >= w or y < 0 or y >= h:
            self.get_logger().warn('Detected bottom is outside depth image.')
            return

        z = float(depth[y, x]) / 1000.0

        with self.lock:
            self.current_distance = z
            self.last_detected_bottom = bottom
            self.last_seen_time = self.get_clock().now()

        self.get_logger().info(f'Current distance: {z:.2f} m')

        if not (0.2 < z < 5.0):
            self.get_logger().warn(
                f'Invalid depth value at bottom: {z:.2f} m'
            )
            return

        if self.follow_mode:
            if z <= STOP_DISTANCE:
                try:
                    self.navigator.cancelTask()
                except Exception as e:
                    self.get_logger().warn(f'Follow mode cancelTask failed: {e}')

                self.cmd_vel_pub.publish(Twist())
                self.stopped = True

                self.get_logger().info(
                    f'Follow mode: target within {STOP_DISTANCE:.2f} m. '
                    f'Navigation cancelled. Holding stop. distance={z:.2f} m'
                )
                return

            try:
                goal_pose, pt_map = self.make_goal_from_bottom(
                    x, y, z, frame_id, K
                )

                self.navigator.goToPose(goal_pose)

                self.stopped = False

                self.get_logger().info(
                    f'Follow mode: updated goal from bottom ({x}, {y}) -> '
                    f'map ({pt_map.point.x:.2f}, {pt_map.point.y:.2f}), '
                    f'distance={z:.2f} m'
                )

            except Exception as e:
                self.get_logger().warn(f'Follow mode TF or goal error: {e}')

            return

        if z <= STOP_DISTANCE:
            self.stop_robot()
            return

        if self.goal_sent:
            return

        try:
            goal_pose, pt_map = self.make_goal_from_bottom(
                x, y, z, frame_id, K
            )

            self.navigator.goToPose(goal_pose)
            self.goal_sent = True

            self.get_logger().info(
                f'Sent FIRST goal from bottom ({x}, {y}) -> '
                f'map ({pt_map.point.x:.2f}, {pt_map.point.y:.2f}), '
                f'distance={z:.2f} m'
            )

        except Exception as e:
            self.get_logger().warn(f'TF or goal error: {e}')

    def update_display(self):
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            bottom = self.detected_bottom
            is_detected = self.is_detected
            distance = self.current_distance
            goal_sent = self.goal_sent
            stopped = self.stopped
            follow_mode = self.follow_mode
            searching = self.searching

        if rgb is None or depth is None:
            return

        rgb_display = rgb.copy()

        depth_normalized = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        depth_display = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

        status = (
            f'goal_sent: {goal_sent}, stopped: {stopped}, '
            f'follow: {follow_mode}, search: {searching}'
        )

        cv2.putText(
            rgb_display,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if goal_sent else (0, 0, 255),
            2
        )

        if distance is not None:
            cv2.putText(
                rgb_display,
                f'distance: {distance:.2f} m / stop: {STOP_DISTANCE:.2f} m',
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        if is_detected and bottom is not None:
            x, y = bottom

            cv2.circle(rgb_display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                rgb_display,
                f'bottom: ({x}, {y})',
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                z = float(depth[y, x]) / 1000.0
                cv2.circle(depth_display, (x, y), 6, (255, 255, 255), -1)

                cv2.putText(
                    depth_display,
                    f'{z:.2f} m',
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        combined = np.hstack((rgb_display, depth_display))

        with self.lock:
            self.display_image = combined.copy()

    def gui_loop(self):
        window_name = 'Tracking Mode | RGB left / Depth right'

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)
        cv2.moveWindow(window_name, 100, 100)

        while not self.gui_thread_stop.is_set():
            with self.lock:
                img = self.display_image.copy() if self.display_image is not None else None

            if img is not None:
                cv2.imshow(window_name, img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.gui_thread_stop.set()
                    self.tracking_finished = True
                    self.finish_reason = 'FINISHED'
                    break
            else:
                cv2.waitKey(10)


def run_tracking_mode():
    node = TrackingModeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.tracking_finished:
            executor.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        node.finish_reason = 'FINISHED'

    node.gui_thread_stop.set()

    if node.gui_thread.is_alive():
        node.gui_thread.join(timeout=1.0)

    reason = node.finish_reason

    try:
        node.navigator.cancelTask()
    except Exception:
        pass

    node.cmd_vel_pub.publish(Twist())

    executor.shutdown()
    node.navigator.destroy_node()
    node.destroy_node()
    cv2.destroyAllWindows()

    return reason


def run_tracking_with_research_loop(navigator, detect_node):
    while rclpy.ok():
        navigator.info('Switching to tracking mode.')

        set_nav2_speed(
            detect_node,
            NORMAL_MAX_VEL_X,
            NORMAL_MAX_VEL_THETA
        )

        try:
            navigator.cancelTask()
        except Exception:
            pass

        detect_node.force_stop_robot(1.0)

        tracking_result = run_tracking_mode()

        navigator.info(f'Tracking mode finished: {tracking_result}')

        if tracking_result == 'GOHOME':
            navigator.info('Tracking failed to find target. Start gohome.')
            detect_node.set_warning(False)
            set_nav2_speed(
                detect_node,
                NORMAL_MAX_VEL_X,
                NORMAL_MAX_VEL_THETA
            )
            run_simple_gohome(navigator, detect_node)
            return 'GOHOME'

        search_result = one_turn_search(
            navigator,
            detect_node,
            search_name='after_tracking_search'
        )

        if search_result == 'TRACKING':
            continue

        if search_result == 'GOHOME':
            navigator.info('Human not found after one-turn search. Start gohome.')
            detect_node.set_warning(False)
            set_nav2_speed(
                detect_node,
                NORMAL_MAX_VEL_X,
                NORMAL_MAX_VEL_THETA
            )
            run_simple_gohome(navigator, detect_node)
            return 'GOHOME'

        return 'FINISHED'


def main():
    rclpy.init(args=[
        '--ros-args',
        '-r', '__ns:=/robot3',
        '-r', '/tf:=/robot3/tf',
        '-r', '/tf_static:=/robot3/tf_static',
    ])

    navigator = TurtleBot4Navigator()
    detect_node = DetectMonitorNode()

    executor = SingleThreadedExecutor()
    executor.add_node(detect_node)

    spin_thread = threading.Thread(
        target=spin_detect_node,
        args=(executor, detect_node),
        daemon=True
    )
    spin_thread.start()

    mission_result = None

    try:
        navigator.info('Simple to_sector route with tracking and gohome started.')

        set_nav2_speed(
            detect_node,
            NORMAL_MAX_VEL_X,
            NORMAL_MAX_VEL_THETA
        )

        if navigator.getDockedStatus():
            navigator.info('Robot is docked. Undocking...')
            navigator.undock()
            navigator.info('Undock finished.')
        else:
            navigator.info('Robot is already undocked. Continue.')

        current_x, current_y = wait_for_current_xy(navigator, detect_node)

        if detect_node.is_detected_requested():
            mission_result = 'TRACKING'

        elif current_x is None or current_y is None:
            navigator.info('Failed to get current position.')
            mission_result = 'FAILED'

        else:
            point_data = build_point_data()

            candidate_names = [
                'point1',
                'point2',
                'point3',
                'point4',
                'point5',
                'point24_mid',
                'point35_mid'
            ]

            nearest_name = None
            nearest_distance = None

            for name in candidate_names:
                px, py = point_data[name]['xy']
                dist = distance_xy(current_x, current_y, px, py)

                navigator.info(f'{name}: distance = {dist:.3f} m')

                if nearest_distance is None or dist < nearest_distance:
                    nearest_name = name
                    nearest_distance = dist

            navigator.info(
                f'Nearest point: {nearest_name}, distance={nearest_distance:.3f} m'
            )

            route_names = make_route(nearest_name)

            navigator.info(f'Final route: {" -> ".join(route_names)}')

            navigator.info('Entering nearest point approach mode.')

            for i, name in enumerate(route_names):
                if i == 1:
                    navigator.info('Nearest point reached. Entering patrol speed mode.')
                    set_nav2_speed(
                        detect_node,
                        PATROL_MAX_VEL_X,
                        PATROL_MAX_VEL_THETA
                    )

                pose = navigator.getPoseStamped(
                    point_data[name]['xy'],
                    point_data[name]['direction']
                )

                result = go_to_pose(navigator, detect_node, pose, name)

                if result == 'TRACKING':
                    mission_result = 'TRACKING'
                    break

                if result != 'SUCCEEDED':
                    navigator.info(f'Navigation failed at {name}. Stop mission.')
                    mission_result = 'FAILED'
                    break

            if mission_result is None:
                navigator.info('Route finished. Target reached.')

                set_nav2_speed(
                    detect_node,
                    NORMAL_MAX_VEL_X,
                    NORMAL_MAX_VEL_THETA
                )

                search_result = one_turn_search(
                    navigator,
                    detect_node,
                    search_name='target_final_search'
                )

                if search_result == 'TRACKING':
                    mission_result = 'TRACKING'

                elif search_result == 'GOHOME':
                    navigator.info('Target final search failed. Start gohome.')
                    detect_node.set_warning(False)
                    set_nav2_speed(
                        detect_node,
                        NORMAL_MAX_VEL_X,
                        NORMAL_MAX_VEL_THETA
                    )
                    run_simple_gohome(navigator, detect_node)
                    mission_result = 'GOHOME'

                else:
                    mission_result = 'SUCCEEDED'

        if mission_result == 'TRACKING':
            set_nav2_speed(
                detect_node,
                NORMAL_MAX_VEL_X,
                NORMAL_MAX_VEL_THETA
            )
            run_tracking_with_research_loop(navigator, detect_node)

    except KeyboardInterrupt:
        navigator.info('Keyboard interrupt. Cancel task.')

    try:
        navigator.cancelTask()
    except Exception:
        pass

    detect_node.set_warning(False)
    detect_node.force_stop_robot(1.0)
    detect_node.stop_requested = True

    executor.shutdown()
    spin_thread.join(timeout=1.0)

    detect_node.destroy_node()
    navigator.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()