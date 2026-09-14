#!/usr/bin/env python3

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy
)

from geometry_msgs.msg import PoseWithCovarianceStamped
from irobot_create_msgs.action import Dock
from rclpy.action import ActionClient

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterValue, ParameterType

from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator
)


ROBOT_NS = '/robot3'

AMCL_POSE_TOPIC = f'{ROBOT_NS}/amcl_pose'
DOCK_ACTION = f'{ROBOT_NS}/dock'
CONTROLLER_SERVER_NAME = f'{ROBOT_NS}/controller_server'

# 순찰 반복 횟수
PATROL_LOOP_COUNT = 2

# 반복 순찰 전 / gohome 속도
NORMAL_MAX_VEL_X = 0.30
NORMAL_MAX_VEL_THETA = 1.00

# 반복 순찰 구간 속도
PATROL_MAX_VEL_X = 0.15
PATROL_MAX_VEL_THETA = 1.00


class PatrolControlNode(Node):

    def __init__(self):
        super().__init__('patrol_control_node')

        self.current_x = None
        self.current_y = None
        self.stop_requested = False

        self.dock_client = ActionClient(self, Dock, DOCK_ACTION)

        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            AMCL_POSE_TOPIC,
            self.amcl_callback,
            amcl_qos
        )

    def amcl_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def has_pose(self):
        return self.current_x is not None and self.current_y is not None

    def get_pose_xy(self):
        return self.current_x, self.current_y


def spin_patrol_node(executor, node):
    while rclpy.ok() and not node.stop_requested:
        executor.spin_once(timeout_sec=0.01)


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


def find_nearest_point(current_x, current_y, point_positions):
    nearest_name = None
    nearest_dist = None

    for name, xy in point_positions.items():
        px, py = xy
        dist = distance_xy(current_x, current_y, px, py)

        if nearest_dist is None or dist < nearest_dist:
            nearest_name = name
            nearest_dist = dist

    return nearest_name, nearest_dist


def go_to_pose_simple(navigator, pose, name):
    navigator.info(f'Going to {name}')
    navigator.startToPose(pose)

    while rclpy.ok():
        if navigator.isTaskComplete():
            navigator.info(f'Arrived at {name}')
            return True

        time.sleep(0.01)

    return False


def get_next_loop_index(loop_names, current_name):
    for i, name in enumerate(loop_names):
        if name == current_name:
            return (i + 1) % len(loop_names)

    return 0


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

    candidate_names = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '24_mid',
        '35_mid'
    ]

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

    dock_success = dock_robot(control_node)

    if dock_success:
        navigator.info('Gohome complete. Docked.')
        return True

    navigator.info('Dock failed.')
    return False


def main():
    rclpy.init(args=[
        '--ros-args',
        '-r', '__ns:=/robot3',
        '-r', '/tf:=/robot3/tf',
        '-r', '/tf_static:=/robot3/tf_static',
    ])

    navigator = TurtleBot4Navigator()
    control_node = PatrolControlNode()

    executor = SingleThreadedExecutor()
    executor.add_node(control_node)

    spin_thread = threading.Thread(
        target=spin_patrol_node,
        args=(executor, control_node),
        daemon=True
    )
    spin_thread.start()

    navigator.info('Basic patrol + gohome + speed control node started.')

    point_positions = {
        'point1': [-2.8649349212646484, -0.10677920281887054],
        'point2': [-2.5, 1.05],
        'point3': [-4.667, 1.275],
        'point4': [-2.557, 4.094],
        'point5': [-4.740, 4.403],
        'point24_mid': [-2.434, 3.028],
        'point35_mid': [-4.700, 2.830],
    }

    poses = {
        'point1': navigator.getPoseStamped(
            point_positions['point1'],
            TurtleBot4Directions.SOUTH
        ),
        'point2': navigator.getPoseStamped(
            point_positions['point2'],
            TurtleBot4Directions.WEST
        ),
        'point3': navigator.getPoseStamped(
            point_positions['point3'],
            TurtleBot4Directions.WEST
        ),
        'point4': navigator.getPoseStamped(
            point_positions['point4'],
            TurtleBot4Directions.EAST
        ),
        'point5': navigator.getPoseStamped(
            point_positions['point5'],
            TurtleBot4Directions.NORTH
        ),
        'point24_mid': navigator.getPoseStamped(
            point_positions['point24_mid'],
            TurtleBot4Directions.SOUTH
        ),
        'point35_mid': navigator.getPoseStamped(
            point_positions['point35_mid'],
            TurtleBot4Directions.NORTH
        ),
    }

    loop_names = [
        'point2',
        'point3',
        'point35_mid',
        'point24_mid',
        'point35_mid',
        'point5',
        'point4',
    ]

    try:
        set_nav2_speed(
            control_node,
            NORMAL_MAX_VEL_X,
            NORMAL_MAX_VEL_THETA
        )

        if navigator.getDockedStatus():
            navigator.info('Robot is docked. Undocking...')
            navigator.undock()
        else:
            navigator.info('Robot is already undocked. Continue.')

        current_pose = wait_for_amcl_pose(control_node)

        if current_pose is None:
            navigator.info('Failed to get AMCL pose.')
            return

        current_x, current_y = current_pose

        nearest_name, nearest_dist = find_nearest_point(
            current_x,
            current_y,
            point_positions
        )

        navigator.info(
            f'Current pose: x={current_x:.3f}, y={current_y:.3f}'
        )
        navigator.info(
            f'Nearest point: {nearest_name}, distance={nearest_dist:.3f}m'
        )

        navigator.info(f'Going to nearest point: {nearest_name}')

        if not go_to_pose_simple(navigator, poses[nearest_name], nearest_name):
            navigator.info('Failed to reach nearest point.')
            return

        if nearest_name == 'point1':
            navigator.info('Arrived at point1. Next goal is point2.')

            if not go_to_pose_simple(navigator, poses['point2'], 'point2'):
                navigator.info('Failed to reach point2.')
                return

            next_index = get_next_loop_index(loop_names, 'point2')

        else:
            next_index = get_next_loop_index(loop_names, nearest_name)

        navigator.info('Entering patrol loop speed mode.')

        set_nav2_speed(
            control_node,
            PATROL_MAX_VEL_X,
            PATROL_MAX_VEL_THETA
        )

        navigator.info(
            f'Starting patrol loop. Target loops: {PATROL_LOOP_COUNT}'
        )

        completed_loops = 0

        while rclpy.ok():
            target_name = loop_names[next_index]

            if not go_to_pose_simple(navigator, poses[target_name], target_name):
                navigator.info('Patrol goal failed.')
                return

            next_index = (next_index + 1) % len(loop_names)

            if next_index == 0:
                completed_loops += 1

                navigator.info(
                    f'Patrol loop {completed_loops}/{PATROL_LOOP_COUNT} completed.'
                )

                if completed_loops >= PATROL_LOOP_COUNT:
                    break

        navigator.info('Patrol completed.')
        navigator.info('Restoring normal speed before gohome.')

        set_nav2_speed(
            control_node,
            NORMAL_MAX_VEL_X,
            NORMAL_MAX_VEL_THETA
        )

        navigator.info('Starting gohome mode.')
        run_simple_gohome(navigator, control_node)

    except KeyboardInterrupt:
        navigator.info('KeyboardInterrupt received.')

    finally:
        navigator.info('Stopping patrol + gohome node.')

        try:
            navigator.cancelTask()
        except Exception:
            pass

        control_node.stop_requested = True

        executor.shutdown()
        spin_thread.join(timeout=1.0)

        control_node.destroy_node()
        navigator.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()