"""
================================================================================
[GreenGuard 통합 관제 시스템 백엔드 엔진 - web.py]
================================================================================
- 목적: ROS 2 환경의 로봇 상태(배터리, 위치) 및 카메라 영상(CompressedImage)을 수신하고,
        AI 탐지 플래그(Bool)와 실시간 동기화하여 바운딩 박스가 포함된 스냅샷을 유실 없이 저장함과 동시에,
        웹 사용자 화면(Flask)에 스트리밍 서비스 및 REST API를 제공하는 관제 통합 엔진임.
- 연동 아키텍처: 멀티스레딩(Threading) 기반으로 'ROS 2 멀티스레드 스핀 루프'와 'Flask 웹 서버'가
                전역 버퍼(robot_data)를 공유하며 비동기로 상시 소통함.
================================================================================
"""

import os, cv2, time, json, threading, numpy as np
from pathlib import Path
from datetime import datetime
from cv_bridge import CvBridge
from green_guard.greenGuard_db import GreenGuardDB
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage, BatteryState
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator


# ------------------------------------------------------------------------------
# 1. 시스템 설정 환경 구성 및 전역 인프라 세팅
# ------------------------------------------------------------------------------

# 스크립트 파일의 절대 경로와 부모 디렉토리(src) 경로를 계산하여 파일 접근의 이식성 확보
current_file_path = Path(__file__).resolve()
src_dir_path = current_file_path.parent

# JSON 설정 파일을 읽어 템플릿 폴더, 스냅샷 저장 폴더, DB 경로 등의 메타데이터 로드
with open(src_dir_path / 'config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Flask 인스턴스 초기화: config에 정의된 상대 경로들을 절대 경로 스트링으로 변환하여 매핑
app = Flask(__name__, 
            template_folder=str(src_dir_path / config['template_folder']),
            static_folder=str(src_dir_path / config['static_folder']))
app.secret_key = os.getenv("GREENGUARD_SECRET_KEY", config.get("secret_key", ""))
if not app.secret_key or app.secret_key == "change-this-secret":
    raise RuntimeError("GREENGUARD_SECRET_KEY를 환경변수로 설정하세요.")

# SQLite 데이터베이스 핸들러 초기화 (컨텍스트 매니저 '__with__' 패턴 지원 객체)
db = GreenGuardDB(src_dir_path / config['db_path'])

# [영상 유실 방지 가드] 로봇 연결 단절 시 웹 화면에 송출할 640x480 기본 검은색 배경의 Placeholder 이미지 가공
no_signal_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(no_signal_img, 'No Signal', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
_, no_signal_bytes = cv2.imencode('.jpg', no_signal_img)
NO_SIGNAL_IMAGE = no_signal_bytes.tobytes() # 메모리 점유 최소화를 위해 최종 바이너리 바이트 배열로 보관

# 시스템 전체에서 사용할 채널 식별 키와 실제 ROS 2 로봇 네임스페이스/장치 키 매핑 테이블
ROBOT_CONFIG = {'cam1': 'pc_cam1', 'cam2': 'pc_cam2', 'robot1': 'robot3', 'robot2': 'robot8'}

# [전역 스레드 공유 버퍼] ROS 2가 데이터를 채우고, Flask가 읽어가는 데이터 교환 공간
robot_data = {
    name: {
        'frame': NO_SIGNAL_IMAGE,         # 웹 브라우저 화면 출력용 초경량 압축 JPG 바이너리
        'raw_snapshot': NO_SIGNAL_IMAGE,  # 탐지 이벤트 발생 시 원본 화질 저장을 위한 수신 OpenCV 행렬 버퍼
        'battery': 0                      # 로봇 잔여 배터리 잔량 정수값 (%)
    } for name in ROBOT_CONFIG
}

# Flask가 ROS 2 노드의 메서드(스냅샷 저장 기능 등)에 다이렉트 접근하기 위한 싱글톤 성격의 전역 포인터
visualizer_node = None


# ------------------------------------------------------------------------------
# 2. ROS 2 관제 데이터 수신 및 이벤트 처리 엔진 클래스
# ------------------------------------------------------------------------------
class WebVisualizerNode(Node):
    """
    WebVisualizerNode는 ROS 2 생태계와 통신하는 핵심 서브시스템 노드.
    여러 로봇의 토픽이 동시다발적으로 유입되므로, 병목현상을 차단하기 위해 
    비동기 다중 스레드 그룹인 ReentrantCallbackGroup을 채택하여 설계.
    """
    def __init__(self):
        # 상위 클래스 Node의 생성자를 호출하여 'web_visualizer_node'라는 이름으로 ROS 컨텍스트에 등록
        super().__init__('web_visualizer_node')
        
        # 상호 간섭 없이 멀티 스레딩으로 콜백들을 처리할 수 있게 하는 병렬 콜백 그룹 인스턴스 생성
        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge() # OpenCV <-> ROS Image 변환 툴 (현재 최적화로 직접 디코딩 처리 중)

        # 상태 관리 멤버 변수 초기화
        self.robot_poses = {n: {'x': 0.0, 'y': 0.0} for n in ['robot1', 'robot2']} # 터틀봇 로봇 실시간 SLAM 좌표
        self.prev_flag_states = {n: False for n in ROBOT_CONFIG} # Rising Edge(신호 상승 에지) 포착용 이전 플래그 버퍼
        self.current_flags = {n: False for n in ROBOT_CONFIG}    # AI 탐지 노드로부터 들어온 실시간 플래그 상태
        self.last_save_time = {n: 0.0 for n in ROBOT_CONFIG}     # 디바운싱(중복 스냅샷 저장 방지) 시스템용 타임스탬프
        self.save_cooldown = 0.5 # 동일 장치에 대한 연속 저장 제한 시간 (0.5초 디바운스 쿨타임)

        # 🌟 자동 출동 및 인터페이스 제어 제어 변수 선언
        self.battery_threshold = 100       # 💡 테스트용 타겟 기준값 (100 미만 시 발동)
        self.is_robot2_dispatched = False  # 중복 출동(언도킹 무한 요청) 차단 가드 0

        # 🌟 [추가] 중복 스레드 호출 및 액션 서버 과부하 방지용 전역 락 및 상태 변수
        self.undock_lock = threading.Lock()
        self.is_undocking_active = False   # 현재 실제 undock() API가 실행 중인지 여부

        # 🌟 네비게이터 초기화 시 자신(self) 노드와 2번 로봇의 네임스페이스(/robot8)를 정확히 주입
        self.get_logger().info("⏳ TurtleBot4Navigator 인프라 구성 중 (Target: /robot8)...")
        try:
            # 💡 namespace='robot8' 인자를 추가하여 내비게이터가 /robot8/undock 액션 서버를 바라보게 만듭니다.
            self.navigator = TurtleBot4Navigator(namespace='robot8')
            
            # 정식 오픈소스 멤버 변수인 undock_action_client를 사용해 부팅 시점 액션 서버 사전 검증
            if self.navigator.undock_action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().info("✅ [/robot8/undock] 터틀봇4 언도킹 액션 서버 커넥션 확인 완료.")
            else:
                self.get_logger().error("❌ [/robot8/undock] 터틀봇4 언도킹 액션 서버 응답 없음 (하드웨어 연결 점검 필요).")
        except Exception as e:
            self.get_logger().error(f"네비게이터 초기화 실패 (시뮬레이터/하드웨어 확인 필요): {e}")
            self.navigator = None

        # [추가] 3번 디버그 버튼 전용: 예약 시간 정보 발행용 Publisher 생성
        self.reserve_publisher = self.create_publisher(
            String,
            '/robot3/reserve',
            10
        )
        self.get_logger().info("✅ [/robot3/reserve] 예약 발신 토픽 퍼블리셔 등록 완료.")

        # 터틀봇 전용 토픽 구독 등록
        for name, robot_id in ROBOT_CONFIG.items():
            if name in ['robot1', 'robot2']:
                # 각 로봇의 고유 네임스페이스를 조합하여 이미지, 배터리, AMCL 위치 토픽 자동 구독 (람다 기본인자 기법으로 name 스코프 고정)
                self.create_subscription(
                    CompressedImage, 
                    f'/{robot_id}/oakd/rgb/image_raw/compressed/person', 
                    lambda m, n=name: self.robot_image_callback(m, n), 
                    10, 
                    callback_group=self.callback_group
                    )
                
                self.create_subscription(
                    BatteryState, 
                    f'/{robot_id}/battery_state', 
                    lambda m, n=name: self.robot_battery_callback(m, n), 
                    10, 
                    callback_group=self.callback_group
                    )
                
                self.create_subscription(
                    PoseWithCovarianceStamped, 
                    f'/{robot_id}/amcl_pose', 
                    lambda m, n=name: self.pose_callback(m, n), 
                    10, 
                    callback_group=self.callback_group
                    )

        # 웹캠 센서 정보 및 이상 상황 탐지 AI 알림 플래그 토픽 등록
        cam_topics = {'cam1': '/detection/tomato/rotten', 'cam2': '/detection/cctv/human'}
        flag_topics = {'cam1': '/detection/tomato/flag', 'cam2': '/detection/cctv/human/flag', 'robot1' : '/detection/robot3/flag', 'robot2': '/detection/robot8/flag'}

        for name, topic in cam_topics.items():
            self.create_subscription(
                CompressedImage, 
                topic, 
                lambda m, n=name: self.robot_image_callback(m, n), 
                10, 
                callback_group=self.callback_group
                )
            
        for name, topic in flag_topics.items():
            self.create_subscription(
                Bool, 
                topic, 
                lambda m, n=name: self.flag_callback(m, n), 
                10, 
                callback_group=self.callback_group
                )

        self.get_logger().info('🟢 영상 내 바운딩 박스 자동 실시간 추적 엔진이 최적화 활성화되었습니다.')
    
    def robot_image_callback(self, msg, name):
        """
        [핵심 함수] 모든 이미지 토픽(CompressedImage)이 유입되는 관문.
        플래그 콜백과 비동기로 따로 돌던 구조에서 탈피해, 영상이 들어온 '그 순간의 스레드 프레임 안'에서 플래그 상태를 즉시 비교 판정.
        """
        global robot_data
        try:
            # 바운딩 박스가 이미 그려진 대용량 압축 바이트를 연산 없이 Flask 버퍼에 통과
            compressed_bytes = msg.data.tobytes()
            robot_data[name]['frame'] = compressed_bytes
            
            # ROS 2 압축 이미지 바이트 배열을 메모리 내부에서 직접 꺼내 고속 Numpy 1차원 행렬로 캐스팅
            np_arr = np.frombuffer(msg.data, np.uint8)
            # OpenCV 이미지 디코딩: 이 연산을 통해 원본 바이트 배열 속 오버레이/바운딩 박스 픽셀이 최종 병합(렌더링)됨
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is None: return

            # 고화질 스냅샷 저장을 위해 최신 이미지 객체 프레임을 원본 그대로 전역 버퍼에 일차 보관
            robot_data[name]['raw_snapshot'] = cv_img

            # 현재 프레임에서 AI 플래그가 True이고, 이전 주기 플래그가 False였던 '상승 에지(최초 탐지 순간)'인지 검증
            if self.current_flags.get(name, False) and not self.prev_flag_states.get(name, False):
                current_time = time.time()
                # 0.5초 쿨타임 검사로 한 번의 이상 상황에 수십 장이 중복 저장되는 디스크 낭비 방지
                if (current_time - self.last_save_time[name]) > self.save_cooldown:
                    # 중요: 다음 프레임이 유입되어 버퍼를 더럽히기 전에 현재 박스가 선명히 그려진 이미지 객체를 통째로 복사(.copy())하여 완벽 격리
                    self.get_logger().info(f"📸 [동기화 캡처] 프레임 확보 완료: {name}")
                    
                    snapshot_to_save = cv_img.copy()
                    
                    # 파일 저장 및 데이터베이스 트랜잭션 함수 비동기 가동 (로봇 장비인 경우에만 위치 좌표 매핑 연동)
                    self.save_detection_event(snapshot_to_save, name, name if name in ['robot1', 'robot2'] else None)
                    self.last_save_time[name] = current_time # 최근 저장 타임스탬프 갱신

            # 현재 상태를 과거 상태 버퍼로 이관하여 다음 프레임 판단 기준으로 삼음
            self.prev_flag_states[name] = self.current_flags.get(name, False)

        except Exception as e:
            self.get_logger().error(f"[{name}] 영상 스트리밍 처리 예외: {e}")

    def flag_callback(self, msg, name):
        # AI 탐지 노드가 발행한 상시 플래그 상태를 수신하여 멤버 딕셔너리에 실시간 기록
        self.current_flags[name] = msg.data

    def save_detection_event(self, cv_img, source_name, pose_source=None):
        # 이상 검출 순간 확보된 이미지 객체(cv_img)를 디스크에 쓰고, DB에 위치 및 시간 정보를 영속화.
        try:
            now = datetime.now()
            # 파일명 충돌을 원천 차단하기 위해 마이크로초단위 시각 정보 뒤 자리를 잘라 고유 파일 네이밍 포맷 생성
            image_name = f"{source_name}_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            save_dir = src_dir_path / config['detection_folder']
            os.makedirs(save_dir, exist_ok=True) # 저장 대상 디렉토리가 없을 경우 자동 생성
            
            # 물리 파일 쓰기 성공 시 DB 적재 프로세스 진입
            if cv2.imwrite(str(save_dir / image_name), cv_img):
                # 로봇이 아닌 고정캠(cam)이거나 좌표 데이터가 유실(None)된 경우 가독성 있게 단 한 줄로 0.0 디폴트 처리
                pose = self.robot_poses.get(pose_source, {}) if pose_source else {}
                pos_x, pos_y = pose.get('x', 0.0) or 0.0, pose.get('y', 0.0) or 0.0

                # SQLite DB 컨텍스트 매니저를 호출하여 인서트 실행 및 풀링 자동 반환
                with db as active_db:
                    new_log_id = active_db.insert_detection_log(image_name, now.strftime('%Y-%m-%d %H:%M:%S'), pos_x, pos_y)
                self.get_logger().info(f"💾 [성공] ID: {new_log_id} | 좌표: X={round(pos_x,2)}, Y={round(pos_y,2)}")

        except Exception as e:
            self.get_logger().error(f"❌ [{source_name}] 스냅샷 영속화 실패: {e}")

    def robot_battery_callback(self, msg, name):
        # 로봇의 배터리 정보를 100% 기준으로 표준화하여 정수 버퍼에 보관
        global robot_data
        percentage = int(msg.percentage if msg.percentage > 1.0 else msg.percentage * 100) if msg.percentage >= 0.0 else 0
        # 데이터가 0.0~1.0 형태율(소수점)로 들어올 경우와 0~100 직결 형태로 들어올 경우를 모두 방어하는 동적 보정식
        robot_data[name]['battery'] = percentage

        # 🌟 [자동화 엔진] 1번 로봇 배터리가 기준값 미만이고 아직 2번 로봇이 미출동 상태인 경우
        if name == 'robot1' and percentage < self.battery_threshold and not self.is_robot2_dispatched:
            self.is_robot2_dispatched = True  
            self.get_logger().warn(f"🚨 [배터리 경고] robot1 배터리 {percentage}%! 2번 로봇 자동 언도킹을 시작합니다.")
            
            # 🌟 교정: 이미 메인 실행기(Executor)가 스핀 중이므로, 액션 블로킹을 피하기 위해 
            # rclpy 자체 스레드 안전 기능을 사용하거나 무겁지 않은 스레드로 분리 호출 가능
            threading.Thread(target=self.execute_robot2_undocking, daemon=True).start()
    
    def execute_robot2_undocking(self):
        if self.navigator is None:
            self.get_logger().error("❌ 네비게이터 가동 불가 상태입니다.")
            return

        if not self.undock_lock.acquire(blocking=False):
            self.get_logger().warn("⚠️ [Action] 이미 언도킹 연산 스레드가 수행 중입니다. 중복 요청을 무시합니다.")
            return

        try:
            is_docked = bool(self.navigator.getDockedStatus())
            self.get_logger().info(f"ℹ️ [Action] 2번 로봇 현재 최종 도킹 상태 판단 결과: {is_docked}")

            # 🌟 교정: 로봇이 이미 도크 밖에 있다고 착각하더라도 수동 버튼이나 자동 명령이 유입되면 
            # 안전하게 무조건 undock() 명령을 한 번 더 액션 서버로 밀어 넣도록 로그 유연화
            self.get_logger().info('🤖 [Action] 2번 로봇 언도킹 명령(navigator.undock())을 전달합니다.')
            
            # 올바른 네임스페이스(/robot8/undock)를 잡았으므로 명령이 정상 전달됩니다.
            self.navigator.undock()
            
            self.get_logger().info('✅ [Action] 2번 로봇 언도킹 프로세스가 완수되었습니다.')

        except Exception as e:
            self.get_logger().error(f"❌ 언도킹 프로세스 실행 실패: {e}")
            self.is_robot2_dispatched = False
            
        finally:
            self.undock_lock.release()

    def pose_callback(self, msg, name):
        # AMCL 오도메트리 추정 위치 노드 데이터를 파싱하여 로봇별 슬롯에 갱신
        try:
            self.robot_poses[name] = {'x': msg.pose.pose.position.x, 'y': msg.pose.pose.position.y}

        except Exception as e:
            self.get_logger().error(f"[{name}] AMCL 좌표 갱신 실패: {e}")

# ------------------------------------------------------------------------------
# 3. 비동기 백그라운드 구동 스레드 및 스트리밍 발전기 함수
# ------------------------------------------------------------------------------
def ros2_thread_loop():
    global visualizer_node
    rclpy.init()
    
    visualizer_node = WebVisualizerNode()
    executor = MultiThreadedExecutor() 
    
    # 1. 메인 웹 관제 비주얼라이저 노드 등록
    executor.add_node(visualizer_node)
    
    # 2. 🌟 Pylance 우회 및 데드락 해제 통합 가드
    # 네비게이터 내부의 노드 핸들러 객체를 정적 분석기(Pylance) 에러 없이 안전하게 추출합니다.
    if visualizer_node.navigator:
        # 오픈소스 버전에 따라 다를 수 있는 내부 노드 속성명을 순차적으로 탐색 (node -> node_handle)
        internal_node = None
        for attr_name in ['node', 'node_handle', '_node']:
            if hasattr(visualizer_node.navigator, attr_name):
                internal_node = getattr(visualizer_node.navigator, attr_name)
                break
        
        # 내부 노드가 성공적으로 확보되었고 유효한 rclpy Node 객체라면 Executor에 탑승
        if internal_node is not None:
            executor.add_node(internal_node)
            visualizer_node.get_logger().info("🔗 [System] 네비게이터 내부 ROS2 노드를 MultiThreadedExecutor에 안전하게 결합했습니다.")
        else:
            visualizer_node.get_logger().error("⚠️ [System] 네비게이터 내부 노드 객체를 참조할 수 없어 액션 데드락 위험이 있습니다.")

    try: 
        executor.spin() 
    except Exception as e: 
        print(f"ROS 2 스핀 종료: {e}")
    finally:
        executor.shutdown()
        visualizer_node.destroy_node()
        rclpy.shutdown()

def gen_ros_frames(name):
    # MJPEG 스트리밍 표준 규격에 맞추어 Flask의 Response 객체에 끊임없이 프레임 바이너리를 양보(yield)하는 제너레이터
    global robot_data
    while True:
        # 공유 전역 버퍼에서 타겟 장비의 압축 JPG 바이트열을 추출, 데이터가 일시 단절되면 NO_SIGNAL 이미지 바인딩
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + robot_data.get(name, {}).get('frame', NO_SIGNAL_IMAGE) + b'\r\n')
        time.sleep(0.03) # 초당 대략 33프레임 수준으로 조율하여 관제용 Flask 내부 소켓 버퍼 과부하 제어


# ==============================================================================
# 4. 🌐 FLASK 웹 통신 라우터 및 데이터 제어 웹 서비스 컨트롤러 영역
# ==============================================================================

@app.route("/")
def home():
    # 루트 경로 접근 시 유저 세션 로그인 유무를 대조하여 메인 관제 페이지 또는 로그인 홈으로 동적 리다이렉트
    return redirect(url_for('main_page') if 'username' in session else url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # 사용자 인증 시스템 컨트롤러: POST 요청 시 DB 암호 대조 절차 수행
    if request.method == 'POST':
        with db as active_db:
            if active_db.verify_user(request.form['username'], request.form['password']):
                session['username'] = request.form['username'] # 세션 쿠키에 사용자 자격 증명 식별키 주입
                flash('로그인 하셨습니다.', 'success')
                return redirect(url_for('main_page'))
        flash('아이디 또는 비밀번호가 틀렸습니다.', 'danger')
    return render_template('login_center.html')

@app.route('/main_page')
def main_page():
    # 메인 종합 대시보드 뷰 라우터: DB 내부의 전체 히스토리 탐지 로그 목록을 함께 뷰 템플릿 엔진에 바인딩하여 렌더링
    if 'username' not in session:
        flash('로그인을 해주십시오.', 'warning')
        return redirect(url_for('login'))
    with db as active_db:
        return render_template('main_page.html', username=session['username'], logs=active_db.get_all_logs())

@app.route('/video_feed')
def video_feed():
    # [MJPEG 실시간 스트리밍 엔드포인트]
    if 'username' not in session: return "Unauthorized", 401
    target_cam = request.args.get('cam_id', 'cam1')
    return Response(gen_ros_frames(target_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/robot_status')
def get_robot_status():
    # 웹 프론트엔드가 실시간으로 화면을 갱신(배터리 바, 로봇 2D 맵 좌표 컴포넌트 등)할 수 있도록 JSON 데이터를 반환하는 REST API
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({
        'robot1_battery': robot_data['robot1']['battery'], 'robot1_amcl_pose': visualizer_node.robot_poses['robot1'] if visualizer_node else {'x': 0.0, 'y': 0.0},
        'robot2_battery': robot_data['robot2']['battery'], 'robot2_amcl_pose': visualizer_node.robot_poses['robot2'] if visualizer_node else {'x': 0.0, 'y': 0.0}
    })

@app.route('/api/logs')
def get_logs_realtime():
    # 실시간으로 적재되는 신규 알림 로그를 풀링/웹소켓 대체용으로 반환하며, 테이블 크기가 100개가 넘어가면 노후화된 물리 파일과 테이블 로우를 자동 청소
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        with db as active_db:
            active_db.delete_logs_by_count(max_count=100, img_dir_path=str(src_dir_path / config['detection_folder']))
            return jsonify(active_db.get_all_logs())
    except Exception as e:
        return jsonify({'status' : 'error', 'message': str(e)}), 500

@app.route('/api/logs/search')
def search_logs_by_date():
    # 대시보드 캘린더 검색 시 연동되는 조회용 API: 특정 날짜 문자열 조건에 일치하는 로그 리스트 필터링 반환
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    if not request.args.get('date', ''): return jsonify({'error': 'Date parameter is missing'}), 400
    try:
        with db as active_db:
            return jsonify(active_db.get_logs_by_date(request.args.get('date')))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 🌟 [웹 UI 연동 추가 API] 관제자가 프론트엔드 화면에서 원격으로 직접 2번 로봇을 출동시킬 때 실행됨
@app.route('/api/debug/robot2/force_undock', methods=['POST'])
def force_undock():
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    if visualizer_node:
        # 수동 출동 역시 영상 멈춤 방지를 위해 백그라운드 스레드 가동
        threading.Thread(target=visualizer_node.execute_robot2_undocking, daemon=True).start()
        return jsonify({'status': 'success', 'message': '2번 로봇 강제 언도킹 시퀀스를 무사히 시작했습니다.'})
        
    return jsonify({'status': 'fail', 'message': 'ROS2 관제 엔진이 아직 기동되지 않았습니다.'}), 500

@app.route('/api/debug/debug_capture', methods=['POST'])
def debug_capture():
    """ 
    [수동 디버깅 전용 엔드포인트]
    웹 UI에서 '수동 스냅샷 촬영' 버추얼 버튼을 누르면 기동하며, 
    현재 살아있는 임의의 채널 버퍼 데이터를 낚아채 즉시 수동 디렉토리 파일 쓰기 및 로그 이벤트를 트리거.
    """
    if 'username' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    active_img, active_robot_name = None, None
    # 등록된 전체 채널 버퍼를 순회하며 유효한 OpenCV 데이터 행렬 객체가 존재하는 슬롯 탐색
    for name in ROBOT_CONFIG:
        temp = robot_data.get(name, {}).get('raw_snapshot', NO_SIGNAL_IMAGE)
        if isinstance(temp, np.ndarray):
            active_img, active_robot_name = temp, name
            break
            
    if active_img is None: return jsonify({'status': 'fail', 'message': '연결된 로봇 영상 신호가 없습니다.'}), 400
    
    if visualizer_node:
        try:
            # 수동 강제 캡처 동작 처리 유도 (타겟 장치명 전달 및 좌표 결합 연산 수행)
            visualizer_node.save_detection_event(active_img, active_robot_name, active_robot_name if active_robot_name in ['robot1', 'robot2'] else None)
            return jsonify({'status': 'success', 'message': f'{active_robot_name} 장비 스냅샷 수동 디버그 저장 성공!'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'fail', 'message': 'ROS2 노드가 준비되지 않았습니다.'}), 500

# ------------------------------------------------------------------------------
# 🌟 [최종 교정] 웹(Front)에서 변환해서 보낸 'HH:MM' 문자열을 그대로 토픽으로 쏘는 라우터
# ------------------------------------------------------------------------------
@app.route('/api/debug/reserve_topic', methods=['POST'])
def api_reserve_topic():
    global visualizer_node
    
    # 1. Pylance Optional 방어선: 싱글톤 노드 유효성 검사 수행
    if visualizer_node is None:
        return jsonify({'status': 'error', 'message': 'ROS2 관제 노드가 아직 활성화되지 않았습니다.'}), 500
        
    data = request.get_json()
    # 🌟 웹프론트에서 완성해서 보낸 'HH:MM' 포맷 문자열을 그대로 수신
    interval_text = data.get('interval', '').strip() 
    
    # 예외 가드: 데이터가 비어있거나 올바른 규격(예: 00:00)이 아니면 차단
    if not interval_text or interval_text == "00:00":
        return jsonify({'status': 'error', 'message': '올바른 반복 주기 간격을 입력해주세요.'}), 400
    
    try:
        # 2. Pylance 안전 참조: 로거 및 퍼블리셔 추출
        logger = visualizer_node.get_logger()
        publisher = visualizer_node.reserve_publisher
        
        msg = String()
        msg.data = interval_text # 웹이 준 문자열 그대로 주입!
        
        # 3. /robot3/reserve 토픽으로 메시지 즉시 발행
        publisher.publish(msg)
        logger.info(f"🚀 [웹 관제 명령] 프론트엔드가 생성한 주기 토픽 전송 완료: {msg.data}")
        
        return jsonify({
            'status': 'success', 
            'message': f"웹에서 설정한 주기 [{interval_text}] 정보가 로봇 제어 토픽으로 즉시 송신되었습니다."
        })
        
    except Exception as e:
        if visualizer_node:
            visualizer_node.get_logger().error(f"❌ 토픽 발행 중 런타임 예외 발생: {e}")
        return jsonify({'status': 'error', 'message': f"토픽 발신 중 예외가 발생했습니다: {e}"}), 500


# ------------------------------------------------------------------------------
# 5. 프로그램 메인 프로세스 진입점 (Entry Point)
# ------------------------------------------------------------------------------
def main(args=None):
    # 어플리케이션 부팅 마스터 함수: ROS 2 전용 데몬 스레드를 먼저 가동한 후 Flask 웹 서비스를 메인 루프에 점화.
    # 파이썬 메인 스레드가 완전히 종료될 때 백그라운드 ROS 소켓 스레드도 동시 강제 처분되도록 daemon=True 명시 설정
    threading.Thread(target=ros2_thread_loop, daemon=True).start()
    
    # 웹 App 구동 포트 5000 할당 및 코드 변경 시 자동 리로더가 스레드를 중복 포크하여 ROS 2 컨텍스트를 파괴하는 버그를 막기 위해 use_reloader=False 설정
    app.run(debug=True, port=5000, use_reloader=False)

if __name__ == "__main__":
    main()