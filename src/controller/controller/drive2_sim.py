#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Accel
from std_msgs.msg import Float32, String
from ament_index_python.packages import get_package_share_directory
import pandas as pd
import numpy as np
import time
import csv
import os
import math
from datetime import datetime

# ==============================================================================
# [1. 유틸리티 클래스] - SimpleKalman
# 센서 데이터(Pose, Yaw)의 노이즈를 제거하기 위한 1차 저주파 통과 필터 기반 칼만 필터
# ==============================================================================
class AdvancedKalman:
    def __init__(self, q=0.1, r=0.1):
        self.q, self.r = q, r
        self.x, self.p = None, 1.0

    def step(self, measurement, prediction_offset=0.0, gate=None):
        if self.x is None:
            self.x = measurement
            return self.x

        # [변경] 현재 위치에 이동 예측량(dx, dy)을 더함
        x_prior = self.x + prediction_offset
        p_prior = self.p + self.q

        # [변경] Gate 로직: 센서가 예측치보다 너무 멀리 튀면 센서 무시
        if gate is not None and abs(measurement - x_prior) > gate:
            self.x = x_prior
            self.p = p_prior
            return self.x

        # 보정 및 업데이트
        k_gain = p_prior / (p_prior + self.r)
        self.x = x_prior + k_gain * (measurement - x_prior)
        self.p = (1 - k_gain) * p_prior
        return self.x

# ==============================================================================
# [2. 통합 주행 노드] - UnifiedFollower (실전 및 시뮬레이션 공용)
# 실차 플랫폼 제어 알고리즘과 상세 데이터 로깅 시스템을 통합 수행
# ==============================================================================
class UnifiedFollower(Node):
    def __init__(self):
        super().__init__('unified_follower')
        
        # ----------------------------------------------------------------------
        # [A] 하드코딩 파라미터 및 경로 설정 존
        # ----------------------------------------------------------------------
        self.car_id = 2  # 차량 고유 번호 (Remapping 가능)
        self.use_prediction = True  # True: 예측 모드, False: 1D in kalman filter
        
        
        
        # 경로 파일(CSV) 로드 설정 (홈 디렉토리 기준 절대 경로 구성)
        home_dir = os.path.expanduser('~')
        self.path_file = os.path.join(home_dir, 'kmc_ws/src/controller/path/path2.csv')

        # 제어 알고리즘 핵심 파라미터 (사용자 요청에 따른 최적화 및 세분화)
        self.params = {
            # 1. 조향 PID 제어 (Crosstrack Error 보정용)
            "p_kp": 3.0,
            "p_ki": 1.5,
            "p_kd": 3.0,
            "p_steer_deadzone": 0.005,  # 연속형 데드존 (m) - 직선 진동 억제
            "p_yaw_deadzone": 3.0,      # [추가] 방향 오차 데드존 (degree)

            # 2. 피드포워드(FF) 및 방향(Yaw) 보정
            "p_ff_gain": 2.0,      # 경로 곡률 기반 선제적 조향 게인
            "p_ff_window": 10,     # 곡률 계산용 전방 윈도우 사이즈
            "p_kyaw": 1.0,         # 차량-경로 간 방향 오차 보정 게인

            # 3. 속도 프로파일 및 가감속 제약
            "p_v_max": 1.0,        # 목표 선속도 상한 (m/s)
            "p_v_min": 0.5,        # 최저 주행 속도 (m/s)
            "p_v_accel": 1.0,      # 최대 가속도 제약 (m/s^2) - 슬립 방지
            "p_v_decel": 2.0,      # 최대 감속도 제약 (m/s^2) - 급제동 허용
            
            # 4. 동적 속도 페널티 계수 (주행 상황별 속도 저감)
            "p_v_curve_gain": 0.3, # 급커브 시 속도 저감 비중
            "p_v_cte_gain": 0.1,   # 경로 이탈 시 속도 저감 비중
            
            # 5. 칼만 필터 게인 세분화 (X, Y 위치 vs Yaw 방향 분리)
            "p_kf_q_pose": 0.1,    # 위치 프로세스 노이즈
            "p_kf_r_pose": 0.1,    # 위치 측정 노이즈
            "p_kf_q_yaw": 0.2,     # Yaw 프로세스 노이즈
            "p_kf_r_yaw": 0.01     # Yaw 측정 노이즈
        }
        
        # UnifiedFollower.__init__ 내부 적당한 위치에 추가
        self.infra_target_v = self.params['p_v_max'] # 초기값은 차량의 최대 속도로 설정
        # 인프라 매니저가 발행하는 가이드 속도 수신
        self.sub_infra_v = self.create_subscription(
            Float32,
            f'/CAV_{self.car_id:02d}_target_v', # 인프라 코드의 토픽명과 매칭 (CAV_01_target_v 등)
            self.infra_v_callback,
            10
        )
        
        # 로그 데이터 저장 디렉토리 설정
        self.log_dir = os.path.join(home_dir, 'kmc_ws/src/controller/logs/sim/')

        # ----------------------------------------------------------------------
        # [B] 차량 상태 변수 및 통계 메모리 초기화
        # ----------------------------------------------------------------------
        self.current_v = 0.0              # 현재 계산된 목표 선속도
        self.filtered_pose = [0.0, 0.0, 0.0]  # [filt_x, filt_y, filt_yaw]
        self.prev_ni = None               # 이전 루프 최인접 인덱스
        
        self.error_integral = 0.0         # PID 적분항
        self.last_error = 0.0             # PID 미분항용 이전 오차
        self.last_time = self.get_clock().now()
        self.start_time = time.time()
        self.is_finished = False          # 주행 종료 플래그
        self.finish_check_time = None     # 완주 시점 기록
        
        # 하드웨어 피드백 데이터
        self.actual_v = 0.0               # 실측 속도
        self.battery_voltage = 0.0        # 배터리 전압
        self.echo_v = 0.0                 # 드라이버 수신 확인 속도
        self.echo_w = 0.0                 # 드라이버 수신 확인 각속도
        self.raw_allstate = ""            # 전체 상태 문자열 (보험용)
        
        # 주행 방향 및 지연 보상 예측 변수
        self.prev_filt_px = None
        self.prev_filt_py = None
        self.current_motion_yaw = 0.0      # 이동 벡터 기반 방향
        self.last_valid_motion_yaw = 0.0
        self.last_path_yaw = 0.0
        self.last_omega = 0.0             # 이전 각속도 명령
        self.last_diff = 0.0              # 각속도 변화량 (Flip 감지)
        
        self.lap_count = 0                # 주행 바퀴 수
        self.halfway_passed = False       # 반환점 통과 여부
        self.flip_history = []            # 조향 진동 기록
        self.last_pose_time = self.get_clock().now() # 초기값 설정
        self.v_buffer = [0.0] * 10  # MA10 버퍼

        # 센서 원본 기록 변수
        self.raw_px = 0.0
        self.raw_py = 0.0
        self.raw_yaw = 0.0

        # ----------------------------------------------------------------------
        # [C] 전역 경로(Global Path) 데이터 로딩
        # ----------------------------------------------------------------------
        try:
            if not os.path.exists(self.path_file):
                raise FileNotFoundError(f"경로 파일 부재: {self.path_file}")
            df = pd.read_csv(self.path_file, header=None)
            self.path = df.apply(pd.to_numeric, errors='coerce').dropna().values
            self.get_logger().info(f"✅ 경로 데이터 로드 성공: {len(self.path)} pts")
        except Exception as e:
            self.get_logger().error(f"❌ 경로 로드 에러: {e}")
            self.path = np.array([[0,0], [1,0]]) 

        # ----------------------------------------------------------------------
        # [D] 고성능 데이터 로깅 시스템 (총 42개 컬럼)
        # ----------------------------------------------------------------------
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 경로 파일의 전체 경로에서 파일 이름만 쏙 뽑아내기
        path_name = os.path.splitext(os.path.basename(self.path_file))[0]
        env = "sim"
        self.csv_filename = f"{self.log_dir}/log_{path_name}_{env}_{timestamp}.csv"
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 상세 로그 헤더 (분석 효율을 위한 체계적 분류)
        self.log_headers = [
            'time', 'ni', 'lap_count', 'dt',                    # [1-4] 기본 정보
            'raw_px', 'raw_py', 'raw_yaw',                      # [5-7] 센서 원본
            'filt_px', 'filt_py', 'filt_yaw',                   # [8-10] 필터 결과 (추가됨)
            'motion_yaw', 'path_yaw',                           # [11-12] 방향 분석
            'cmd_v', 'cmd_w', 'echo_v', 'echo_w',               # [13-16] 명령 및 응답
            'actual_v', 'battery', 'is_flip',                   # [17-19] 실측 피드백
            'curvature', 'cte', 'omega_pid', 'omega_ff', 'omega_yaw', # [20-24] 제어 성분
            'p_kp', 'p_ki', 'p_kd', 'p_steer_deadzone',         # [25-28] PID 파라미터
            'p_ff_gain', 'p_ff_window', 'p_kyaw',               # [29-31] FF/Yaw 파라미터
            'p_v_max', 'p_v_min', 'p_v_accel', 'p_v_decel',     # [32-35] 속도 파라미터
            'p_v_curve_gain', 'p_v_cte_gain',                   # [36-37] 페널티 파라미터
            'p_kf_q_pose', 'p_kf_r_pose', 'p_kf_q_yaw', 'p_kf_r_yaw', # [38-41] 필터 게인 (세분화)
            'p_yaw_deadzone', 
            'kf_mode',  # <--- 마지막 쯤에 추가 (1: 예측, 0: 1D)
            'raw_allstate'                                      # [42] 하드웨어 전문
        ]
        self.csv_writer.writerow(self.log_headers)

        # ----------------------------------------------------------------------
        # [E] 필터 초기화 및 통신 환경 구축
        # ----------------------------------------------------------------------
        # 세분화된 파라미터를 적용한 칼만 필터 인스턴스 생성
        self.kf_x = AdvancedKalman(self.params['p_kf_q_pose'], self.params['p_kf_r_pose'])
        self.kf_y = AdvancedKalman(self.params['p_kf_q_pose'], self.params['p_kf_r_pose'])
        self.kf_yaw = AdvancedKalman(self.params['p_kf_q_yaw'], self.params['p_kf_r_yaw'])

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        # 퍼블리셔: 제어 명령 accel 발행
        self.pub_accel = self.create_publisher(Accel, f'/CAV_0{self.car_id}_accel', 10)
        
        # 서브스크라이버: 위치, 속도, 전압, 에코, 전체 상태 수신
        self.sub_pose = self.create_subscription(
            PoseStamped, 
            f'/CAV_0{self.car_id}', 
            self.pose_callback, 
            qos
        )
        self.sub_actual_v = self.create_subscription(Float32, 'vehicle_speed', self.actual_v_callback, 10)
        self.sub_battery = self.create_subscription(Float32, 'battery_voltage', self.battery_callback, 10)
        self.sub_echo = self.create_subscription(Twist, 'cmd_echo', self.echo_callback, 10)
        self.sub_allstate = self.create_subscription(String, 'allstate_text', self.allstate_callback, 10)
        
        # 제어 주기 타이머: 20Hz (0.05s)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.curr_pose = None

    # --------------------------------------------------------------------------
    # [콜백 함수] - 실시간 데이터 수신 및 전처리
    # --------------------------------------------------------------------------
    def pose_callback(self, msg):
        # 1. 센서 간의 실제 시간 간격(pose_dt) 계산
        now = self.get_clock().now()
        pose_dt = (now - self.last_pose_time).nanoseconds / 1e9
        self.last_pose_time = now
        
        if pose_dt <= 0 or pose_dt > 0.2: pose_dt = 0.05
            
        raw_px, raw_py = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        raw_yaw_val = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        
        self.raw_px, self.raw_py, self.raw_yaw = raw_px, raw_py, raw_yaw_val

        # 위상 도약 방지 (Yaw 센서는 로그용으로만 필터링)
        if self.kf_yaw.x is not None:
            while raw_yaw_val - self.kf_yaw.x > np.pi: raw_yaw_val -= 2*np.pi
            while raw_yaw_val - self.kf_yaw.x < -np.pi: raw_yaw_val += 2*np.pi
        
        # ==========================================================
        # [핵심 수정] 2번 모델: Motion Yaw + Steering Command 예측
        # ==========================================================
        # 1. 현재 진행 방향(motion_yaw)에 내가 꺾은 각속도(omega)만큼의 변화량을 더함
        # pose_dt 동안 차가 회전할 각도를 계산해서 방향을 미리 틉니다.
        predicted_yaw = self.current_motion_yaw + (self.last_omega * pose_dt)
        
        # [수정] 플래그에 따른 칼만 필터 예측 로직 분기
        if self.use_prediction:
            # V2 예측 모드: 물리 모델(v, omega) 반영
            predicted_yaw = self.current_motion_yaw + (self.last_omega * pose_dt)
            dx = self.current_v * np.cos(predicted_yaw) * pose_dt
            dy = self.current_v * np.sin(predicted_yaw) * pose_dt
            
        else:
            # 1D 모드: 예측 없이 센서 스무딩 위주
            dx = 0.0
            dy = 0.0
            
        dynamic_gate = abs(self.current_v * pose_dt) + 0.2  # 넉넉한 게이트

        # 필터 업데이트 (dx, dy, gate가 모드에 따라 자동 적용됨)
        self.filtered_pose = [
            self.kf_x.step(raw_px, dx, gate=dynamic_gate),
            self.kf_y.step(raw_py, dy, gate=dynamic_gate),
            self.kf_yaw.step(raw_yaw_val) 
        ]
        self.curr_pose = msg
        
        # 클래스 내부 메서드로 추가
    def infra_v_callback(self, msg):
        # 인프라가 허용하는 최대 속도를 업데이트합니다.
        self.infra_target_v = msg.data
        
    def actual_v_callback(self, msg):
        self.actual_v = msg.data
        
    def echo_callback(self, msg):
        self.echo_v = msg.linear.x
        self.echo_w = msg.angular.z

    def allstate_callback(self, msg):
        self.raw_allstate = msg.data
    
    def battery_callback(self, msg):
        self.battery_voltage = msg.data

    # --------------------------------------------------------------------------
    # [제어 유틸리티] - 경로 추적 및 곡률 분석
    # --------------------------------------------------------------------------
    def find_nearest_global(self, px, py):
        """ 로컬 윈도우 기반 최인접 포인트 탐색 (5초 카운트 방지 포함) """
        path_len = len(self.path)
        # 점 간격 1cm이므로 100개면 전후 1m씩, 총 2m 범위를 봅니다.
        # 이 범위 안에서만 찾으면 센서가 8m를 튀어도 인덱스는 제자리를 지킵니다.
        window_size = 100 
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # [A] 탐색 범위 결정
        # 시작 후 5초 동안은 위치를 확실히 잡기 위해 전체에서 찾습니다.
        if self.prev_ni is None or elapsed_time < 5.0:
            indices = np.arange(path_len)
        else:
            # 주행 중에는 이전 인덱스 근처(window_size)만 탐색합니다.
            indices = np.arange(self.prev_ni - window_size, self.prev_ni + window_size)
            indices = indices % path_len

        # 결정된 범위 내에서 최단 거리 인덱스 추출
        search_path = self.path[indices]
        dists = np.sqrt(np.sum((search_path - [px, py])**2, axis=1))
        ni = indices[np.argmin(dists)]

        # [B] Lap 카운팅 로직 (수정: 시작 후 5초가 지나야만 카운트 시작)
        if ni > path_len * 0.5: 
            self.halfway_passed = True
            
        # 5초가 경과했고, 반환점을 돌았을 때만 결승선 통과를 인정합니다.
        if elapsed_time > 5.0 and self.halfway_passed and self.prev_ni is not None:
            if self.prev_ni > path_len * 0.9 and ni < path_len * 0.1:
                self.lap_count += 1
                self.halfway_passed = False
                self.get_logger().info(f"🚩 Lap 카운트: {self.lap_count}")
                
                if self.lap_count >= 5 and self.finish_check_time is None:
                    self.get_logger().info(f'🏁 {self.lap_count}바퀴 완주 성공! 0.5초 후 종료합니다.')
                    self.finish_check_time = time.time()
        
        self.prev_ni = ni
        return ni
    

    def get_control_metrics(self, px, py, ni):
        """ 국부 경로의 주성분 분석(PCA)을 통해 진행 방향과 CTE 산출 """
        path_len = len(self.path)
        indices = [(ni + i) % path_len for i in range(-5, 6)]
        pts = self.path[indices]
        
        center = np.mean(pts, axis=0)
        norm_pts = pts - center
        cov = np.dot(norm_pts.T, norm_pts)
        val, vec = np.linalg.eigh(cov)
        tangent = vec[:, np.argmax(val)]
        
        path_yaw = np.arctan2(tangent[1], tangent[0])
        next_idx = (ni + 1) % path_len
        if np.dot(tangent, self.path[next_idx] - self.path[ni]) < 0:
            path_yaw += np.pi
                
        # 횡방향 이탈 오차(Crosstrack Error) 계산
        dx, dy = px - self.path[ni][0], py - self.path[ni][1]
        cte = -np.sin(path_yaw)*dx + np.cos(path_yaw)*dy
        return path_yaw, cte

    def get_curvature(self, ni, window):
        """ 전방 데이터 윈도우 기반 경로 곡률 계산 """
        path_len = len(self.path)
        p1 = self.path[ni]
        p2 = self.path[(ni + window // 2) % path_len]
        p3 = self.path[(ni + window) % path_len]
        
        v1, v2 = p2 - p1, p3 - p2
        ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        
        dist = np.linalg.norm(p3 - p1)
        if dist < 0.01: return 0.0
        return ang / dist

    # --------------------------------------------------------------------------
    # [메인 제어 루프] - 20Hz 알고리즘 실행 및 로깅
    # --------------------------------------------------------------------------
    def control_loop(self):
        if self.curr_pose is None or self.is_finished: return

        # 주행 종료 조건 체크
        if self.finish_check_time and (time.time() - self.finish_check_time > 0.5):
            self.close_node(); return

        # 1. 샘플링 타임(dt) 계산
        now = self.get_clock().now()
        dt = max(0.001, (now - self.last_time).nanoseconds / 1e9)
        self.last_time = now

        # 2. 필터링된 좌표 확보
        filt_px, filt_py, filt_yaw = self.filtered_pose
        
        # [연결] 현재 위치에서 가장 가까운 경로의 방향을 저장
        # 이 값이 다음 pose_callback의 '예측 힌트'로 쓰입니다.
        ni_temp = self.find_nearest_global(filt_px, filt_py)
        path_yaw, _ = self.get_control_metrics(filt_px, filt_py, ni_temp)
        self.last_path_yaw = path_yaw
        
        # 3. 이동 벡터 기반 차량 방향(Motion Yaw) 추정 - 센서 데이터 대체재
        temp_ni = self.find_nearest_global(filt_px, filt_py)
        temp_path_yaw, _ = self.get_control_metrics(filt_px, filt_py, temp_ni)
        self.last_path_yaw = temp_path_yaw
        
        if self.prev_filt_px is not None:
            dx, dy = filt_px - self.prev_filt_px, filt_py - self.prev_filt_py
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.02:  # 최소 이동 거리 2cm
                self.current_motion_yaw = np.arctan2(dy, dx)
                self.last_valid_motion_yaw = self.current_motion_yaw
            else:
                self.current_motion_yaw = self.last_path_yaw
        else:
            self.current_motion_yaw = self.last_path_yaw

        # 4. Latency 보상 예측 (미래 위치 기반 제어)
        pred_px = filt_px + (self.current_v * np.cos(self.current_motion_yaw) * dt)
        pred_py = filt_py + (self.current_v * np.sin(self.current_motion_yaw) * dt)

        # 5. 제어 지표 산출
        ni = self.find_nearest_global(pred_px, pred_py)
        path_yaw, cte = self.get_control_metrics(pred_px, pred_py, ni)
        curv_ff = self.get_curvature(ni, int(self.params['p_ff_window']))
        
        # 6. 속도 제어 로직 (수정: 조향 데드존과 통합 적용)
        # 조향에서 쓰는 p_steer_deadzone(5mm)을 속도 페널티에도 똑같이 적용합니다.
        v_dead = self.params['p_steer_deadzone']
        
        # 데드존 이내면 0, 넘어가면 그 차이만큼만 오차로 인정 (연속형)
        e_v_cte = 0.0 if abs(cte) < v_dead else abs(cte) - v_dead
        
        # 이제 5mm 이내의 미세한 흔들림에는 속도를 깎지 않습니다.
        v_penalty = (abs(curv_ff) * self.params['p_v_curve_gain']) + (e_v_cte * self.params['p_v_cte_gain'])
        
        # 1. 내 로직상 안전한 타겟 속도 계산
        local_target_v = np.clip(self.params['p_v_max'] - v_penalty, self.params['p_v_min'], self.params['p_v_max'])

        # 2. [수정] 내 속도와 인프라 권장 속도 중 더 낮은(안전한) 값을 선택
        # 인프라가 0.2로 줄이라고 하면, 내 로직이 0.8이라도 0.2를 따르게 됩니다.
        target_v = min(local_target_v, self.infra_target_v)
        
        # [수정] 이동평균(MA10) 모사: 버퍼에 넣고 평균내기
        self.v_buffer.pop(0)
        self.v_buffer.append(target_v)
        v_smoothed = sum(self.v_buffer) / 10.0
        
        # [수정] 기존 Slew-rate Limit을 v_smoothed에 적용하여 current_v 업데이트
        accel_limit = self.params['p_v_accel'] * dt if v_smoothed > self.current_v else self.params['p_v_decel'] * dt
        self.current_v = np.clip(v_smoothed, self.current_v - accel_limit, self.current_v + accel_limit)
        # 7. 통합 조향 제어 (PID + FF + Yaw Correction)
        
        # PID: 연속형 데드존 적용
        deadzone = self.params['p_steer_deadzone']
        e_dead = 0.0 if abs(cte) < deadzone else cte - (np.sign(cte) * deadzone)
        
        self.error_integral = np.clip(self.error_integral + e_dead * dt, -1.0, 1.0)
        cte_d = (e_dead - self.last_error) / dt
        omega_pid = -((self.params['p_kp'] * e_dead) + (self.params['p_ki'] * self.error_integral) + (self.params['p_kd'] * cte_d))
        self.last_error = e_dead

        # Feed Forward: 경로 곡률 비례 조향
        omega_ff = self.current_v * curv_ff * self.params['p_ff_gain']

        # Yaw 보정 (데드존 적용 버전)
        yaw_err = self.current_motion_yaw - path_yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        
        # degree를 radian으로 변환하여 데드존 계산
        y_dead = np.radians(self.params.get('p_yaw_deadzone', 3.0))
        
        # 연속형 데드존: 데드존 이내면 0, 넘어가면 그 차이만큼만 보정 (급격한 변화 방지)
        yaw_err_filtered = 0.0 if abs(yaw_err) < y_dead else yaw_err - (np.sign(yaw_err) * y_dead)
        
        yaw_gate = 1.0 / (1.0 + abs(curv_ff) * 10.0) 
        omega_yaw = -self.params['p_kyaw'] * yaw_err_filtered * yaw_gate

        # [사용자 요청 반영] 곡률 한계 3.0 기반 동적 각속도 제한 적용 (omega_limit = v * 3.0)
        omega_limit = abs(self.current_v) * 3.0
        final_omega = np.clip(omega_pid + omega_ff + omega_yaw, -omega_limit, omega_limit)

        # 8. 제어 명령 Twist 발행
        # 기존 msg = Twist() 로직 전체를 아래로 교체
        msg = Accel()
        msg.linear.x = float(self.current_v)
        msg.angular.z = float(final_omega)
        self.pub_accel.publish(msg)
        
        # 9. 실시간 데이터 로깅 (총 42개 컬럼 정확히 매칭)
        diff = final_omega - self.last_omega
        is_flip = 1 if (diff * self.last_diff) < 0 and abs(diff) > 0.01 else 0
        self.flip_history.append(is_flip)

        row_data = [
            time.time(), ni, self.lap_count, dt,                   # [1-4]
            self.raw_px, self.raw_py, self.raw_yaw,                 # [5-7]
            filt_px, filt_py, filt_yaw,                             # [8-10] 필터링된 Yaw 기록
            self.current_motion_yaw, path_yaw,                      # [11-12]
            float(self.current_v), float(final_omega),              # [13-14] cmd_v, cmd_w
            self.echo_v, self.echo_w,                               # [15-16] echo_v, echo_w
            self.actual_v, self.battery_voltage, is_flip,           # [17-19]
            curv_ff, cte, omega_pid, omega_ff, omega_yaw,           # [20-24]
            self.params['p_kp'], self.params['p_ki'], self.params['p_kd'], self.params['p_steer_deadzone'], # [25-28]
            self.params['p_ff_gain'], self.params['p_ff_window'], self.params['p_kyaw'], # [29-31]
            self.params['p_v_max'], self.params['p_v_min'], self.params['p_v_accel'], self.params['p_v_decel'], # [32-35]
            self.params['p_v_curve_gain'], self.params['p_v_cte_gain'], # [36-37]
            self.params['p_kf_q_pose'], self.params['p_kf_r_pose'], # [38-39]
            self.params['p_kf_q_yaw'], self.params['p_kf_r_yaw'],   # [40-41] 필터 게인 기록
            self.params['p_yaw_deadzone'],
            1 if self.use_prediction else 0,  # kf_mode 기록
            self.raw_allstate                                       # [42]
        ]
        self.csv_writer.writerow(row_data)

        # 이전 상태 업데이트
        self.prev_filt_px, self.prev_filt_py = filt_px, filt_py
        self.last_omega = final_omega
        self.last_diff = diff

    def stop_vehicle(self):
        msg = Accel()
        msg.linear.x, msg.angular.z = 0.0, 0.0
        for _ in range(10):
            self.pub_accel.publish(msg)
            time.sleep(0.01)

    def close_node(self):
        self.is_finished = True
        self.stop_vehicle()
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.get_logger().info(f"💾 로그 완료: {self.csv_filename}")
        time.sleep(0.5)
        os._exit(0) # rclpy.spin()

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()