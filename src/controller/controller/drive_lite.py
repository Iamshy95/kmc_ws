#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Accel
from std_msgs.msg import Float32, String, Bool
from ament_index_python.packages import get_package_share_directory
import pandas as pd
import numpy as np
import time
import csv
import os
import math
from datetime import datetime

# ==============================================================================
class AdvancedKalman:
    def __init__(self, q=0.1, r=0.1):
        self.q, self.r = q, r
        self.x, self.p = None, 1.0
        self.reject_count = 0 
        self.stall_count = 0   # [신규] 데이터 정체 카운트
        self.prev_raw = None   # [신규] 이전 센서 원본값 저장

    def step(self, measurement, prediction_offset=0.0, gate=None):
        if self.x is None:
            self.x = measurement
            self.prev_raw = measurement
            return self.x

        x_prior = self.x + prediction_offset
        p_prior = self.p + self.q

        # [A] 데이터 정체(Stall) 처리 로직
        # 센서값이 이전 프레임과 토씨 하나 안 틀리고 똑같으면 '지연'으로 판단
        if measurement == self.prev_raw:
            self.stall_count += 1
            # 6회(0.3초) 초과 정체 시 실제 정지 혹은 센서 고장으로 판단하여 수용
            if self.stall_count > 6:
                self.x = measurement
                self.p = 1.0
                return self.x
            # 정체 중에는 예측치(v * dt)를 사용하여 필터를 미리 전진시킴
            self.x = x_prior
            self.p = p_prior
            return self.x
        
        # 새로운 값이 들어오면 정체 카운트 초기화 및 원본 갱신
        self.stall_count = 0
        self.prev_raw = measurement

        # [B] 게이트 체크 (기존 로직 유지)
        if gate is not None and abs(measurement - x_prior) > gate:
            self.reject_count += 1
            if self.reject_count > 6: # 0.3초 이상 튐 지속 시 강제 수용
                self.x = measurement
                self.p = 1.0
                self.reject_count = 0
                return self.x
            self.x = x_prior
            self.p = p_prior
            return self.x

        # [C] 정상 업데이트
        self.reject_count = 0
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
        self.car_id = 3  # 차량 고유 번호 (Remapping 가능)
        self.use_prediction = False # True: 예측 모드, False: 1D in kalman filter
        
        # 경로 파일(CSV) 로드 설정 (홈 디렉토리 기준 절대 경로 구성)
        home_dir = os.path.expanduser('~')
        self.path_file = os.path.join(home_dir, 'kmc_ws/src/controller/path/path3.csv')

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
            "p_v_max": 1.8,        # 목표 선속도 상한 (m/s)
            "p_v_min": 1.2,        # 최저 주행 속도 (m/s)
            "p_v_accel": 1.0, 
            "p_v_decel": 10.0,
            
            # 4. 동적 속도 페널티 계수 (주행 상황별 속도 저감)
            "p_v_curve_gain": 0.3, # 급커브 시 속도 저감 비중
            "p_v_cte_gain": 5.0,   # 경로 이탈 시 속도 저감 비중
            
            # 5. 칼만 필터 게인 세분화 (X, Y 위치 vs Yaw 방향 분리)
            "p_kf_q_pose": 0.1,    # 위치 프로세스 노이즈
            "p_kf_r_pose": 0.1,    # 위치 측정 노이즈
            "p_kf_q_yaw": 0.2,     # Yaw 프로세스 노이즈
            "p_kf_r_yaw": 0.01     # Yaw 측정 노이즈
        }
        

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
        
        self.actual_v_age = 0.0
        
        self.lap_count = 0                # 주행 바퀴 수
        self.halfway_passed = False       # 반환점 통과 여부
        self.flip_history = []            # 조향 진동 기록
        self.last_pose_time = self.get_clock().now() # 초기값 설정
        self.v_buffer = [0.0] * 10  # MA10 버퍼
        self.last_actual_v_time = self.get_clock().now()
        
        # [추가] 정지 판단을 위한 구역 및 상태 변수
        self.roundabout_center = np.array([1.67, 0.0])
        self.go_signal = True
        self.is_active_braking = False
        self.brake_count = 0
        
        # [HV 속도 계산 변수 추가]
        self.latest_hv_pos = None
        self.latest_hv_time = None
        self.last_calc_hv_pos = None
        self.last_calc_hv_time = None
        
        self.kf_hv_v = AdvancedKalman(q=0.1, r=0.1) # 사용자 요청 게인
        self.hv_ma_buffer = [] # MA10용 리스트
        self.hv_filtered_v = 0.0
        
        self.v_smoothed = 0.0

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
            self.pre_aggregated_curvatures = self.precompute_curvatures()
            self.get_logger().info(f"✅ 곡률 지도 미리 계산 완료")
        except Exception as e:
            self.get_logger().error(f"❌ 곡률 계산 에러: {e}")
            self.path = np.array([[0,0], [1,0]]) 

        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # [E] 필터 초기화 및 통신 환경 구축
        # ----------------------------------------------------------------------
        # 세분화된 파라미터를 적용한 칼만 필터 인스턴스 생성
        self.kf_x = AdvancedKalman(self.params['p_kf_q_pose'], self.params['p_kf_r_pose'])
        self.kf_y = AdvancedKalman(self.params['p_kf_q_pose'], self.params['p_kf_r_pose'])
        self.kf_yaw = AdvancedKalman(self.params['p_kf_q_yaw'], self.params['p_kf_r_yaw'])

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        # 퍼블리셔: 제어 명령 Twist 발행 (실차 표준 토픽명 사용)
        self.pub_ctrl = self.create_publisher(Twist, f'/CAV_{self.car_id:02d}/cmd_vel', 10)
        
        # 서브스크라이버: 위치, 속도, 전압, 에코, 전체 상태 수신
        # 서브스크라이버: 실제 위치 데이터 수신 (차량 번호 포함)
        self.sub_pose = self.create_subscription(
            PoseStamped, 
            f'/CAV_{self.car_id:02d}',  # 'pose' 대신 원래 쓰시던 이 형식이 더 정확할 겁니다!
            self.pose_callback, 
            qos
        )
        self.sub_actual_v = self.create_subscription(Float32, f'/CAV_{self.car_id:02d}/vehicle_speed', self.actual_v_callback, 10)
        self.sub_battery = self.create_subscription(Float32, f'/CAV_{self.car_id:02d}/battery_voltage', self.battery_callback, 10)
        self.sub_echo = self.create_subscription(Twist, f'/CAV_{self.car_id:02d}/cmd_echo', self.echo_callback, 10)
        self.sub_allstate = self.create_subscription(String, f'/CAV_{self.car_id:02d}/allstate_text', self.allstate_callback, 10)
        
        self.sub_infra = self.create_subscription(Bool, f'/infra/CAV_{self.car_id:02d}/go_signal', self.infra_callback, 10)
        self.sub_hv = self.create_subscription(PoseStamped, '/HV_19', self.hv_callback, qos)
        
        # 제어 주기 타이머: 20Hz (0.05s)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.curr_pose = None
        
        
    def infra_callback(self, msg): self.go_signal = msg.data
    def hv_callback(self, msg):
        # 데이터 수신 시 최신 값만 저장
        self.latest_hv_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        self.latest_hv_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        
    # [신규 메서드 추가]
    def precompute_curvatures(self):
        """ 경로 전체에 대해 [상위 20% 평균 + 윈도우 100] 곡률을 미리 계산 """
        n = len(self.path)
        raw_curvatures = np.zeros(n)
        gap = 10  # 10cm 간격으로 점을 찍어 노이즈 억제 (3점 곡률 방식)
        
        # 1. 모든 점에 대해 순수 곡률(Raw) 계산
        for i in range(n):
            p1 = self.path[(i - gap) % n]
            p2 = self.path[i]
            p3 = self.path[(i + gap) % n]
            
            # 세 점이 만드는 삼각형 면적 기반 곡률 계산
            area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p2 - p3)
            c = np.linalg.norm(p3 - p1)
            
            if a*b*c > 1e-6:
                curv = (4 * area) / (a * b * c)
            else:
                curv = 0.0
            raw_curvatures[i] = min(curv, 2.0) # [요청반영] 최대 곡률 2.0 제한

        # 2. 윈도우 100개를 돌며 상위 20% 평균 산출
        aggregated = []
        window_size = 100
        top_n = 20
        for i in range(n):
            # 미래 100개 지점의 곡률 확보
            win_indices = [(i + j) % n for j in range(window_size)]
            window = raw_curvatures[win_indices]
            # 상위 20개 추출 후 평균
            top_vals = np.sort(window)[-top_n:]
            aggregated.append(np.mean(top_vals))
            
        return np.array(aggregated)

    # --------------------------------------------------------------------------
    # [콜백 함수] - 실시간 데이터 수신 및 전처리
    # --------------------------------------------------------------------------
    def pose_callback(self, msg):
        # 1. 센서 간의 실제 시간 간격(pose_dt) 계산
        now = self.get_clock().now()
        pose_dt = (now - self.last_pose_time).nanoseconds / 1e9
        self.last_pose_time = now
    
            
        raw_px, raw_py = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        raw_yaw_val = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        
        self.raw_px, self.raw_py, self.raw_yaw = raw_px, raw_py, raw_yaw_val

        # 위상 도약 방지 (Yaw 센서는 로그용으로만 필터링)
        if self.kf_yaw.x is not None:
            while raw_yaw_val - self.kf_yaw.x > np.pi: raw_yaw_val -= 2*np.pi
            while raw_yaw_val - self.kf_yaw.x < -np.pi: raw_yaw_val += 2*np.pi
            
        # 🌟 [추가] 첫 프레임 스마트 초기화 로직
        # ==========================================================
        if self.kf_x.x is None:
            self.kf_x.x = raw_px
            self.kf_y.x = raw_py
            self.kf_yaw.x = raw_yaw_val 
            
            # 5초 로직 덕분에 전체 경로에서 가장 가까운 곳을 찾습니다.
            ni_init = self.find_nearest_global(raw_px, raw_py)
            # 해당 위치의 경로 방향(path_yaw)을 따옵니다.
            init_path_yaw, _ = self.get_control_metrics(raw_px, raw_py, ni_init)
            
            # [중요] 초기 방향을 경로 방향으로 강제 셋팅해서 게이트 이탈 방지
            self.current_motion_yaw = init_path_yaw
            self.last_valid_motion_yaw = init_path_yaw
            self.get_logger().info(f"✅ 초기화: Path Yaw({np.degrees(init_path_yaw):.1f} deg) 적용")
            return # 첫 루프는 여기서 끝내야 필터 오류가 안 납니다.
        # ==========================================================
        
        # ==========================================================
        # [핵심 수정] 2번 모델: Motion Yaw + Steering Command 예측
        # ==========================================================
        # [추가] 데이터 신선도 체크 (현재 시각 - 마지막 수신 시각)
        self.actual_v_age = (self.get_clock().now() - self.last_actual_v_time).nanoseconds / 1e9

        v_for_prediction = self.v_smoothed

        # [수정] 결정된 v_for_prediction을 사용하여 dx, dy 계산
        if self.use_prediction:
            # V2 예측 모드: 물리 모델(v, omega) 반영
            predicted_yaw = self.current_motion_yaw + (self.last_omega * pose_dt)
            dx = v_for_prediction * np.cos(predicted_yaw) * pose_dt
            dy = v_for_prediction * np.sin(predicted_yaw) * pose_dt
        else:
            dx = dy = 0.0
            
        dynamic_gate =  0.5  # dynamic 아님;;

        # 필터 업데이트 (dx, dy, gate가 모드에 따라 자동 적용됨)
        self.filtered_pose = [
            self.kf_x.step(raw_px, dx, gate=dynamic_gate),
            self.kf_y.step(raw_py, dy, gate=dynamic_gate),
            self.kf_yaw.step(raw_yaw_val) 
        ]
        self.curr_pose = msg
        
    def actual_v_callback(self, msg):
        self.actual_v = msg.data
        # [수정] 데이터 수신 시각 업데이트
        self.last_actual_v_time = self.get_clock().now()
        
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
        window_size = 300 
        
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
                
                if self.lap_count >= 10 and self.finish_check_time is None:
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
        p1, p2, p3 = self.path[ni], self.path[(ni+window//2)%len(self.path)], self.path[(ni+window)%len(self.path)]
        v1, v2 = p2 - p1, p3 - p2
        ang = (np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]) + np.pi) % (2*np.pi) - np.pi
        dist = np.linalg.norm(p3 - p1)
        return ang / dist if dist > 0.01 else 0.0

    # --------------------------------------------------------------------------
    # [메인 제어 루프] - 20Hz 알고리즘 실행 및 로깅
    # --------------------------------------------------------------------------
    def control_loop(self):
        if self.curr_pose is None or self.is_finished: return

        # 주행 종료 조건 체크
        if self.finish_check_time and (time.time() - self.finish_check_time > 0.5):
            self.close_node(); return
            
        # --- [HV 속도 계산 로직 시작] ---
        if self.latest_hv_pos is not None and self.last_calc_hv_time is not None:
            actual_dt = self.latest_hv_time - self.last_calc_hv_time
            
            # dt가 0.02초 초과일 때만 새 속도 계산 (스킵 로직)
            if actual_dt > 0.02:
                dist = np.linalg.norm(self.latest_hv_pos - self.last_calc_hv_pos)
                raw_vel = np.clip(dist / actual_dt, 0.1, 2.0) # 클램핑

                
                # 칼만 필터 (1차)
                kf_v = self.kf_hv_v.step(raw_vel)
                
                # MA20 (2차)
                self.hv_ma_buffer.append(kf_v)
                if len(self.hv_ma_buffer) > 20:
                    self.hv_ma_buffer.pop(0)
                self.hv_filtered_v = sum(self.hv_ma_buffer) / len(self.hv_ma_buffer)
                
                # 계산에 사용된 시점 업데이트
                self.last_calc_hv_pos = self.latest_hv_pos.copy()
                self.last_calc_hv_time = self.latest_hv_time
        elif self.latest_hv_pos is not None:
            # 초기값 설정
            self.last_calc_hv_pos = self.latest_hv_pos.copy()
            self.last_calc_hv_time = self.latest_hv_time
        # --- [HV 속도 계산 로직 끝] ---

        # 1. 샘플링 타임(dt) 계산
        now = self.get_clock().now()
        dt = max(0.001, (now - self.last_time).nanoseconds / 1e9)
        dt = min(dt, 0.1)
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
        
        # --- [Step 6. 속도 제어 로직 수정] ---
        
        # 1. 미리 계산된 상위 20% 평균 곡률값 즉시 획득
        avg_future_curv = self.pre_aggregated_curvatures[ni]

        # 속도 전용 3cm 데드존 적용
        # 1. 속도 페널티 전용 3cm(0.03) 데드존 설정
        v_dead = 0.02 

        # 2. 연속형(Soft) 데드존 로직 적용
        # 3cm 이내면 페널티 0, 3cm를 넘어서는 순간 0부터 부드럽게 페널티 증가
        if abs(cte) < v_dead:
            e_v_cte = 0.0
        else:
            e_v_cte = abs(cte) - v_dead

        v_penalty = (avg_future_curv * self.params['p_v_curve_gain']) + (e_v_cte * self.params['p_v_cte_gain'])

        # 하한선을 1.2로 낮추고 타겟 속도 산출
        target_v = np.clip(self.params['p_v_max'] - v_penalty, self.params['p_v_min'], self.params['p_v_max'])
        
        # --- [여기서부터 삽입] ---
        # 1. 정지 조건 판단 (인프라 신호 + 구역 체크)
        dist_to_round = np.linalg.norm(np.array([filt_px, filt_py]) - self.roundabout_center)
        is_4way = (-4.3 <= filt_px <= -0.4) and (-1.6 <= filt_py <= 1.6)
        is_zone1 = (-4.1 <= filt_px <= -1.4) and (1.1 <= filt_py <= 2.6) 
        is_zone2 = (-3.3 <= filt_px <= -0.5) and (-2.6 <= filt_py <= -1.1)

        stop_condition = not self.go_signal and ((1.1 < dist_to_round < 1.9) or is_4way or is_zone1 or is_zone2)
        
        # [Step 2] HV 차량 속도 추종 (곡률 감속 무시 + 하한선 제거)
        # [Step 2] HV 차량 속도 추종 및 후방 추돌 방지 (수정됨)
        if dist_to_round < 1.3 and self.latest_hv_pos is not None:
            path_len = len(self.path)
            is_hv_nearby = False
            look_ahead_count = 300  # 약 3m 전방 확인
            
            # 1. 내 앞 경로 3m 구간 중 HV가 위치한 곳이 있는지 전수 조사 (길막 체크)
            for i in range(1, look_ahead_count + 1):
                check_idx = (ni + i) % path_len
                d_to_hv = np.linalg.norm(self.path[check_idx] - self.latest_hv_pos)
                if d_to_hv < 0.5: # 0.5m 이내면 "내 앞 경로에 차가 있다"고 판단
                    is_hv_nearby = True
                    break
            
            # 2. 속도 결정
            hv_v = float(self.hv_filtered_v)
            # [사용자 요청] 0.1 하한선 안전장치 (데이터 소실 대비)
            hv_v = max(0.1, hv_v) 

            if is_hv_nearby:
                # [길막 중] 앞차 속도에 맞춰서 서행/정지
                target_v = hv_v
            else:
                # [길막 없음] 뒤차(HV)가 나보다 빠르면 그 속도에 맞춰서 빨리 탈출 (max 적용)
                target_v = max(target_v, hv_v)

        # 2. 속도 가로채기 (Override)
        if stop_condition:
            # 음수 제동 시퀀스 시작
            if not self.is_active_braking and self.current_v > 0.1:
                self.is_active_braking = True
                self.brake_count = 6  # 10회 동안 역방향 출력

            if self.is_active_braking and self.brake_count > 0:
                target_v = -0.05      # 역방향 제동값
                self.brake_count -= 1
            else:
                target_v = 0.0        # 제동 완료 후 정지 유지
                
            self.error_integral = 0.0 # 정지 중 PID 적분항 초기화 (Anti-windup)
        else:
            # 주행 신호가 들어오면 제동 상태 해제
            self.is_active_braking = False
            self.brake_count = 0
        # --- [여기까지 삽입] ---
        
        # 4. MA10 필터링 (기존 동일)
        self.v_buffer.pop(0)
        self.v_buffer.append(target_v)
        self.v_smoothed = sum(self.v_buffer) / 10.0
        
        # 감속도를 상황에 따라 이원화
        if target_v > 0.1:
            current_decel = 2.0  # 주행 중 감속 (커브 등) - 부드럽게!
        else:
            current_decel = self.params.get('p_v_decel', 4.0)  # 정지 상황 - 빡세게!
            
        
        acc_lim = (self.params.get('p_v_accel', 1.0) if target_v > self.current_v else current_decel) * dt
        self.current_v = np.clip(target_v, self.current_v - acc_lim, self.current_v + acc_lim)
                
        # 7. 통합 조향 제어 (PID + FF + Yaw Correction)
        
        # PID: 연속형 데드존 적용
        deadzone = self.params['p_steer_deadzone']
        e_dead = 0.0 if abs(cte) < deadzone else cte - (np.sign(cte) * deadzone)
        
        # 수정 로직 추가
        if self.current_v < 0.1:        # 차 속도가 0.1m/s 이하일 때는
            self.error_integral = 0.0   # 적분항을 강제로 0으로 묶어둠
        else:
            # 기존의 적분항 계산 로직 실행
            self.error_integral = np.clip(self.error_integral + e_dead * dt, -1.0, 1.0)
        cte_d = (e_dead - self.last_error) / dt
        d_deadzone = 0.02  # 0.01에서 0.02로 상향 (노이즈 컷)

        if abs(cte_d) < d_deadzone:
            cte_d_soft = 0.0
        else:
            cte_d_soft = cte_d - (np.sign(cte_d) * d_deadzone)

        # D-항만 따로 계산해서 1.0으로 클램핑 (발작 봉쇄)
        d_term = -(self.params['p_kd'] * cte_d_soft)
        d_term_clamped = np.clip(d_term, -1.0, 1.0)

        # 최종 PID 합체 (P, I항은 그대로, D항만 클램핑된 것 사용)
        omega_pid = -((self.params['p_kp'] * e_dead) + 
                    (self.params['p_ki'] * self.error_integral)) + d_term_clamped
        self.last_error = e_dead

        # Feed Forward: 경로 곡률 비례 조향
        # 수정 후
        if self.current_v < 0:
            # 음수 제동 중에는 피드포워드(곡률 보정)를 0으로 만들어 바퀴가 반대로 튀는 것을 막습니다.
            omega_ff = 0.0
        else:
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
        msg = Twist()
        msg.linear.x = float(self.current_v)
        msg.angular.z = float(final_omega)
        self.pub_ctrl.publish(msg)
        
        # 9. 실시간 데이터 로깅 (총 42개 컬럼 정확히 매칭)
        diff = final_omega - self.last_omega
        is_flip = 1 if (diff * self.last_diff) < 0 and abs(diff) > 0.01 else 0
        self.flip_history.append(is_flip)

        

        # 이전 상태 업데이트
        self.prev_filt_px, self.prev_filt_py = filt_px, filt_py
        self.last_omega = final_omega
        self.last_diff = diff

    def stop_vehicle(self):
        msg = Twist()
        msg.linear.x, msg.angular.z = 0.0, 0.0
        for _ in range(10):
            self.pub_ctrl.publish(msg)
            time.sleep(0.01)

    def close_node(self):
        self.is_finished = True
        self.stop_vehicle()
        
        time.sleep(0.5)
        

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_node()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()