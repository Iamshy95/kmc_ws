#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32  # <--- 이 줄 추가
from ament_index_python.packages import get_package_share_directory
import pandas as pd
import numpy as np
import time
import csv
import os
import math
from datetime import datetime

# ==============================================================================
# [1. 유틸리티 클래스] - 칼만 필터 (센서 노이즈 제거용)
# ==============================================================================
class SimpleKalman:
    def __init__(self, q=0.1, r=0.1):
        self.q = q  # Process Noise (예측 오차 공분산)
        self.r = r  # Measurement Noise (측정 오차 공분산)
        self.x = None  # 상태 추정값
        self.p = 1.0   # 오차 공분산

    def step(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        # 1. Prediction (이전 값을 그대로 예측한다고 가정)
        p_prior = self.p + self.q
        # 2. Update (측정값 반영)
        k_gain = p_prior / (p_prior + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1 - k_gain) * p_prior
        return self.x

# ==============================================================================
# [2. 통합 주행 노드] - UnifiedFollower (실전용)
# ==============================================================================
class UnifiedFollower(Node):
    def __init__(self):
        # 런치파일에서 네임스페이스를 부여하므로 노드 이름은 고정
        super().__init__('unified_follower')
        
        # ----------------------------------------------------------------------
        # [A] 하드코딩 파라미터 존 (현장에서 여기만 수정하면 됨)
        # ----------------------------------------------------------------------
        self.car_id = 1  # 차량 번호 (1, 2, 3, 4)
        
        # 주행 경로 파일 설정 (절대 경로 사용 권장)
        # 예: /home/user/kmc_ws/src/controller/path/path3-1.csv
        home_dir = os.path.expanduser('~')
        self.path_file = os.path.join(home_dir, 'kmc_ws/src/controller/path/path3-1.csv')

        # 제어 파라미터 (Optuna 제거, PP 제거, 실전 최적화 값)
        self.params = {
            # 1. PID 제어 계수
            "p_kp": 3.0,
            "p_ki": 1.5,
            "p_kd": 3.0,
            "p_steer_deadzone": 0.005,  # 연속형 데드존 (0.005m 이하 무시)

            # 2. FeedForward & Yaw 보정
            "p_ff_gain": 2.0,      # 곡률 기반 FF 게인
            "p_ff_window": 10,     # 곡률 계산 윈도우 (데이터 포인트 수)
            "p_kyaw": 1.0,         # Yaw 오차 보정 게인 (제출용 코드 핵심)
            "p_gamma": 1.0,        # 최종 출력 스케일링

            # 3. 속도 프로파일 (가감속 제한 필수)
            "p_v_max": 2.0,        # 최대 속도 (m/s)
            "p_v_min": 0.5,        # 최소 속도 (코너 등에서)
            "p_v_accel": 1.5,      # 가속도 제한 (m/s^2) - 급출발 방지
            "p_v_decel": 3.0,      # 감속도 제한 (m/s^2) - 급제동 허용
            
            # 4. 상황별 감속 계수 (커브, 조향 시 감속)
            "p_v_curve_gain": 0.3, # 곡률이 클 때 감속
            "p_v_steer_gain": 0.0, # 핸들 많이 꺾을 때 감속 (현재 0.0)
            "p_v_cte_gain": 0.1,   # 경로 이탈 시 감속
            
            # 5. 칼만 필터 게인
            "p_kf_q": 0.1,
            "p_kf_r": 0.1
        }
        
        # 로그 저장 경로 (사용자 요청: controller/logs)
        self.log_dir = os.path.join(home_dir, 'kmc_ws/src/controller/logs')

        # ----------------------------------------------------------------------
        # [B] 초기화 및 상태 변수
        # ----------------------------------------------------------------------
        self.current_v = 0.0
        self.filtered_pose = [0.0, 0.0, 0.0] # x, y, yaw
        self.prev_ni = None
        
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()
        self.start_time = time.time()
        self.is_finished = False
        self.finish_check_time = None
        
        # 초기화 및 상태 변수 부분에 추가
        self.actual_v = 0.0
        self.battery_voltage = 0.0
        
        # 위치 예측 및 Yaw 계산용 메모리
        self.prev_filt_px = None
        self.prev_filt_py = None
        self.current_motion_yaw = 0.0
        self.last_valid_motion_yaw = 0.0
        self.last_path_yaw = 0.0
        self.last_omega = 0.0
        self.last_diff = 0.0
        
        self.lap_count = 0
        self.halfway_passed = False
        self.flip_history = []

        # 로그용 원본 데이터
        self.raw_px = 0.0
        self.raw_py = 0.0
        self.raw_yaw = 0.0

        # ----------------------------------------------------------------------
        # [C] 경로 로드
        # ----------------------------------------------------------------------
        try:
            if not os.path.exists(self.path_file):
                raise FileNotFoundError(f"파일 없음: {self.path_file}")
            df = pd.read_csv(self.path_file, header=None)
            self.path = df.apply(pd.to_numeric, errors='coerce').dropna().values
            self.get_logger().info(f"✅ 경로 로드 완료: {len(self.path)} points")
        except Exception as e:
            self.get_logger().error(f"❌ 경로 로드 실패: {e}")
            # 비상시 빈 경로라도 생성하여 노드 다운 방지
            self.path = np.array([[0,0], [1,0]]) 

        # ----------------------------------------------------------------------
        # [D] 로그 파일 설정 (요청사항 반영)
        # ----------------------------------------------------------------------
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f"{self.log_dir}/log_car{self.car_id}_{timestamp}.csv"
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 💡 [로그 헤더] PP 관련 제거, dt 추가 (총 36개)
        self.log_headers = [
            'time', 'ni', 'lap_count',                              # Basic
            'raw_px', 'raw_py', 'raw_yaw',                          # Raw Sensor
            'filt_px', 'filt_py', 'motion_yaw', 'path_yaw',         # Filtered
            'pred_px', 'pred_py', 'dt',                             # Prediction (dt 추가!)
            'velocity', 'curvature', 'cte', 'final_omega',          # Control Output
            'p_kp', 'p_ki', 'p_kd', 'p_steer_deadzone',             # PID Params
            'p_ff_gain', 'p_ff_window', 'p_kyaw', 'p_gamma',        # FF & Yaw Params
            'p_v_max', 'p_v_min', 'p_v_accel', 'p_v_decel',         # Speed Params
            'p_v_curve_gain', 'p_v_steer_gain', 'p_v_cte_gain',     # Speed Penalties
            'omega_pid', 'omega_ff', 'omega_yaw',                   # Control Components
            'is_flip','actual_v', 'battery'                         # Debug
        ]
        self.csv_writer.writerow(self.log_headers)

        # ----------------------------------------------------------------------
        # [E] 필터 및 통신 설정
        # ----------------------------------------------------------------------
        self.kf_x = SimpleKalman(self.params['p_kf_q'], self.params['p_kf_r'])
        self.kf_y = SimpleKalman(self.params['p_kf_q'], self.params['p_kf_r'])
        self.kf_yaw = SimpleKalman(0.2, 0.01)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        # ⚠️ [중요] 실차 SDK 규격에 맞춰 Twist 메시지 사용 & 토픽명 변경
        # 네임스페이스가 씌워지므로 토픽명은 그냥 'cmd_vel', 'pose' 등을 사용해도 됨
        # 하지만 명시적으로 기존 구조를 유지하려면 아래와 같이 사용
        topic_cmd = f'/CAV_0{self.car_id}/cmd_vel' if False else '/cmd_vel' # 네임스페이스 사용 시
        # 여기서는 사용자 요청대로 네임스페이스 없이도 돌 수 있게 명시적 이름 사용하되 Twist로 변경
        # (현장 런치파일에서 remapping 하거나 namespace 씌우면 됨)
        
        self.pub_ctrl = self.create_publisher(Twist, 'cmd_vel', 10) # 런치파일이 이름 붙여줌
        self.sub_pose = self.create_subscription(PoseStamped, 'pose', self.pose_callback, qos) # 런치파일이 이름 붙여줌
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.curr_pose = None
        
        
        # 실제 속도 구독
        self.sub_actual_v = self.create_subscription(
            Float32, 'vehicle_speed', self.actual_v_callback, 10)
        # 배터리 전압 구독 (필요 시)
        self.sub_battery = self.create_subscription(
            Float32, 'battery_voltage', self.battery_callback, 10)

    def pose_callback(self, msg):
        """센서 데이터 수신 및 칼만 필터링"""
        raw_px, raw_py = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        # 쿼터니언 -> 오일러 (Yaw) 변환
        raw_yaw_val = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        
        self.raw_px, self.raw_py, self.raw_yaw = raw_px, raw_py, raw_yaw_val

        # Yaw Unwrapping (불연속점 제거)
        if self.kf_yaw.x is not None:
            while raw_yaw_val - self.kf_yaw.x > np.pi: raw_yaw_val -= 2*np.pi
            while raw_yaw_val - self.kf_yaw.x < -np.pi: raw_yaw_val += 2*np.pi
        
        self.filtered_pose = [
            self.kf_x.step(raw_px),
            self.kf_y.step(raw_py),
            self.kf_yaw.step(raw_yaw_val)
        ]
        self.curr_pose = msg
        
    def actual_v_callback(self, msg):
        self.actual_v = msg.data

    def battery_callback(self, msg):
        self.battery_voltage = msg.data

    # ==========================================================================
    # [핵심 로직] 제출용 코드의 로직 100% 유지 (Nearest, Control Metric, Curvature)
    # ==========================================================================
    def find_nearest_global(self, px, py):
        path_len = len(self.path)
        dists = np.sqrt(np.sum((self.path - [px, py])**2, axis=1))
        
        # 시작 초기에는 전역 탐색
        if self.prev_ni is None or time.time() - self.start_time < 5.0:
            ni = np.argmin(dists)
            self.prev_ni = ni
            return ni

        # 이후에는 이전 인덱스 주변 탐색 및 역주행 방지 페널티 적용
        look_range = 200
        indices = np.arange(path_len)
        diff = np.abs(indices - self.prev_ni)
        diff = np.minimum(diff, path_len - diff) # 순환 구조 대응
        
        # 멀리 있는 점에는 페널티를 주어 인덱스 튐 방지
        dists += np.where(diff > look_range, 0.2, 0.0) 
        ni = np.argmin(dists)
        
        # 바퀴 수 카운팅 로직
        if ni > path_len * 0.5: self.halfway_passed = True
        if self.halfway_passed and self.prev_ni > path_len * 0.9 and ni < path_len * 0.1:
            self.lap_count += 1
            self.halfway_passed = False
            self.get_logger().info(f"🚩 Lap Count Up! ({self.lap_count} laps)")
            if self.lap_count >= 10 and self.finish_check_time is None:
                self.finish_check_time = time.time()

        self.prev_ni = ni
        return ni

    def get_control_metrics(self, px, py, ni):
        path_len = len(self.path)
        # LS(최소자승)를 위한 주변 점 추출
        indices = [(ni + i) % path_len for i in range(-5, 6)]
        pts = self.path[indices]
        
        center = np.mean(pts, axis=0)
        norm_pts = pts - center
        cov = np.dot(norm_pts.T, norm_pts)
        val, vec = np.linalg.eigh(cov)
        tangent = vec[:, np.argmax(val)]
        
        path_yaw = np.arctan2(tangent[1], tangent[0])
        next_idx = (ni + 1) % path_len
        # 주행 방향으로 Yaw 정렬
        if np.dot(tangent, self.path[next_idx] - self.path[ni]) < 0:
            path_yaw += np.pi
                
        dx, dy = px - self.path[ni][0], py - self.path[ni][1]
        cte = -np.sin(path_yaw)*dx + np.cos(path_yaw)*dy
        return path_yaw, cte

    def get_curvature(self, ni, window):
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

    # ==========================================================================
    # [메인 제어 루프] 제출용 코드 로직 + 로그 기능 + Twist 변환
    # ==========================================================================
    def control_loop(self):
        if self.curr_pose is None or self.is_finished: return

        if self.finish_check_time and (time.time() - self.finish_check_time > 0.5):
            self.close_node(); return

        # 1. dt 계산 (실시간성 반영)
        now = self.get_clock().now()
        dt = max(0.001, (now - self.last_time).nanoseconds / 1e9)
        self.last_time = now

        # 2. 필터링된 위치 가져오기
        filt_px, filt_py, _ = self.filtered_pose
        
        # 3. Motion Yaw 계산 (제출용 코드 핵심 로직)
        # Yaw 데이터 노이즈를 피하기 위해 실제 이동 벡터로 방향을 계산
        temp_ni = self.find_nearest_global(filt_px, filt_py)
        temp_path_yaw, _ = self.get_control_metrics(filt_px, filt_py, temp_ni)
        self.last_path_yaw = temp_path_yaw # 백업용
        
        if self.prev_filt_px is not None:
            dx = filt_px - self.prev_filt_px
            dy = filt_py - self.prev_filt_py
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.005: # 5mm 이상 움직여야 유효
                self.current_motion_yaw = np.arctan2(dy, dx)
                self.last_valid_motion_yaw = self.current_motion_yaw
            else:
                self.current_motion_yaw = self.last_path_yaw # 정지 시 경로 방향 사용
        else:
            self.current_motion_yaw = self.last_path_yaw

        # 4. 위치 예측 (Latency 보상 - 제출용 코드 핵심)
        # dt만큼 미래 위치를 예측하여 제어 지연 보상
        pred_px = filt_px + (self.current_v * np.cos(self.current_motion_yaw) * dt)
        pred_py = filt_py + (self.current_v * np.sin(self.current_motion_yaw) * dt)

        # 5. 제어 지표 산출
        ni = self.find_nearest_global(pred_px, pred_py)
        path_yaw, cte = self.get_control_metrics(pred_px, pred_py, ni)
        curv_ff = self.get_curvature(ni, int(self.params['p_ff_window']))
        
        # 6. 속도 프로파일 (Slew-rate limit 적용)
        v_penalty = (abs(curv_ff) * self.params['p_v_curve_gain']) + \
                    (abs(self.last_omega) * self.params['p_v_steer_gain']) + \
                    (abs(cte) * self.params['p_v_cte_gain'])
        
        target_v = np.clip(self.params['p_v_max'] - v_penalty, self.params['p_v_min'], self.params['p_v_max'])
        
        # 가감속 제한 (급격한 속도 변화 방지)
        accel_limit = self.params['p_v_accel'] * dt if target_v > self.current_v else self.params['p_v_decel'] * dt
        self.current_v = np.clip(target_v, self.current_v - accel_limit, self.current_v + accel_limit)

        # 7. 조향 제어 (PP 삭제, PID + FF + Yaw 보정)
        
        # (1) PID - 연속형 데드존 (Continuous Deadzone - 사용자님 아이디어)
        deadzone = self.params['p_steer_deadzone']
        # 오차에서 데드존을 뺀 값을 사용하여 0부터 부드럽게 시작
        e_dead = 0.0 if abs(cte) < deadzone else cte - (np.sign(cte) * deadzone)
        
        self.error_integral = np.clip(self.error_integral + e_dead * dt, -1.0, 1.0)
        cte_d = (e_dead - self.last_error) / dt
        omega_pid = -((self.params['p_kp'] * e_dead) + (self.params['p_ki'] * self.error_integral) + (self.params['p_kd'] * cte_d))
        self.last_error = e_dead

        # (2) Feed Forward
        omega_ff = self.current_v * curv_ff * self.params['p_ff_gain']

        # (3) Yaw 보정 (PP 대체재)
        yaw_err = self.current_motion_yaw - path_yaw
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        # 곡률이 클수록(커브) Yaw 보정 힘을 뺌 (진동 방지)
        yaw_gate = 1.0 / (1.0 + abs(curv_ff) * 10.0)
        omega_yaw = -self.params['p_kyaw'] * yaw_err * yaw_gate

        # 최종 합산
        omega_raw = omega_pid + omega_ff + omega_yaw
        final_omega = np.clip(omega_raw * self.params['p_gamma'], -6.0, 6.0)

        # 8. 메시지 발행 (Twist로 변경!)
        msg = Twist()
        msg.linear.x = float(self.current_v)
        msg.angular.z = float(final_omega)
        self.pub_ctrl.publish(msg)
        
        # 9. 로그 기록 (요청사항 반영)
        # Flip 감지
        diff = final_omega - self.last_omega
        is_flip = 1 if (diff * self.last_diff) < 0 and abs(diff) > 0.01 else 0
        self.flip_history.append(is_flip)

        row_data = [
            time.time(), ni, self.lap_count,
            self.raw_px, self.raw_py, self.raw_yaw,
            filt_px, filt_py, self.current_motion_yaw, path_yaw,
            pred_px, pred_py, dt,  # dt 추가
            self.current_v, curv_ff, cte, final_omega,
            self.params['p_kp'], self.params['p_ki'], self.params['p_kd'], self.params['p_steer_deadzone'],
            self.params['p_ff_gain'], self.params['p_ff_window'], self.params['p_kyaw'], self.params['p_gamma'],
            self.params['p_v_max'], self.params['p_v_min'], self.params['p_v_accel'], self.params['p_v_decel'],
            self.params['p_v_curve_gain'], self.params['p_v_steer_gain'], self.params['p_v_cte_gain'],
            omega_pid, omega_ff, omega_yaw,
            is_flip
        ]
        self.csv_writer.writerow(row_data)

        # 다음 루프 준비
        self.prev_filt_px, self.prev_filt_py = filt_px, filt_py
        self.last_omega = final_omega
        self.last_diff = diff

    def stop_vehicle(self):
        msg = Twist() # Twist 사용
        msg.linear.x, msg.angular.z = 0.0, 0.0
        for _ in range(10):
            self.pub_ctrl.publish(msg)
            time.sleep(0.01)

    def close_node(self):
        self.is_finished = True
        self.stop_vehicle()
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        self.get_logger().info(f"💾 로그 저장 완료: {self.csv_filename}")
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()