#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Accel
from std_msgs.msg import Float32, String, Bool
import pandas as pd
import numpy as np
import time
import csv
import os
import math
from datetime import datetime

# ==============================================================================
# [1. 유틸리티 클래스] - AdvancedKalman (원본 유지)
# ==============================================================================
class AdvancedKalman:
    def __init__(self, q=0.1, r=0.1):
        self.q, self.r = q, r
        self.x, self.p = None, 1.0

    def step(self, measurement, prediction_offset=0.0, gate=None):
        if self.x is None:
            self.x = measurement
            return self.x
        x_prior = self.x + prediction_offset
        p_prior = self.p + self.q
        if gate is not None and abs(measurement - x_prior) > gate:
            self.x = x_prior
            self.p = p_prior
            return self.x
        k_gain = p_prior / (p_prior + self.r)
        self.x = x_prior + k_gain * (measurement - x_prior)
        self.p = (1 - k_gain) * p_prior
        return self.x

# ==============================================================================
# [2. 통합 주행 노드] - UnifiedFollower
# ==============================================================================
class UnifiedFollower(Node):
    def __init__(self):
        super().__init__('unified_follower')

        # [A] 파라미터 및 경로 설정
        self.car_id = 11
        self.use_prediction = True
        self.path_file = "/home/cav-11/Desktop/path3-2.csv"

        self.params = {
            "p_kp": 3.0, "p_ki": 1.5, "p_kd": 3.0,
            "p_steer_deadzone": 0.005, "p_yaw_deadzone": 3.0,
            "p_ff_gain": 2.0, "p_ff_window": 10, "p_kyaw": 1.0,
            "p_v_max": 1.2 , "p_v_min": 1.0, "p_v_accel": 1.0, "p_v_decel": 4.0,
            "p_v_curve_gain": 0.3, "p_v_cte_gain": 0.1,
            "p_kf_q_pose": 0.1, "p_kf_r_pose": 0.1, "p_kf_q_yaw": 0.2, "p_kf_r_yaw": 0.01
        }

        self.roundabout_center = np.array([1.67, 0.0])
        self.go_signal = True
        self.hv_filtered_v = 0.0
        self.last_hv_pos = None
        self.last_hv_time = None
        self.kf_hv_v = AdvancedKalman(q=0.05, r=0.5)

        # [B] 상태 변수
        self.current_v = 0.0
        self.filtered_pose = [0.0, 0.0, 0.0]
        self.prev_ni = None
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = self.get_clock().now()
        self.start_time = time.time()
        self.is_finished = False
        self.finish_check_time = None
        self.actual_v, self.battery_voltage = 0.0, 0.0
        self.echo_v, self.echo_w = 0.0, 0.0
        self.raw_allstate = ""
        self.prev_filt_px, self.prev_filt_py = None, None
        self.current_motion_yaw = 0.0
        self.last_valid_motion_yaw = 0.0
        self.last_path_yaw = 0.0
        self.last_omega, self.last_diff = 0.0, 0.0
        self.lap_count = 0
        self.halfway_passed = False
        self.last_pose_time = self.get_clock().now()
        self.v_buffer = [0.0] * 10
        
        # __init__ 메서드 내부에 추가
        self.is_active_braking = False  # 현재 역방향 제동 중인지 여부
        self.brake_count = 0           # 역방향 명령을 보낼 횟수 (10개 이동평균 고려)
        
        self.raw_px, self.raw_py, self.raw_yaw = 0.0, 0.0, 0.0
        self.curr_pose = None

        self.load_path_data()

        # [C] 필터 및 통신 설정
        self.kf_x = AdvancedKalman(self.params['p_kf_q_pose'], self.params['p_kf_r_pose'])
        self.kf_y = AdvancedKalman(self.params['p_kf_q_pose'], self.params['p_kf_r_pose'])
        self.kf_yaw = AdvancedKalman(self.params['p_kf_q_yaw'], self.params['p_kf_r_yaw'])

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_accel = self.create_publisher(Twist, f'/CAV_{self.car_id}/cmd_vel', 10)
        self.sub_pose = self.create_subscription(PoseStamped, f'/CAV_{self.car_id}', self.pose_callback, qos)
        self.sub_actual_v = self.create_subscription(Float32, 'vehicle_speed', self.actual_v_callback, 10)
        self.sub_battery = self.create_subscription(Float32, 'battery_voltage', self.battery_callback, 10)
        self.sub_echo = self.create_subscription(Twist, 'cmd_echo', self.echo_callback, 10)
        self.sub_allstate = self.create_subscription(String, 'allstate_text', self.allstate_callback, 10)
        self.sub_infra = self.create_subscription(Bool, f'/infra/CAV_{self.car_id}/go_signal', self.infra_callback, 10)
        self.sub_hv = self.create_subscription(PoseStamped, '/HV_19', self.hv_callback, qos)

        self.timer = self.create_timer(0.05, self.control_loop)

    def infra_callback(self, msg): self.go_signal = msg.data
    def hv_callback(self, msg):
        curr_time = self.get_clock().now()
        curr_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        if self.last_hv_pos is not None:
            dt = (curr_time - self.last_hv_time).nanoseconds / 1e9
            if dt > 0.001:
                raw_hv_v = np.linalg.norm(curr_pos - self.last_hv_pos) / dt
                self.hv_filtered_v = self.kf_hv_v.step(raw_hv_v)
        self.last_hv_pos, self.last_hv_time = curr_pos, curr_time

    def load_path_data(self):
        df = pd.read_csv(self.path_file, header=None)
        self.path = df.apply(pd.to_numeric, errors='coerce').dropna().values

    def pose_callback(self, msg):
        now = self.get_clock().now()
        pose_dt = (now - self.last_pose_time).nanoseconds / 1e9
        self.last_pose_time = now
        if pose_dt <= 0 or pose_dt > 0.2: pose_dt = 0.05
        q = msg.pose.orientation
        ry = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.raw_px, self.raw_py, self.raw_yaw = msg.pose.position.x, msg.pose.position.y, ry
        if self.kf_yaw.x is not None:
            while ry - self.kf_yaw.x > np.pi: ry -= 2*np.pi
            while ry - self.kf_yaw.x < -np.pi: ry += 2*np.pi
        if self.use_prediction:
            pred_yaw = self.current_motion_yaw + (self.last_omega * pose_dt)
            dx = self.current_v * np.cos(pred_yaw) * pose_dt
            dy = self.current_v * np.sin(pred_yaw) * pose_dt
        else: dx = dy = 0.0
        gate = abs(self.current_v * pose_dt) + 0.2
        self.filtered_pose = [self.kf_x.step(self.raw_px, dx, gate=gate), self.kf_y.step(self.raw_py, dy, gate=gate), self.kf_yaw.step(ry)]
        self.curr_pose = msg

    def actual_v_callback(self, msg): self.actual_v = msg.data
    def echo_callback(self, msg): self.echo_v, self.echo_w = msg.linear.x, msg.angular.z
    def allstate_callback(self, msg): self.raw_allstate = msg.data
    def battery_callback(self, msg): self.battery_voltage = msg.data

    def find_nearest_global(self, px, py):
        path_len = len(self.path)
        if self.prev_ni is None or (time.time() - self.start_time) < 5.0: indices = np.arange(path_len)
        else: indices = np.arange(self.prev_ni - 100, self.prev_ni + 100) % path_len
        search_path = self.path[indices]
        dists = np.sqrt(np.sum((search_path - [px, py])**2, axis=1))
        ni = indices[np.argmin(dists)]
        if (time.time() - self.start_time) > 5.0:
            if ni > path_len * 0.5: self.halfway_passed = True
            if self.halfway_passed and self.prev_ni > path_len * 0.9 and ni < path_len * 0.1:
                self.lap_count += 1; self.halfway_passed = False
                if self.lap_count >= 100 and self.finish_check_time is None: self.finish_check_time = time.time()
        self.prev_ni = ni
        return ni

    def get_control_metrics(self, px, py, ni):
        path_len = len(self.path)
        indices = [(ni + i) % path_len for i in range(-5, 6)]
        pts = self.path[indices]
        center = np.mean(pts, axis=0)
        norm_pts = pts - center
        cov = np.dot(norm_pts.T, norm_pts)
        val, vec = np.linalg.eigh(cov)
        tangent = vec[:, np.argmax(val)]
        path_yaw = np.arctan2(tangent[1], tangent[0])
        if np.dot(tangent, self.path[(ni + 1) % path_len] - self.path[ni]) < 0: path_yaw += np.pi
        dx, dy = px - self.path[ni][0], py - self.path[ni][1]
        cte = -np.sin(path_yaw)*dx + np.cos(path_yaw)*dy
        return path_yaw, cte

    def get_curvature(self, ni, window):
        p1, p2, p3 = self.path[ni], self.path[(ni+window//2)%len(self.path)], self.path[(ni+window)%len(self.path)]
        v1, v2 = p2 - p1, p3 - p2
        ang = (np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]) + np.pi) % (2*np.pi) - np.pi
        dist = np.linalg.norm(p3 - p1)
        return ang / dist if dist > 0.01 else 0.0

    def control_loop(self):
        if self.curr_pose is None or self.is_finished: return
        if self.finish_check_time and (time.time() - self.finish_check_time > 0.5):
            self.close_node(); return

        now = self.get_clock().now()
        dt = max(0.001, (now - self.last_time).nanoseconds / 1e9)
        self.last_time = now
        fpx, fpy, fyaw = self.filtered_pose

        if self.prev_filt_px is not None:
            dx, dy = fpx - self.prev_filt_px, fpy - self.prev_filt_py
            if np.sqrt(dx**2 + dy**2) > 0.02: self.current_motion_yaw = np.arctan2(dy, dx)
        else: self.current_motion_yaw = fyaw

        pred_px = fpx + (self.current_v * np.cos(self.current_motion_yaw) * dt)
        pred_py = fpy + (self.current_v * np.sin(self.current_motion_yaw) * dt)

        ni = self.find_nearest_global(pred_px, pred_py)
        path_yaw, cte = self.get_control_metrics(pred_px, pred_py, ni)
        curv_ff = self.get_curvature(ni, int(self.params['p_ff_window']))

        v_dead = self.params['p_steer_deadzone']
        e_v_cte = max(0.0, abs(cte) - v_dead)
        v_penalty = (abs(curv_ff) * self.params['p_v_curve_gain']) + (e_v_cte * self.params['p_v_cte_gain'])
        target_v = np.clip(self.params['p_v_max'] - v_penalty, self.params['p_v_min'], self.params['p_v_max'])

        # --- [수정된 정지 구역 및 즉각 정지 로직] ---
        dist_to_round = np.linalg.norm(np.array([fpx, fpy]) - self.roundabout_center)
        is_4way = (-3.6 <= fpx <= -0.9) and (-1.3 <= fpy <= 1.3)
        is_zone1 = (-3.6 <= fpx <= -1.4) and (1.4 <= fpy <= 2.6)
        is_zone2 = (-3.3 <= fpx <= -1.1) and (-2.6 <= fpy <= -1.4)

        stop_condition = not self.go_signal and ((1.1 < dist_to_round < 1.9) or is_4way or is_zone1 or is_zone2)

        if stop_condition:
            # 1. 정지 조건이 처음 발생했고, 현재 차가 움직이고 있다면 제동 시작
            if not self.is_active_braking and self.current_v > 0.0:
                self.is_active_braking = True
                self.brake_count = 10  # 모터의 10개 이동평균 필터를 상쇄하기 위한 횟수

            # 2. 제동 상태에 따른 속도 결정
            if self.is_active_braking:
                if self.brake_count > 0:
                    # 지정된 횟수만큼 역방향 속도 명령 (-0.05)
                    self.current_v = -0.05
                    self.brake_count -= 1
                else:
                    # 역방향 명령이 끝나면 속도를 0으로 유지
                    self.current_v = 0.0
            else:
                # 이미 멈춰있는 상태라면 0 유지
                self.current_v = 0.0

            # 제동 중에는 내부 속도 버퍼를 0으로 초기화하여 재출발 시 간섭 방지
            self.v_buffer = [0.0] * 10
            
            # 조향(Omega)은 기존 PID 로직 그대로 사용하여 라인 유지
            e_dead = 0.0 if abs(cte) < v_dead else cte - (np.sign(cte) * v_dead)
            self.error_integral = np.clip(self.error_integral + e_dead * dt, -1.0, 1.0)
            omega_pid = -((self.params['p_kp'] * e_dead) + (self.params['p_ki'] * self.error_integral))
            
            msg = Twist()
            msg.linear.x = float(self.current_v)
            msg.angular.z = float(np.clip(omega_pid, -1.5, 1.5))
            self.pub_accel.publish(msg)
            return

        else:
            # 주행 신호(go_signal)가 들어오면 제동 관련 플래그 초기화
            self.is_active_braking = False
            self.brake_count = 0

        # 정지 신호가 풀린 직후(current_v=0) 가속 지연 방지를 위해 버퍼 강제 업데이트
        if self.current_v <= 0.0 and self.go_signal:
            self.current_v = target_v
            self.v_buffer = [target_v] * 10
        # ------------------------------------------

        rsu_override = False
        if dist_to_round < 1.2 and self.hv_filtered_v > 0.1:
            target_v = float(max(self.hv_filtered_v, self.params['p_v_min']))
            rsu_override = True

        if rsu_override: # 위에서 처리한 current_v == 0 조건은 별도로 관리하므로 override만 체크
            self.current_v = target_v
            self.v_buffer = [target_v] * 10
        else:
            self.v_buffer.pop(0); self.v_buffer.append(target_v)
            v_smoothed = sum(self.v_buffer) / 10.0
            acc_lim = (self.params['p_v_accel'] if v_smoothed > self.current_v else self.params['p_v_decel']) * dt
            self.current_v = np.clip(v_smoothed, self.current_v - acc_lim, self.current_v + acc_lim)

        e_dead = 0.0 if abs(cte) < v_dead else cte - (np.sign(cte) * v_dead)
        self.error_integral = np.clip(self.error_integral + e_dead * dt, -1.0, 1.0)
        cte_d = (e_dead - self.last_error) / dt
        self.last_error = e_dead
        omega_pid = -((self.params['p_kp'] * e_dead) + (self.params['p_ki'] * self.error_integral) + (self.params['p_kd'] * cte_d))
        omega_ff = self.current_v * curv_ff * self.params['p_ff_gain']
        y_err = (self.current_motion_yaw - path_yaw + np.pi) % (2 * np.pi) - np.pi
        y_dead_rad = np.radians(self.params['p_yaw_deadzone'])
        y_err_f = 0.0 if abs(y_err) < y_dead_rad else y_err - (np.sign(y_err) * y_dead_rad)
        omega_yaw = -self.params['p_kyaw'] * y_err_f * (1.0 / (1.0 + abs(curv_ff) * 10.0))
        final_omega = np.clip(omega_pid + omega_ff + omega_yaw, -abs(self.current_v)*3.0, abs(self.current_v)*3.0)

        msg = Twist()
        msg.linear.x, msg.angular.z = float(self.current_v), float(final_omega)
        self.pub_accel.publish(msg)

        self.prev_filt_px, self.prev_filt_py = fpx, fpy
        self.last_diff = final_omega - self.last_omega
        self.last_omega = final_omega

    def close_node(self):
        self.is_finished = True
        for _ in range(10): self.pub_accel.publish(Twist()); time.sleep(0.01)
        os._exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.close_node(); rclpy.shutdown()

if __name__ == '__main__': main()