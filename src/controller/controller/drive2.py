#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
import pandas as pd
import numpy as np
import time
import os
import math

# [1. 유틸리티 클래스] - 칼만 필터
class SimpleKalman:
    def __init__(self, q=0.1, r=0.1):
        self.q, self.r = q, r
        self.x, self.p = None, 1.0

    def step(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        p_prior = self.p + self.q
        k_gain = p_prior / (p_prior + self.r)
        self.x = self.x + k_gain * (measurement - self.x)
        self.p = (1 - k_gain) * p_prior
        return self.x

# [2. 통합 주행 노드] - UnifiedFollower (로그 제거 버전)
class UnifiedFollower(Node):
    def __init__(self):
        super().__init__('unified_follower')
        
        # 하드코딩 파라미터 (현장 수정용)
        self.car_id = 2  
        home_dir = os.path.expanduser('~')
        self.path_file = os.path.join(home_dir, f'kmc_ws/src/controller/path/path3-{self.car_id}.csv')

        self.params = {
            "p_kp": 3.0, "p_ki": 1.5, "p_kd": 3.0, "p_kyaw": 1.0,
            "p_ff_gain": 2.0, "p_gamma": 1.0,
            "p_v_max": 1.0, "p_v_min": 0.5, "p_v_accel": 0.5, "p_v_decel": 1.0,
            "p_v_curve_gain": 0.2, "p_v_steer_gain": 0.0, "p_v_cte_gain": 0.1,
            "p_kf_q": 0.1, "p_kf_r": 0.1, "p_steer_deadzone": 0.008,
            "p_ff_window": 10
        }

        # 상태 및 인프라 변수
        self.current_v = 0.0
        self.filtered_pose = [0.0, 0.0, 0.0]
        self.prev_ni = 0
        self.error_integral, self.last_error = 0.0, 0.0
        self.last_time = self.get_clock().now()
        self.start_time = time.time()
        self.is_finished = False
        self.prev_filt_px, self.prev_filt_py = None, None
        self.go_signal = True

        # HV 및 교차로 변수
        self.intersection_centers = [np.array([1.67, 0.0])]
        self.hv_velocity = 0.0
        self.last_hv_pos, self.last_hv_time = None, None
        self.hv_vel_count = 0

        # 경로 데이터 로드
        try:
            df = pd.read_csv(self.path_file, header=None)
            self.path = df.apply(pd.to_numeric, errors='coerce').dropna().values
            self.get_logger().info(f"✅ CAV{self.car_id} 경로 로드 성공")
        except Exception as e:
            self.get_logger().error(f"❌ 경로 로드 실패: {e}")
            self.path = np.array([[0,0], [1,0]])

        # 필터 설정
        self.kf_x = SimpleKalman(self.params['p_kf_q'], self.params['p_kf_r'])
        self.kf_y = SimpleKalman(self.params['p_kf_q'], self.params['p_kf_r'])
        self.kf_yaw = SimpleKalman(0.2, 0.01)

        # QoS 및 통신 (실차 규격)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub_ctrl = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub_pose = self.create_subscription(PoseStamped, f'/CAV_0{self.car_id}', self.pose_callback, qos)
        self.sub_infra = self.create_subscription(Bool, f'/infra/CAV_0{self.car_id}/go_signal', self.infra_callback, qos)
        self.sub_hv = self.create_subscription(PoseStamped, '/HV_19_pose', self.hv_callback, qos)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.curr_pose = None

    def infra_callback(self, msg):
        self.go_signal = msg.data

    def hv_callback(self, msg):
        curr_time = self.get_clock().now()
        curr_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        if self.last_hv_pos is not None:
            dt = (curr_time - self.last_hv_time).nanoseconds / 1e9
            if dt > 0.01:
                calc_v = np.linalg.norm(curr_pos - self.last_hv_pos) / dt
                if self.hv_vel_count < 10:
                    self.hv_velocity = (self.hv_velocity * self.hv_vel_count + calc_v) / (self.hv_vel_count + 1)
                    self.hv_vel_count += 1
                else:
                    self.hv_velocity = calc_v
        self.last_hv_pos, self.last_hv_time = curr_pos, curr_time

    def pose_callback(self, msg):
        raw_px, raw_py = msg.pose.position.x, msg.pose.position.y
        q = msg.pose.orientation
        raw_yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        if self.kf_yaw.x is not None:
            while raw_yaw - self.kf_yaw.x > np.pi: raw_yaw -= 2*np.pi
            while raw_yaw - self.kf_yaw.x < -np.pi: raw_yaw += 2*np.pi
        self.filtered_pose = [self.kf_x.step(raw_px), self.kf_y.step(raw_py), self.kf_yaw.step(raw_yaw)]
        self.curr_pose = msg

    def find_nearest_global(self, px, py):
        path_len = len(self.path)
        dists = np.sqrt(np.sum((self.path - [px, py])**2, axis=1))
        
        # 300포인트 지역 페널티 로직
        if self.prev_ni is not None and time.time() - self.start_time > 5.0:
            look_range = 300
            indices = np.arange(path_len)
            diff = np.abs(indices - self.prev_ni)
            diff = np.minimum(diff, path_len - diff)
            dists += np.where(diff > look_range, 0.2, 0.0)

        ni = np.argmin(dists)
        self.prev_ni = ni
        return ni

    def get_control_metrics(self, px, py, ni):
        start, end = max(0, ni - 5), min(len(self.path), ni + 6)
        pts = self.path[start:end]
        center = np.mean(pts, axis=0)
        norm_pts = pts - center
        cov = np.dot(norm_pts.T, norm_pts)
        val, vec = np.linalg.eigh(cov)
        tangent = vec[:, np.argmax(val)]
        path_yaw = np.arctan2(tangent[1], tangent[0])
        if ni < len(self.path)-1 and np.dot(tangent, self.path[ni+1]-self.path[ni]) < 0:
            path_yaw += np.pi
        dx, dy = px - self.path[ni][0], py - self.path[ni][1]
        cte = -np.sin(path_yaw)*dx + np.cos(path_yaw)*dy
        return path_yaw, cte

    def get_curvature(self, ni, window):
        end = min(ni + window, len(self.path) - 1)
        if ni >= end: return 0.0
        p1, p2, p3 = self.path[ni], self.path[(ni+end)//2], self.path[end]
        v1, v2 = p2-p1, p3-p2
        ang = (np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]) + np.pi) % (2*np.pi) - np.pi
        dist = np.linalg.norm(self.path[end] - self.path[ni])
        return ang / dist if dist > 0.01 else 0.0

    def control_loop(self):
        if self.curr_pose is None or self.is_finished: return
        now = self.get_clock().now()
        dt = max(0.001, (now - self.last_time).nanoseconds / 1e9)
        self.last_time = now
        filt_px, filt_py, yaw = self.filtered_pose

        # 인프라 신호에 따른 정지
        dist_to_round = np.linalg.norm(np.array([filt_px, filt_py]) - self.intersection_centers[0])
        is_4way = (-3.5 <= filt_px <= -0.9) and (-1.3 <= filt_py <= 1.1)
        is_zone1 = (-3.3 <= filt_px <= -1.4) and (1.6 <= filt_py <= 2.6)
        is_zone2 = (-3.3 <= filt_px <= -1.4) and (-2.6 <= filt_py <= -1.76)

        if not self.go_signal and ((1.1 < dist_to_round < 1.5) or is_4way or is_zone1 or is_zone2):
            self.current_v = 0.0
            self.pub_ctrl.publish(Twist())
            return

        # 위치 예측 및 지표 산출
        pred_px = filt_px + (self.current_v * np.cos(yaw) * 0.05)
        pred_py = filt_py + (self.current_v * np.sin(yaw) * 0.05)
        ni = self.find_nearest_global(pred_px, pred_py)
        path_yaw, cte = self.get_control_metrics(pred_px, pred_py, ni)
        curv_ff = self.get_curvature(ni, int(self.params['p_ff_window']))

        # 속도 및 조향 제어 (HV 추종 로직 포함)
        if dist_to_round < 1.2:
            self.current_v = float(max(self.hv_velocity, 0.01))
        else:
            v_penalty = (abs(curv_ff) * self.params['p_v_curve_gain']) + (abs(cte) * self.params['p_v_cte_gain'])
            target_v = np.clip(self.params['p_v_max'] - v_penalty, self.params['p_v_min'], self.params['p_v_max'])
            self.current_v = float(target_v)

        deadzone = self.params['p_steer_deadzone']
        e_dead = 0.0 if abs(cte) < deadzone else cte - (np.sign(cte) * deadzone)
        self.error_integral = np.clip(self.error_integral + e_dead * dt, -1.0, 1.0)
        omega_pid = -((self.params['p_kp'] * e_dead) + (self.params['p_ki'] * self.error_integral) + (self.params['p_kd'] * (e_dead - self.last_error)/dt))
        self.last_error = e_dead

        omega_ff = self.current_v * curv_ff * self.params['p_ff_gain']
        final_omega = np.clip((omega_pid + omega_ff) * self.params['p_gamma'], -6.0, 6.0)

        msg = Twist()
        msg.linear.x, msg.angular.z = float(self.current_v), float(final_omega)
        self.pub_ctrl.publish(msg)

    def stop_vehicle(self):
        msg = Twist()
        for _ in range(5): self.pub_ctrl.publish(msg); time.sleep(0.05)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_vehicle()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()