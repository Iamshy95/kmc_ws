#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import pandas as pd
import numpy as np
import os
import time

class PathAwareRSU(Node):
    def __init__(self):
        super().__init__('integrated_rsu_manager')
        
        # [수정] 경로 파일 위치를 실차 환경 하드코딩으로 변경
        home_dir = os.path.expanduser('~')
        self.base_path = os.path.join(home_dir, 'kmc_ws/src/controller/path')
        
        self.cav_ids = [1, 2, 3, 4]
        self.hv_ids = ['19', '20']

        self.cars = {}
        for i in self.cav_ids:
            # 문제 3번 전용 경로(path3-x.csv) 로드
            path = self.load_path(i)
            self.cars[f'CAV{i}'] = {
                'pos': None, 'path': path, 'entry_time': 0,
                'last_signal': True, 'current_zone': None,
                'in_roundabout': False,
                'min_ttc_record': 99.0,
                'rebound_released': False,
                # [메시지 형식] Bool 타입 go_signal 발행
                'pub': self.create_publisher(Bool, f'/infra/CAV_0{i}/go_signal', 10)
            }

        self.hvs = {f'HV{hid}': {'pos': None, 'vel': 0.0} for hid in self.hv_ids}

        self.hv_speed_samples = {f'HV{hid}': [] for hid in self.hv_ids}
        self.hv_fixed_speeds = {f'HV{hid}': 0.0 for hid in self.hv_ids}
        self.is_speed_learned = {f'HV{hid}': False for hid in self.hv_ids}
        self.MAX_SAMPLES = 20

        # 구역 설정 (사용자님 로직 유지)
        self.zones = {
            "4way": {"x": [-3.5, -0.9], "y": [-1.1, 1.1]},
            "zone1": {"x": [-3.3, -1.4], "y": [1.6, 2.6]},
            "zone2": {"x": [-3.3, -1.4], "y": [-2.6, -1.76]}
        }
        self.round_center = np.array([1.67, 0.0])

        # [메시지 형식] 실차 모캡 데이터 수신을 위한 QoS 설정
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.VOLATILE, depth=10)

        # CAV 및 HV 구독 설정
        for i in self.cav_ids:
            self.create_subscription(PoseStamped, f'/CAV_0{i}',
                                     lambda msg, c_id=f'CAV{i}': self.pose_cb(msg, c_id), qos_profile)
        for hid in self.hv_ids:
            self.create_subscription(PoseStamped, f'/HV_{hid}',
                                     lambda msg, h_id=f'HV{hid}': self.hv_cb(msg, h_id), qos_profile)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("🚀 RSU: 문제 3번 전용 멀티 CAV 제어 노드 가동")

    def load_path(self, i):
        try:
            # 사용자님 요청대로 path3-{i}.csv 형식으로 로드
            path_file = os.path.join(self.base_path, f'path3-{i}.csv')
            return pd.read_csv(path_file, header=None).values[:, :2]
        except Exception as e:
            self.get_logger().warn(f"⚠️ CAV{i} 경로 파일 로드 실패: {e}")
            return None

    def pose_cb(self, msg, car_id):
        self.cars[car_id]['pos'] = np.array([msg.pose.position.x, msg.pose.position.y])

    def hv_cb(self, msg, hv_id):
        curr_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        if self.hvs[hv_id]['pos'] is not None:
            if not self.is_speed_learned[hv_id]:
                dt = 0.05
                current_vel = np.linalg.norm(curr_pos - self.hvs[hv_id]['pos']) / dt
                if current_vel > 0.1:
                    self.hv_speed_samples[hv_id].append(current_vel)
                if len(self.hv_speed_samples[hv_id]) >= self.MAX_SAMPLES:
                    self.hv_fixed_speeds[hv_id] = np.mean(self.hv_speed_samples[hv_id])
                    self.is_speed_learned[hv_id] = True
            
            if self.is_speed_learned[hv_id]:
                self.hvs[hv_id]['vel'] = self.hv_fixed_speeds[hv_id]
            else:
                self.hvs[hv_id]['vel'] = np.linalg.norm(curr_pos - self.hvs[hv_id]['pos']) / 0.05
        self.hvs[hv_id]['pos'] = curr_pos

    def is_path_crossing(self, id1, id2):
        p1_full, p2_full = self.cars[id1]['path'], self.cars[id2]['path']
        pos1, pos2 = self.cars[id1]['pos'], self.cars[id2]['pos']
        if p1_full is None or p2_full is None: return True
        idx1 = np.argmin(np.linalg.norm(p1_full - pos1, axis=1))
        idx2 = np.argmin(np.linalg.norm(p2_full - pos2, axis=1))
        future1 = p1_full[idx1 : idx1 + 80]
        future2 = p2_full[idx2 : idx2 + 80]
        for pt1 in future1:
            if np.any(np.linalg.norm(future2 - pt1, axis=1) < 0.18): return True
        return False

    def control_loop(self):
        active_cavs = [cid for cid, data in self.cars.items() if data['pos'] is not None]
        zone_queues = {name: [] for name in self.zones}

        # 구역 진입 확인 및 FIFO 큐 생성
        for cid in active_cavs:
            data = self.cars[cid]
            x, y = data['pos'][0], data['pos'][1]
            in_zone = False
            for z_name, limit in self.zones.items():
                if (limit['x'][0] <= x <= limit['x'][1]) and (limit['y'][0] <= y <= limit['y'][1]):
                    if data['entry_time'] == 0: data['entry_time'] = time.time()
                    zone_queues[z_name].append(cid)
                    data['current_zone'] = z_name
                    in_zone = True; break
            if not in_zone: data['entry_time'] = 0; data['current_zone'] = None

        # 진입 시간 순으로 큐 정렬
        for z_name in zone_queues:
            zone_queues[z_name].sort(key=lambda x: self.cars[x]['entry_time'])

        # 각 CAV별 제어 신호 결정 (TTC + FIFO 로직 유지)
        for cid in active_cavs:
            data = self.cars[cid]
            can_go = True
            reason = "Clear"
            dist_round = np.linalg.norm(data['pos'] - self.round_center)

            if dist_round < 1.3:
                data['in_roundabout'] = True
            if dist_round > 1.8:
                data['in_roundabout'] = False
                data['min_ttc_record'] = 99.0
                data['rebound_released'] = False

            if data['in_roundabout']:
                can_go = True
                reason = "In-Process"
            else:
                if (1.0 <= dist_round <= 1.6):
                    min_current_ttc = 99.0
                    is_low_speed_hazard = False

                    for hid, hv in self.hvs.items():
                        if hv['pos'] is not None:
                            dist_to_hv = np.linalg.norm(data['pos'] - hv['pos'])
                            if np.linalg.norm(hv['pos'] - self.round_center) < 1.4:
                                if hv['vel'] <= 0.5:
                                    if dist_to_hv < 0.8:
                                        is_low_speed_hazard = True
                                        reason = f"Low Speed Hazard ({hid})"
                                        break
                                elif hv['vel'] > 0.1:
                                    ttc = dist_to_hv / hv['vel']
                                    if ttc < min_current_ttc: min_current_ttc = ttc

                    if is_low_speed_hazard:
                        can_go = False
                        data['rebound_released'] = False
                    elif min_current_ttc < 99.0:
                        if data['rebound_released']:
                            can_go = True
                        else:
                            if data['last_signal'] == True:
                                if min_current_ttc < 1.5:
                                    can_go = False; reason = f"TTC Hazard: {min_current_ttc:.2f}s"
                                    data['min_ttc_record'] = min_current_ttc
                            else:
                                if min_current_ttc < data['min_ttc_record']: data['min_ttc_record'] = min_current_ttc
                                if min_current_ttc > data['min_ttc_record'] + 0.4:
                                    can_go = True; data['rebound_released'] = True
                                else:
                                    can_go = False; reason = "Waiting Rebound"
                else:
                    data['rebound_released'] = False

                # FIFO 로직
                if can_go and data['current_zone']:
                    z_name = data['current_zone']
                    q = zone_queues[z_name]
                    idx = q.index(cid)

                    if idx > 0:
                        for prev_idx in range(idx):
                            front_cid = q[prev_idx]
                            if self.cars[front_cid]['pos'] is not None:
                                dist = np.linalg.norm(data['pos'] - self.cars[front_cid]['pos'])
                                is_crossing = self.is_path_crossing(cid, front_cid)
                                safe_margin = 1.5 if is_crossing else 0.5
                                if dist < safe_margin:
                                    can_go = False
                                    reason = f"FIFO Waiting ({front_cid})"
                                    break

            if data['last_signal'] != can_go:
                status = "🟢 GO" if can_go else "🔴 STOP"
                self.get_logger().info(f"📢 [{cid}] {status} | {reason}")
                data['last_signal'] = can_go

            # 신호 발행
            msg = Bool(); msg.data = can_go; data['pub'].publish(msg)

def main():
    rclpy.init(); node = PathAwareRSU(); rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()