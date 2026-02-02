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

        # [기존 유지] 경로 설정 및 매핑
        self.base_path = "/home/njh/Desktop/" 
        self.id_map = { 22: 1 }
        self.cav_ids = list(self.id_map.keys())
        self.hv_ids = ['19', '20']

        self.get_logger().info("🔍 RSU: 경로 파일 로딩 및 실시간 교차 판정 모드 시작...")

        self.cars = {}
        for i in self.cav_ids:
            path_no = self.id_map[i]
            path = self.load_path(path_no)
            self.cars[f'CAV{i}'] = {
                'pos': None, 'path': path, 'entry_time': 0,
                'last_signal': True, 'current_zone': None,
                'in_roundabout': False,
                'min_ttc_record': 99.0,
                'rebound_released': False,
                'pub': self.create_publisher(Bool, f'/infra/CAV_{i}/go_signal', 10)
            }

        # [수정] HV 상태 구조 확장 (prev_timer_pos 추가)
        self.hvs = {
            f'HV{hid}': {
                'pos': None, 
                'prev_timer_pos': None,  # 0.05초 전 위치 저장용
                'vel': 0.0
            } for hid in self.hv_ids
        }

        self.zones = {
            "4way": {"x": [-3.6, -0.9], "y": [-1.3, 1.3]},
            "zone1": {"x": [-3.6, -1.4], "y": [1.4, 2.6]},
            "zone2": {"x": [-3.3, -1.1], "y": [-2.6, -1.4]}
        }
        self.round_center = np.array([1.67, 0.0])

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.VOLATILE, depth=10)

        for i in self.cav_ids:
            self.create_subscription(PoseStamped, f'/CAV_{i}',
                                     lambda msg, c_id=f'CAV{i}': self.pose_cb(msg, c_id), qos_profile)
        
        for hid in self.hv_ids:
            self.create_subscription(PoseStamped, f'/HV_{hid}',
                                     lambda msg, h_id=f'HV{hid}': self.hv_cb(msg, h_id), qos_profile)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("RSU: 0.05s HV 속도 계산 버전")

    def load_path(self, num):
        full_path = os.path.join(self.base_path, f'path3-{num}.csv')
        try:
            if not os.path.exists(full_path):
                self.get_logger().error(f"❌ 파일을 찾을 수 없음: {full_path}")
                return None
            data = pd.read_csv(full_path, header=None).values[:, :2]
            return data
        except Exception as e:
            self.get_logger().error(f"❌ 로드 중 오류: {str(e)}")
            return None

    def pose_cb(self, msg, car_id):
        self.cars[car_id]['pos'] = np.array([msg.pose.position.x, msg.pose.position.y])

    def hv_cb(self, msg, hv_id):
        # [수정] 콜백은 데이터 수신만 담당 (계산 로직 제거)
        self.hvs[hv_id]['pos'] = np.array([msg.pose.position.x, msg.pose.position.y])

    def control_loop(self):
        # --- [1단계: HV 속도 계산 (수정 포인트)] ---
        for hid, hv in self.hvs.items():
            curr_pos = hv['pos']
            prev_pos = hv['prev_timer_pos']

            if curr_pos is not None and prev_pos is not None:
                # 데이터가 갱신되었을 때만 계산 (Hold 로직: 같으면 이전 속도 유지)
                if not np.array_equal(curr_pos, prev_pos):
                    dt = 0.05  # 타이머 주기로 고정
                    dist = np.linalg.norm(curr_pos - prev_pos)
                    raw_vel = dist / dt
                    
                    # [상한선] 마하 1.0 방지 (Clamping)
                    raw_vel = min(raw_vel, 2.0)
                    
                    # [필터] 저주파 통과 필터 (LPF)
                    alpha = 0.15
                    if hv['vel'] == 0.0:
                        hv['vel'] = raw_vel
                    else:
                        hv['vel'] = (alpha * raw_vel) + ((1.0 - alpha) * hv['vel'])
            
            # 현재 위치를 다음 루프를 위해 저장
            if curr_pos is not None:
                hv['prev_timer_pos'] = curr_pos.copy()

        # --- [2단계: CAV 경로 교차 판단 (기존 유지)] ---
        active_cavs = [cid for cid, data in self.cars.items() if data['pos'] is not None]
        zone_queues = {name: [] for name in self.zones}
        current_crossing_status = {}

        for i in range(len(active_cavs)):
            for j in range(i + 1, len(active_cavs)):
                id1, id2 = active_cavs[i], active_cavs[j]
                p1, p2 = self.cars[id1]['path'], self.cars[id2]['path']
                pos1, pos2 = self.cars[id1]['pos'], self.cars[id2]['pos']

                is_cross = False
                if p1 is not None and p2 is not None:
                    idx1 = np.argmin(np.linalg.norm(p1 - pos1, axis=1))
                    idx2 = np.argmin(np.linalg.norm(p2 - pos2, axis=1))
                    future1 = p1[idx1 : idx1 + 300]
                    future2 = p2[idx2 : idx2 + 300]
                    for pt1 in future1[::10]:
                        if np.any(np.linalg.norm(future2 - pt1, axis=1) < 0.18):
                            is_cross = True
                            break
                current_crossing_status[(id1, id2)] = is_cross
                current_crossing_status[(id2, id1)] = is_cross

        # --- [3단계: FIFO 및 TTC/회전교차로 로직 (기존 유지)] ---
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

        for z_name in zone_queues:
            zone_queues[z_name].sort(key=lambda x: self.cars[x]['entry_time'])

        for cid in active_cavs:
            data = self.cars[cid]
            can_go = True
            reason = "Clear"
            dist_round = np.linalg.norm(data['pos'] - self.round_center)

            if dist_round < 1.45: data['in_roundabout'] = True
            if dist_round > 2.0:
                data['in_roundabout'] = False
                data['min_ttc_record'] = 99.0
                data['rebound_released'] = False

            if data['in_roundabout']:
                reason = "In-Process (Roundabout)"
            elif (1.0 <= dist_round <= 1.9):
                min_current_ttc = 99.0
                is_low_speed_hazard = False
                target_hv_id = "None"

                for hid, hv in self.hvs.items():
                    if hv['pos'] is not None:
                        dist_to_hv = np.linalg.norm(data['pos'] - hv['pos'])
                        if np.linalg.norm(hv['pos'] - self.round_center) < 1.4:
                            # [핵심] 여기서 수정된 hv['vel']이 사용됨
                            if hv['vel'] <= 0.5:
                                if dist_to_hv < 1.0:
                                    is_low_speed_hazard = True
                                    target_hv_id = hid
                                    reason = f"Low Speed: HV({hid}) at {hv['vel']:.2f}m/s"
                                    break
                            elif hv['vel'] > 0.1:
                                ttc = dist_to_hv / hv['vel']
                                if ttc < min_current_ttc:
                                    min_current_ttc = ttc
                                    target_hv_id = hid

                if is_low_speed_hazard:
                    can_go = False
                elif min_current_ttc < 99.0:
                    if data['rebound_released']: can_go = True
                    else:
                        if data['last_signal'] == True:
                            if min_current_ttc < 1.6:
                                can_go = False
                                data['min_ttc_record'] = min_current_ttc
                        else:
                            if min_current_ttc > data['min_ttc_record'] + 0.3:
                                can_go = True
                                data['rebound_released'] = True
                            elif min_current_ttc < data['min_ttc_record']:
                                data['min_ttc_record'] = min_current_ttc
                            else: can_go = False

            # FIFO 정지 제어
            if can_go and data['current_zone']:
                q = zone_queues[data['current_zone']]
                idx = q.index(cid)
                if idx > 0:
                    for prev_idx in range(idx):
                        front_cid = q[prev_idx]
                        dist = np.linalg.norm(data['pos'] - self.cars[front_cid]['pos'])
                        is_crossing = current_crossing_status.get((cid, front_cid), True)
                        safe_margin = 1.5 if is_crossing else 0.18
                        if dist < safe_margin:
                            can_go = False; break

            if data['last_signal'] != can_go:
                status = "🟢 GO" if can_go else "🔴 STOP"
                self.get_logger().info(f"📢 [{cid}] {status} | {reason} | HV_Vel: {hv['vel']:.2f}")
                data['last_signal'] = can_go

            msg = Bool(); msg.data = can_go
            data['pub'].publish(msg)

def main():
    rclpy.init()
    node = PathAwareRSU()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()