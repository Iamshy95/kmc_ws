import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import pandas as pd
import numpy as np
import os
import time

class AdvancedKalman:
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


class PathAwareRSU(Node):
    def __init__(self):
        super().__init__('path_aware_rsu')

        # [1] CSV 로그 폴더 및 타임스탬프 파일명 설정
        self.log_dir = "/home/njh/Desktop/ros_bags"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.log_dir, f"rsu_log_{timestamp}.csv")

        self.f_log = open(self.log_path, 'w', encoding='utf-8')
        self.f_log.write("Time,ID,X,Y, Siganl, can_go,TTC,Vel\n")

        self.pub_total_log = self.create_publisher(String, '/infra/total_status_log', 10)

        self.base_path = "/home/njh/Desktop"
        self.id_map = {3: 1, 22: 2, 39: 3, 9: 4}
        self.cav_ids = list(self.id_map.keys())
        self.hv_ids = ['19', '20']

        self.get_logger().info("RSU: loading paths...")

        self.cars = {}
        for i in self.cav_ids:
            formatted_id = f"{i:02d}"
            path_no = self.id_map[i]  # 1~4
            path = self.load_path(path_no)
            self.cars[f'CAV{formatted_id}'] = {
                'pos': None,
                'vel': 0.0,
                'path': path,
                'path_id': path_no,  # roundabout 각도필터용
                'entry_time': 0,
                'last_signal': True,
                'current_zone': None,
                'in_roundabout': False,
                'min_ttc_record': 99.0,
                'min_dist_record': 999.0,
                'rebound_released': False,
                'current_ttc': 99.0,
                'pub': self.create_publisher(Bool, f'/infra/CAV_{formatted_id}/go_signal', 10),
            }

        loaded_count = sum(1 for c in self.cars.values() if c['path'] is not None)
        self.get_logger().info(f"Paths loaded: {loaded_count}/{len(self.cav_ids)}")

        # [수정 후] 이렇게 바꿔주세요!
        self.hvs = {}
        for hid in self.hv_ids:
            self.hvs[f'HV{hid}'] = {
                'pos': None,
                'time': None,              # 최신 수신 시각
                'last_calc_pos': None,     # 계산에 사용된 마지막 위치
                'last_calc_time': None,    # 계산에 사용된 마지막 시각
                'vel': 0.0,
                'kalman': AdvancedKalman(q=0.1, r=0.1), # 차량별 전용 필터
                'ma_buffer': []            # 차량별 전용 MA10 버퍼
            }

        self.zones = {
            "4way":  {"x": [-3.6, -0.9], "y": [-1.6, 1.6]},
            "zone1": {"x": [-3.6, -1.4], "y": [1.4, 2.6]},
            "zone2": {"x": [-3.3, -1.1], "y": [-2.6, -1.4]},
        }
        self.round_center = np.array([1.67, 0.0])

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        for i in self.cav_ids:
            fid = f"{i:02d}"
            self.create_subscription(
                PoseStamped, f'/CAV_{fid}',
                lambda msg, c_id=f'CAV{fid}': self.pose_cb(msg, c_id),
                qos_profile
            )
            self.create_subscription(
                Twist, f'/CAV_{fid}/cmd_vel',
                lambda msg, c_id=f'CAV{fid}': self.cav_vel_cb(msg, c_id),
                10
            )

        for hid in self.hv_ids:
            self.create_subscription(
                PoseStamped, f'/HV_{hid}',
                lambda msg, h_id=f'HV{hid}': self.hv_cb(msg, h_id),
                qos_profile
            )

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("RSU ready.")

    def load_path(self, num):
        full_path = os.path.join(self.base_path, f'path3-{num}.csv')
        try:
            if not os.path.exists(full_path):
                self.get_logger().error(f"File not found: {full_path}")
                return None
            data = pd.read_csv(full_path, header=None).values[:, :2]
            return data
        except Exception as e:
            self.get_logger().error(f"Load error ({full_path}): {str(e)}")
            return None

    def pose_cb(self, msg, car_id):
        self.cars[car_id]['pos'] = np.array([msg.pose.position.x, msg.pose.position.y])

    def cav_vel_cb(self, msg, car_id):
        self.cars[car_id]['vel'] = msg.linear.x

    def hv_cb(self, msg, hv_id):
        # 위치 저장
        self.hvs[hv_id]['pos'] = np.array([msg.pose.position.x, msg.pose.position.y])
        # [추가] 시간 저장 (초 단위 변환)
        self.hvs[hv_id]['time'] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def control_loop(self):
        now_ts = self.get_clock().now().nanoseconds / 1e9

        # [수정된 HV 속도 계산 로직]
        for hid, hv in self.hvs.items():
            curr_pos = hv['pos']
            curr_time = hv['time']
            
            # 데이터가 있고, 이전에 계산한 적이 있을 때만 계산 시도
            if curr_pos is not None and hv['last_calc_time'] is not None:
                actual_dt = curr_time - hv['last_calc_time']

                # dt가 0.02초(20ms)보다 클 때만 계산 (너무 빠른 중복 계산 방지)
                if actual_dt > 0.02:
                    dist = np.linalg.norm(curr_pos - hv['last_calc_pos'])
                    raw_vel = min(dist / actual_dt, 2.0) # 2.0m/s 클램핑

                    # 1. 칼만 필터
                    kf_v = hv['kalman'].step(raw_vel)

                    # 2. MA10 (이동 평균)
                    hv['ma_buffer'].append(kf_v)
                    if len(hv['ma_buffer']) > 10:
                        hv['ma_buffer'].pop(0)
                    
                    # 최종 속도 저장
                    hv['vel'] = sum(hv['ma_buffer']) / len(hv['ma_buffer'])

                    # 계산 기준점 업데이트
                    hv['last_calc_pos'] = curr_pos.copy()
                    hv['last_calc_time'] = curr_time
            
            # 초기화 로직: 처음 데이터를 받았을 때 기준점 설정
            elif curr_pos is not None and hv['last_calc_time'] is None:
                hv['last_calc_pos'] = curr_pos.copy()
                hv['last_calc_time'] = curr_time
            
            # [로그 기록용]
            if curr_pos is not None:
                # (로그 찍는 코드는 그대로 유지)
                hv_row = f"{now_ts:.3f},{hid},{curr_pos[0]:.3f},{curr_pos[1]:.3f},None,None,{hv['vel']:.2f}\n"
                self.f_log.write(hv_row)

        active_cavs = [cid for cid, data in self.cars.items() if data['pos'] is not None]
        zone_queues = {name: [] for name in self.zones}
        current_crossing_status = {}

        # 교차 판단
        for i in range(len(active_cavs)):
            for j in range(i + 1, len(active_cavs)):
                id1, id2 = active_cavs[i], active_cavs[j]
                p1, p2 = self.cars[id1]['path'], self.cars[id2]['path']
                pos1, pos2 = self.cars[id1]['pos'], self.cars[id2]['pos']
                is_cross = False
                if p1 is not None and p2 is not None:
                    idx1 = np.argmin(np.linalg.norm(p1 - pos1, axis=1))
                    idx2 = np.argmin(np.linalg.norm(p2 - pos2, axis=1))
                    future1, future2 = p1[idx1: idx1 + 300], p2[idx2: idx2 + 300]
                    for pt1 in future1[::10]:
                        if np.any(np.linalg.norm(future2 - pt1, axis=1) < 0.18):
                            is_cross = True
                            break
                current_crossing_status[(id1, id2)] = is_cross
                current_crossing_status[(id2, id1)] = is_cross

        # Zone 큐
        for cid in active_cavs:
            data = self.cars[cid]
            x, y = data['pos'][0], data['pos'][1]
            in_zone = False
            for z_name, limit in self.zones.items():
                if (limit['x'][0] <= x <= limit['x'][1]) and (limit['y'][0] <= y <= limit['y'][1]):
                    if data['entry_time'] == 0:
                        data['entry_time'] = time.time()
                    zone_queues[z_name].append(cid)
                    data['current_zone'] = z_name
                    in_zone = True
                    break
            if not in_zone:
                data['entry_time'] = 0
                data['current_zone'] = None

        for z_name in zone_queues:
            zone_queues[z_name].sort(key=lambda x: self.cars[x]['entry_time'])

        # CAV 판단
        for cid in active_cavs:
            data = self.cars[cid]
            can_go = True
            reason = "Clear (No Hazard)"

            dist_round = np.linalg.norm(data['pos'] - self.round_center)

            # roundabout 상태 업데이트
            if dist_round < 1.3:
                data['in_roundabout'] = True
            if dist_round > 2.0:
                data['in_roundabout'] = False
                data['min_ttc_record'] = 99.0
                data['min_dist_record'] = 999.0
                data['rebound_released'] = False

            data['current_ttc'] = 99.0

            if data['in_roundabout']:
                can_go = True
                reason = "In-Process (Inside Roundabout)"

            elif (1.0 <= dist_round <= 2.0):
                # ====== 회전교차로 진입부 HV 위험 판단 (TTC OR 거리, 해제도 OR) ======
                min_ttc = 99.0
                min_dist = 999.0
                target_hv = "None"

                for hid, hv in self.hvs.items():
                    if hv['pos'] is None:
                        continue

                    # 각도 섹터 필터
                    v_hv = hv['pos'] - self.round_center
                    hv_deg = np.degrees(np.arctan2(v_hv[1], v_hv[0])) % 360
                    is_in_sector = False

                    # 네가 현재 넣은 섹터 조건 그대로 유지
                    if data['path_id'] in [1, 4]:
                        if 50 <= hv_deg <= 230:
                            is_in_sector = True
                    elif data['path_id'] in [2, 3]:
                        if hv_deg >= 320 or hv_deg <= 140:
                            is_in_sector = True

                    if not is_in_sector:
                        continue

                    # (원 안쪽 제한도 네 기존 로직 유지)
                    if np.linalg.norm(v_hv) >= 1.4:
                        continue

                    dist_hv = np.linalg.norm(data['pos'] - hv['pos'])

                    # min_dist 갱신
                    if dist_hv < min_dist:
                        min_dist = dist_hv
                        target_hv = hid

                    # min_ttc 갱신
                    if hv['vel'] > 0.01:
                        ttc = dist_hv / hv['vel']
                        if ttc < min_ttc:
                            min_ttc = ttc
                            target_hv = hid

                data['current_ttc'] = min_ttc

                hazard_now = (min_ttc < 2.6) or (min_dist < 1.3)

                if data['rebound_released']:
                    can_go = True
                    reason = f"Passing: HV({target_hv}) TTC:{min_ttc:.2f}s Dist:{min_dist:.2f}m"

                elif data['last_signal'] and hazard_now:
                    can_go = False
                    data['min_ttc_record'] = min_ttc
                    data['min_dist_record'] = min_dist
                    reason = f"Hazard: HV({target_hv}) TTC:{min_ttc:.2f}s Dist:{min_dist:.2f}m"

                elif not data['last_signal']:
                    data['min_ttc_record'] = min(data['min_ttc_record'], min_ttc)
                    data['min_dist_record'] = min(data['min_dist_record'], min_dist)

                    ttc_recovered = (min_ttc > data['min_ttc_record'] + 0.8)
                    dist_recovered = (min_dist >= 0.8)

                    if ttc_recovered and dist_recovered:
                        can_go = True
                        data['rebound_released'] = True
                        reason = (f"Released: HV({target_hv}) "
                                  f"MinTTC:{data['min_ttc_record']:.2f}s "
                                  f"MinDist:{data['min_dist_record']:.2f}m "
                                  f"NowTTC:{min_ttc:.2f}s NowDist:{min_dist:.2f}m")
                    else:
                        can_go = False
                        reason = f"Waiting: HV({target_hv}) TTC:{min_ttc:.2f}s Dist:{min_dist:.2f}m"

            else:
                # FIFO 구간(2번 코드 그대로)
                data['rebound_released'] = False
                if can_go and data['current_zone']:
                    q = zone_queues[data['current_zone']]
                    idx = q.index(cid)
                    if idx > 0:
                        for prev_idx in range(idx):
                            front_cid = q[prev_idx]
                            if self.cars[front_cid]['pos'] is not None:
                                dist = np.linalg.norm(data['pos'] - self.cars[front_cid]['pos'])
                                is_crossing = current_crossing_status.get((cid, front_cid), True)
                                safe_margin = 2.0 if is_crossing else 0.18
                                if dist < safe_margin:
                                    can_go = False
                                    break

            # 터미널 로그
            if data['last_signal'] != can_go:
                status = "GO" if can_go else "STOP"
                current_q = zone_queues.get(data['current_zone'], [])
                log_msg = f"[{cid}] {status} | {reason} | Zone: {data['current_zone']} | Q: {current_q}"
                self.get_logger().info(log_msg)
                data['last_signal'] = can_go

            # CSV 기록
            cav_row = (
                f"{now_ts:.3f},{cid},{data['pos'][0]:.3f},{data['pos'][1]:.3f},"
                f"{'GO' if can_go else 'STOP'},{data['current_ttc']:.2f},{data['vel']:.2f}\n"
            )
            self.f_log.write(cav_row)

            # 신호 전송
            msg = Bool()
            msg.data = can_go
            data['pub'].publish(msg)

        self.f_log.flush()

    def destroy_node(self):
        if hasattr(self, 'f_log'):
            self.f_log.close()
            self.get_logger().info(f"Log saved: {self.log_path}")
        super().destroy_node()


def main():
    rclpy.init()
    node = PathAwareRSU()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
