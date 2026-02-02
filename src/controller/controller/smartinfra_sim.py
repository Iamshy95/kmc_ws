#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Accel
from std_msgs.msg import Float32
import numpy as np
import json
import os
import csv
import time
from datetime import datetime

# ==============================================================================
# [1. 유틸리티]
# ==============================================================================
class AdvancedKalman:
    def __init__(self, q=0.05, r=0.1):
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

class OBB:
    def __init__(self, x, y, yaw, length, width):
        self.center = np.array([x, y])
        self.extents = np.array([length/2, width/2])
        self.dir_x = np.array([np.cos(yaw), np.sin(yaw)])
        self.dir_y = np.array([-np.sin(yaw), np.cos(yaw)])

# 클래스 밖으로 빼야 NameError가 안 납니다.
def check_obb_collision(obb1, obb2):
    L = obb2.center - obb1.center
    axes = [obb1.dir_x, obb1.dir_y, obb2.dir_x, obb2.dir_y]
    for axis in axes:
        dist = abs(np.dot(L, axis))
        proj1 = obb1.extents[0] * abs(np.dot(obb1.dir_x, axis)) + \
                obb1.extents[1] * abs(np.dot(obb1.dir_y, axis))
        proj2 = obb2.extents[0] * abs(np.dot(obb2.dir_x, axis)) + \
                obb2.extents[1] * abs(np.dot(obb2.dir_y, axis))
        if dist > (proj1 + proj2): return False
    return True

# ==============================================================================
# [2. 스마트인프라 매니저]
# ==============================================================================
class SmartInfraManager(Node):
    def __init__(self):
        super().__init__('smart_infra_manager')
        
        # --- 핵심 파라미터 (사용자 지침 반영) ---
        self.SAMPLING_INTERVAL = 0.1  # 10cm 간격 샘플링
        self.TIME_MARGIN = 0.5        # 충돌 판단 시간차 마진 (0.5s)
        self.SAFETY_GAP = 0.5         # Yielder 감속 목표 마진 (0.5s)
        self.INSPECTION_DIST = 2.0    # 2.0m 이내 조사
        self.LOOKAHEAD_TIME = 1.0     # 1.0초 주시 시간
        self.PATH_OFFSET = 20         # 20pts 오프셋
        self.V_MAX = 1.0  # 인프라가 허용하는 전역 최고 속도 (필요에 따라 수정)
        self.V_MIN_LIMIT = 0.2        # 최저 속도 제한 (이하는 정지)
        self.CAR_L, self.CAR_W = 0.33, 0.16
        self.CORRIDOR_W = 0.20  # 복도 생성 시 사용할 더 넓은 폭
        
        self.cav_ids = [1, 2, 3, 4]
        self.hv_ids = [19, 20]
        self.target_laps = 5
        
        home_dir = os.path.expanduser('~')
        hv_path_file = os.path.join(home_dir, 'kmc_ws/src/controller/path/roundabout_lane_two.json')
        cav_path_file = os.path.join(home_dir, 'kmc_ws/src/controller/path/path1.csv')
        
        try:
            with open(hv_path_file, 'r') as f:
                data = json.load(f)
                self.hv_ref_path = np.array(list(zip(data['X'], data['Y'])))
        except: self.hv_ref_path = None
            
        try: self.cav_ref_path = np.loadtxt(cav_path_file, delimiter=',')[:, :2]
        except: self.cav_ref_path = None

        self.vehicles = {}
        for vid in self.cav_ids + self.hv_ids:
            self.vehicles[vid] = {
                'kf_x': AdvancedKalman(0.05, 0.1),
                'kf_y': AdvancedKalman(0.05, 0.1),
                'last_pose': [0.0, 0.0, 0.0],
                'v_est': 0.0,
                'v_cmd': 0.0,
                'omega': 0.0,      # [추가] 차량의 각속도 명령값 저장용
                'motion_yaw': 0.0,  # [추가] 실제 이동 궤적으로 계산한 방향 저장용
                'lap_count': 0,
                'halfway': False,
                'last_time': self.get_clock().now(),
                'is_cav': vid in self.cav_ids
            }
            
        # [추가] __init__ 함수 맨 아래쪽 (self.is_terminated = False 위나 아래)
        self.start_node_time = self.get_clock().now()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        for vid in self.cav_ids + self.hv_ids:
            name = f"CAV_0{vid}" if vid < 10 else (f"CAV_{vid}" if vid < 19 else f"HV_{vid}")
            self.create_subscription(PoseStamped, f'/{name}', lambda msg, v=vid: self.pose_callback(msg, v), qos)
            
        # 기존 Accel 퍼블리셔 삭제 후 Float32 타입의 target_v 퍼블리셔로 변경
        self.pubs_target_v = {vid: self.create_publisher(Float32, f'/CAV_{vid:02d}_target_v', 10) for vid in self.cav_ids}
        
        # 주행 노드가 실제로 내뱉는 Accel 명령값을 인프라가 구독 (예측 정밀도 향상)
        for vid in self.cav_ids:
            # 주행 노드(CAV_0i)가 발행하는 accel 토픽 구독
            name = f"CAV_0{vid}" if vid < 10 else f"CAV_{vid}"
            self.create_subscription(Accel, f'/{name}_accel', 
                                     lambda msg, v=vid: self.v_cmd_callback(msg, v), 10)
            
        
        # 차량 상태 데이터 초기값 수정 (v_cmd를 0.0으로 시작)
        for vid in self.cav_ids + self.hv_ids:
            self.vehicles[vid]['v_cmd'] = 0.0

        self.log_dir = os.path.join(home_dir, 'kmc_ws/src/controller/logs/infra/')
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = open(f"{self.log_dir}/infra_log_{timestamp}.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'id', 'x', 'y', 'yaw', 'v_est', 'lap', 'target_v'])

        self.timer = self.create_timer(0.05, self.manager_loop)
        self.is_terminated = False

    # __init__ 안에서 이 함수 정의 부분을 삭제하고, 클래스 메서드 위치로 옮겨주세요.
    def v_cmd_callback(self, msg, vid):
        # 이제 주행 노드(CAV)가 실제로 발행한 속도 명령을 인프라가 정확히 수신합니다.
        self.vehicles[vid]['v_cmd'] = msg.linear.x
        self.vehicles[vid]['omega'] = msg.angular.z  # [추가] 각속도 명령값 수신
    
    
    def get_corridor(self, vid):
        v_data = self.vehicles[vid]
        # CAV와 HV의 레퍼런스 경로 선택
        path = self.cav_ref_path if v_data['is_cav'] else self.hv_ref_path
        if path is None: return {'current': None, 'future': []}

        curr_x, curr_y, m_yaw = v_data['last_pose']
        v_curr = max(0.2, v_data['v_est']) # 사각지대 방지용 최저 속도 보정

        # 1. 최인접 인덱스(ni) 찾기 - [초기 3초간 전역 탐색으로 (0,0) 탈출]
        elapsed = (self.get_clock().now() - self.start_node_time).nanoseconds / 1e9
        dists = np.linalg.norm(path - [curr_x, curr_y], axis=1)
        
        if elapsed < 3.0:
            ni = np.argmin(dists) # 전역 탐색
        else:
            # 원래는 근처만 뒤져야 하지만, 안전을 위해 일단 전체 탐색 유지 (연산량 충분함)
            ni = np.argmin(dists)

        # 2. [1층] 현재 실제 몸체 (Motion Yaw 사용)
        # 실제 필터링된 XY 좌표를 중심으로 딱 하나의 OBB 생성
        current_obb = OBB(curr_x, curr_y, m_yaw, self.CAR_L, self.CAR_W)

        # 3. [2층] 미래 예측 복도 (Path Yaw 사용)
        # [수정] ni + 20이 아니라 ni(현재 위치)부터 즉시 조사 시작 (사각지대 제거)
        start_idx = ni 
        
        # [수정] 조사 거리에 20cm를 더해줌 (v*t + 0.2m)
        lookahead_limit = (v_curr * self.LOOKAHEAD_TIME) + (self.PATH_OFFSET * 0.01)
        
        future_corridor = []
        accum_dist = 0.0
        # 10개(10cm) 간격으로 다운샘플링하며 복도 생성
        for i in range(start_idx, len(path)-1, 10):
            p1, p2 = path[i], path[i+1]
            dist_step = np.linalg.norm(p2 - p1)
            accum_dist += dist_step
            
            if accum_dist > lookahead_limit: break
            
            # 경로의 방향(Path Yaw)으로 복도 정렬
            p_yaw = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
            arrival_time = accum_dist / v_curr
            
            future_corridor.append({
                'obb': OBB(p1[0], p1[1], p_yaw, self.CAR_L, self.CORRIDOR_W),
                't': arrival_time,
                'd': accum_dist
            })
            
        return {'current': current_obb, 'future': future_corridor}

    def pose_callback(self, msg, vid):
        
        
        raw_x, raw_y = msg.pose.position.x, msg.pose.position.y
        # 2. [여기!] (0,0) 늪 방지 로직을 가장 먼저 수행
        # x와 y가 모두 0에 아주 가깝다면, 아직 시뮬레이션이 안 켜졌거나 센서가 튄 것이므로 무시하고 나갑니다.
        if abs(raw_x) < 0.001 and abs(raw_y) < 0.001:
            return
        
        v_data = self.vehicles[vid]
        now = self.get_clock().now()
        dt = max(0.001, (now - v_data['last_time']).nanoseconds / 1e9)
        
        # [설계 준수] 센서 Yaw는 로그 기록용으로만 쓰고 제어 예측에는 버립니다.
        q = msg.pose.orientation
        raw_yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

        # 1. 주행 노드와 동일한 물리 모델 예측 (Prediction Offset)
        if v_data['is_cav']:
            # [핵심] 명령받은 각속도(omega)를 반영하여 예측 방향 계산
            predicted_yaw = v_data['motion_yaw'] + (v_data['omega'] * dt)
            dx = v_data['v_cmd'] * np.cos(predicted_yaw) * dt
            dy = v_data['v_cmd'] * np.sin(predicted_yaw) * dt
        else:
            # HV는 명령을 모르므로 추정 속도와 경로 방향(Path Yaw) 기반 예측
            # (HV 경로는 나중에 ni 계산 시점에 더 정밀해지겠지만 우선 기본값 적용)
            dx = v_data['v_est'] * np.cos(v_data['motion_yaw']) * dt
            dy = v_data['v_est'] * np.sin(v_data['motion_yaw']) * dt

        # 2. 칼만 필터 업데이트 (예측량 dx, dy 반영)
        filt_x = v_data['kf_x'].step(raw_x, dx, gate=0.5)
        filt_y = v_data['kf_y'].step(raw_y, dy, gate=0.5)

        # 3. Motion Yaw 계산 (실제 필터링된 좌표 이동량 기반)
        dx_actual = filt_x - v_data['last_pose'][0]
        dy_actual = filt_y - v_data['last_pose'][1]
        dist_moved = np.sqrt(dx_actual**2 + dy_actual**2)
        
        if dist_moved > 0.02: # 2cm 이상 움직였을 때만 방향 갱신 (주행 노드와 동일)
            v_data['motion_yaw'] = np.arctan2(dy_actual, dx_actual)

        # 4. 속도 데이터 확정 (v_est)
        if v_data['is_cav']:
            v_data['v_est'] = v_data['v_cmd'] # CAV는 명령값이 곧 예측 속도
        else:
            v_data['v_est'] = (v_data['v_est'] * 0.5) + ((dist_moved / dt) * 0.5)
            v_data['v_est'] = min(v_data['v_est'], 2.0)

        # 5. 최종 상태 저장
        v_data['last_pose'] = [filt_x, filt_y, v_data['motion_yaw']] # Yaw 자리에 Motion Yaw 저장
        v_data['last_time'] = now

        # Lap 카운트 로직은 기존 유지 (filt_y 기준)
        if filt_y > 0 and not v_data['halfway']: v_data['halfway'] = True
        if filt_y < -0.5 and v_data['halfway']:
            v_data['lap_count'] += 1
            v_data['halfway'] = False

    def manager_loop(self):
        if self.is_terminated: return

        results = {vid: self.get_corridor(vid) for vid in self.vehicles}
        target_v_dict = {vid: self.V_MAX for vid in self.cav_ids}

        vids = list(self.vehicles.keys())
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                id1, id2 = vids[i], vids[j]
                res1, res2 = results[id1], results[id2] # 정의 위치 수정

                if res1['current'] is None or res2['current'] is None: continue

                # --- [A] 현재 vs 미래 (반응적 제동) ---
                # id1(고정체)이 id2(주행차)의 길을 막는 경우
                for item in res2['future']:
                    if check_obb_collision(res1['current'], item['obb']):
                        if id2 in target_v_dict:
                            target_v_dict[id2] = 0.0
                        break

                # id2(고정체)가 id1(주행차)의 길을 막는 경우
                for item in res1['future']:
                    if check_obb_collision(res2['current'], item['obb']):
                        if id1 in target_v_dict:
                            target_v_dict[id1] = 0.0
                        break

                # --- [B] 미래 vs 미래 (협력적 감속 + 후방 추돌 방지) ---
                t1_list, t2_list, d1_list, d2_list = [], [], [], []
                for it1 in res1['future']:
                    for it2 in res2['future']:
                        if check_obb_collision(it1['obb'], it2['obb']):
                            t1_list.append(it1['t']); t2_list.append(it2['t'])
                            d1_list.append(it1['d']); d2_list.append(it2['d'])
                
                if t1_list and t2_list:
                    t_e1, t_e2 = max(t1_list), max(t2_list)
                    t_s1, t_s2 = min(t1_list), min(t2_list)
                    d_s1, d_s2 = min(d1_list), min(d2_list)
                    
                    if not (t_e1 + self.TIME_MARGIN < t_s2 or t_e2 + self.TIME_MARGIN < t_s1):
                        # [핵심 수정] 거리 기반 우선순위 (누가 더 충돌점에 가까운가?)
                        # 내가 이미 충돌점에 진입(30cm 이내)했다면 내가 Winner!
                        if d_s1 < 0.3: w_id, y_id, t_e_w, d_s_y = id1, id2, t_e1, d_s2
                        elif d_s2 < 0.3: w_id, y_id, t_e_w, d_s_y = id2, id1, t_e2, d_s1
                        # 둘 다 멀리 있다면 기존 원칙 (HV 우선 -> 먼저 도착하는 놈 우선)
                        elif id1 in self.hv_ids: w_id, y_id, t_e_w, d_s_y = id1, id2, t_e1, d_s2
                        elif id2 in self.hv_ids: w_id, y_id, t_e_w, d_s_y = id2, id1, t_e2, d_s1
                        elif t_s1 < t_s2: w_id, y_id, t_e_w, d_s_y = id1, id2, t_e1, d_s2
                        else: w_id, y_id, t_e_w, d_s_y = id2, id1, t_e2, d_s1
                        
                        v_target = d_s_y / (t_e_w + self.SAFETY_GAP)
                        if y_id in target_v_dict:
                            target_v_dict[y_id] = min(target_v_dict[y_id], v_target)
        
    # send_speed 메서드 수정 (Accel -> Float32)
    def send_speed(self, vid, v):
        msg = Float32()
        msg.data = float(v)
        self.pubs_target_v[vid].publish(msg)

    # terminate_all 메서드 내 정지 명령 수정
    def terminate_all(self):
        self.is_terminated = True
        for vid in self.cav_ids:
            self.send_speed(vid, 0.0) # Float32 타입으로 0.0 전송
        self.csv_file.close()
        os._exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = SmartInfraManager()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.terminate_all()
        rclpy.shutdown()

if __name__ == '__main__':
    main()