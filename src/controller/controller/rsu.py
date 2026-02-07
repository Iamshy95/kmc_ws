import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import pandas as pd
import numpy as np
import os
import time
import threading
import queue

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


class PathAwareRSU(Node):
    def __init__(self):
        super().__init__('path_aware_rsu')
        
        # [1-1] 제어 파라미터 변수화 (상단 배치)
        self.ENTER_DIST = 1.3        # 회전교차로 진입 판정
        self.EXIT_DIST  = 2.0        # 
        self.WATCH_MIN  = 1.0        # ttc 범위
        self.WATCH_MAX  = 2.0        # ttc 범위
        
        self.SAFE_CROSS  = 10.0       # safe margin
        self.SAFE_FOLLOW = 0.18      # 경로 안 겹칠때
        
        self.SECTOR_1_4 = [50.0, 230.0]
        self.SECTOR_2_3 = [320.0, 140.0]
        
        # --- 제어 파라미터 (직접 수정하며 테스트할 변수) ---
        self.TTC_THRESHOLD = 1.8       # 진입을 위한 최소 TTC (초)
        self.SAFE_ANGLE_MARGIN = 20.0  # 충돌지점 앞뒤로 정지할 각도 범위 (도)

        # [1-2] CSV 로그 설정 (Method B: 헤더 주석 포함)
        self.log_dir = os.path.expanduser("~/kmc_ws/src/controller/logs/infra")
        # [추가] 폴더가 없으면 생성, 이미 있으면 무시 (exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.log_dir, f"rsu_log_{timestamp}.csv")
        self.f_log = open(self.log_path, 'w', encoding='utf-8')
        
        # [비동기 로깅 설정 추가]
        self.log_queue = queue.Queue()
        self.stop_logging = False
        self.logging_thread = threading.Thread(target=self._logging_worker)
        self.logging_thread.start()

        # [수정] 21개 컬럼 헤더 (Raw_X, Raw_Y 추가)
        header = ("Time,ID,X,Y,Raw_X,Raw_Y,Path_ID,Zone,Can_go,Signal,Target_HV,Target_CAV,"
                  "Velocity,HV_Deg,TTC,Dist,Decision_Mode\n")
        self.f_log.write(header)

        # 실험 설정값(Params)을 첫 줄에 주석으로 기록
        # [수정] 모든 파라미터 기록 (줄바꿈 주의)
        params_line = (f"# PARAMS: ENTER={self.ENTER_DIST}, EXIT={self.EXIT_DIST}, "
                       f"W_MIN={self.WATCH_MIN}, W_MAX={self.WATCH_MAX}, "
                       f"TTC_TH={self.TTC_THRESHOLD}, ANG_MAR={self.SAFE_ANGLE_MARGIN}, "
                       f"S_CROSS={self.SAFE_CROSS}, S_FOLLOW={self.SAFE_FOLLOW}\n")
        self.f_log.write(params_line)

        

        self.base_path = os.path.expanduser("~/kmc_ws/src/controller/path")
        self.id_map = {1: 1, 2: 2, 3: 3, 4: 4}
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
                'raw_pos': np.array([0.0, 0.0]),   # [추가] 로그용 원본
                'kf_x': AdvancedKalman(0.1, 0.1), # [분리] X축 필터
                'kf_y': AdvancedKalman(0.1, 0.1), # [분리] Y축 필터
                'last_pose_time': self.get_clock().now(), # [추가] dt 계산용
                'omega': 0.0,                # [추가] 각속도 저장용
                'motion_yaw': 0.0,           # [추가] 계산된 방향
                'prev_pos': None,     # 이전 위치 저장용
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
                'decision_mode': "NONE",
                'zone_entries': {},  # [추가] 겹치는 구역 시간 관리용
                # --- [신규 추가: 로그 및 판단 근거용] ---
                'actual_vel': 0.0,        # 컬럼 11: 실제 차에서 오는 속도 토픽
                'target_hv': "None",      # 컬럼 8: 나를 멈추게 한 HV ID
                'target_cav': "None",     # 컬럼 9: 나를 멈추게 한 앞차 ID
                'hv_deg': 0.0,            # 컬럼 13: 위험 HV의 각도
                'current_ttc': 99.0,      # 컬럼 14: 실시간 TTC
                'current_dist': 999.0,    # 컬럼 15: 타겟과의 거리
                'pub': self.create_publisher(Bool, f'/infra/CAV_{formatted_id}/go_signal', 10),
            }

        loaded_count = sum(1 for c in self.cars.values() if c['path'] is not None)
        self.get_logger().info(f"Paths loaded: {loaded_count}/{len(self.cav_ids)}")

        # [수정 후] 이렇게 바꿔주세요!
        self.hvs = {}
        for hid in self.hv_ids:
            self.hvs[f'HV{hid}'] = {
                'pos': None,
                'raw_pos': np.array([0.0, 0.0]),   # [추가] 로그용 원본
                'kf_x': AdvancedKalman(0.1, 0.1),  # [추가] X축 필터
                'kf_y': AdvancedKalman(0.1, 0.1),  # [추가] Y축 필터
                'motion_yaw': 0.0,                 # [추가] 이동 방향
                'prev_pos': None,                  # [추가] Motion Yaw 계산용
                'time': None,              # 최신 수신 시각
                'last_calc_pos': None,     # 계산에 사용된 마지막 위치
                'last_calc_time': None,    # 계산에 사용된 마지막 시각
                'vel': 0.0,
                'kalman': AdvancedKalman(q=0.1, r=0.1), # 차량별 전용 필터
                'ma_buffer': []            # 차량별 전용 MA10 버퍼
            }

        self.zones = {
            "4way":  {"x": [-4.3, -0.4], "y": [-1.6, 1.6]},
            "zone1": {"x": [-4.1, -1.4], "y": [1.1, 2.7]},
            "zone2": {"x": [-3.3, -0.5], "y": [-2.7, -1.1]},
        }
        self.round_center = np.array([1.67, 0.0])

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
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
            self.create_subscription(
                Float32, f'/CAV_{fid}/vehicle_speed',
                lambda msg, c_id=f'CAV{fid}': self.speed_cb(msg, c_id),
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
        full_path = os.path.expanduser(os.path.join(self.base_path, f'path{num}.csv'))
        try:
            if not os.path.exists(full_path):
                self.get_logger().error(f"File not found: {full_path}")
                return None
            data = pd.read_csv(full_path, header=None).values[:, :2]
            return data
        except Exception as e:
            self.get_logger().error(f"Load error ({full_path}): {str(e)}")
            return None
    
    
    def _logging_worker(self):
        """제어 루프와 독립적으로 파일 쓰기 수행"""
        while not self.stop_logging or not self.log_queue.empty():
            try:
                line = self.log_queue.get(timeout=1.0)
                if line:
                    self.f_log.write(line)
                    # 큐가 비었을 때만 디스크에 기록(Flush)하여 부하 최소화
                    if self.log_queue.empty():
                        self.f_log.flush()
                self.log_queue.task_done()
            except queue.Empty:
                continue

    def pose_cb(self, msg, car_id):
        data = self.cars[car_id]
        now = self.get_clock().now()
        dt = max(0.001, (now - data['last_pose_time']).nanoseconds / 1e9)
        data['last_pose_time'] = now

        # 1. 방향(Motion Yaw) 계산
        if data['pos'] is not None and data['prev_pos'] is not None:
            dx_m, dy_m = data['pos'][0] - data['prev_pos'][0], data['pos'][1] - data['prev_pos'][1]
            if np.hypot(dx_m, dy_m) > 0.01:
                data['motion_yaw'] = np.arctan2(dy_m, dx_m)

        # 2. 예측 오프셋(dx, dy) 계산 오류수정!!! 일단 0으로 바꿈
        pred_yaw = data['motion_yaw'] + (data['omega'] * dt)
        dx = 0
        dy = 0

        # 3. ★게이트(gate=0.25) 적용하여 필터 업데이트★
        raw_x, raw_y = msg.pose.position.x, msg.pose.position.y
        data['raw_pos'] = np.array([raw_x, raw_y]) # <--- 이 줄을 추가하면 CAV 원본도 기록됩니다.
        filt_x = data['kf_x'].step(raw_x, dx, gate=0.5)
        filt_y = data['kf_y'].step(raw_y, dy, gate=0.5)

        # 4. 결과 업데이트
        data['prev_pos'] = data['pos'].copy() if data['pos'] is not None else None
        data['pos'] = np.array([filt_x, filt_y])

    def cav_vel_cb(self, msg, car_id):
        self.cars[car_id]['vel'] = msg.linear.x
        self.cars[car_id]['omega'] = msg.angular.z  # 각속도 저장 추가
        
    def speed_cb(self, msg, car_id):
        # 18개 컬럼 중 'Actual_Vel' 자리에 들어갈 데이터만 업데이트
        self.cars[car_id]['actual_vel'] = msg.data

    def hv_cb(self, msg, hv_id):
        data = self.hvs[hv_id]
        raw_x, raw_y = msg.pose.position.x, msg.pose.position.y
        
        # [로그용 원본 저장]
        data['raw_pos'] = np.array([raw_x, raw_y])

        # 1. dt 계산
        dt = 0.05
        if data['time'] is not None:
            curr_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            dt = max(0.001, curr_ts - data['time'])

        # 2. Motion Yaw 계산 (prev_pos 사용)
        if data['pos'] is not None and data['prev_pos'] is not None:
            dx_m = data['pos'][0] - data['prev_pos'][0]
            dy_m = data['pos'][1] - data['prev_pos'][1]
            if np.hypot(dx_m, dy_m) > 0.01:
                data['motion_yaw'] = np.arctan2(dy_m, dx_m)

        # 3. 예측 (속도 * Motion Yaw)
        dx = 0
        dy = 0

        # 4. 동적 게이트
        dynamic_gate = 1.0

        # 5. 필터링 (예측 적용)
        filt_x = data['kf_x'].step(raw_x, dx, gate=dynamic_gate)
        filt_y = data['kf_y'].step(raw_y, dy, gate=dynamic_gate)

        # 6. 저장
        data['prev_pos'] = data['pos'].copy() if data['pos'] is not None else None
        data['pos'] = np.array([filt_x, filt_y])
        data['time'] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

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
                    raw_vel = np.clip(dist / actual_dt, 0.1, 2.0) # 클램핑

                    # 1. 칼만 필터
                    kf_v = hv['kalman'].step(raw_vel)

                    # 2. MA20 (이동 평균)
                    hv['ma_buffer'].append(kf_v)
                    if len(hv['ma_buffer']) > 20:
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
            
            # [추가할 코드] Raw 좌표 포함해서 바로 기록
            if curr_pos is not None:
                raw_p = hv['raw_pos']
                # X, Y 뒤에 raw_p[0], raw_p[1] 추가됨
                hv_row = (f"{now_ts:.3f},{hid},{curr_pos[0]:.3f},{curr_pos[1]:.3f},"
                          f"{raw_p[0]:.3f},{raw_p[1]:.3f},None,None,None,None,None,None,"
                          f"{hv['vel']:.2f},0.0,0.0,0.0,NONE\n")
                self.log_queue.put(hv_row)

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

        # Zone 큐 (겹침 구간 우선순위 처리 로직 적용)
        for cid in active_cavs:
            data = self.cars[cid]
            
            # [3-1] 매 루프 시작 시 타겟 정보 초기화 (기존 코드 유지)
            data['target_hv'] = "None"
            data['target_cav'] = "None"
            data['hv_deg'] = 0.0
            data['current_dist'] = 999.0 
            
            x, y = data['pos'][0], data['pos'][1]
            
            # 1. 현재 위치하고 있는 모든 Zone 찾기
            current_in_zones = []
            for z_name, limit in self.zones.items():
                if (limit['x'][0] <= x <= limit['x'][1]) and (limit['y'][0] <= y <= limit['y'][1]):
                    current_in_zones.append(z_name)
            
            # 2. 구역별 진입 시간 관리 (zone_entries가 없으면 생성)
            if 'zone_entries' not in data:
                data['zone_entries'] = {}
            
            # (A) 벗어난 구역은 기록 삭제
            for z in list(data['zone_entries'].keys()):
                if z not in current_in_zones:
                    del data['zone_entries'][z]
            
            # (B) 새로 들어온 구역은 현재 시간 기록
            for z in current_in_zones:
                if z not in data['zone_entries']:
                    # 3번, 4번 차라면 진입 시간을 0.05초 뒤로 밀어서 기록 (우선순위 하락)
                    offset = 0.05 if cid in ['CAV03', 'CAV04'] else 0.0
                    data['zone_entries'][z] = time.time() + offset
            
            # 3. 대표 구역(Primary Zone) 결정: 가장 늦게(최근에) 진입한 곳
            if data['zone_entries']:
                # 진입 시간이 가장 큰(최신) 구역을 선택
                primary_zone = max(data['zone_entries'], key=data['zone_entries'].get)
                
                data['current_zone'] = primary_zone
                data['entry_time'] = data['zone_entries'][primary_zone]
                
                # 해당 Zone 대기열에 추가
                zone_queues[primary_zone].append(cid)
            else:
                data['current_zone'] = None
                data['entry_time'] = 0

        for z_name in zone_queues:
            zone_queues[z_name].sort(key=lambda x: self.cars[x]['entry_time'])

        # CAV 판단
        for cid in active_cavs:
            data = self.cars[cid]
            can_go = True
            reason = "Clear (No Hazard)"

            dist_round = np.linalg.norm(data['pos'] - self.round_center)

            # roundabout 상태 업데이트
            if dist_round < self.ENTER_DIST:
                data['in_roundabout'] = True
            if dist_round > self.EXIT_DIST:
                data['in_roundabout'] = False
                data['min_ttc_record'] = 99.0
                data['min_dist_record'] = 999.0
                data['rebound_released'] = False

            data['current_ttc'] = 99.0

            if data['in_roundabout']:
                can_go = True
                reason = "In-Process (Inside Roundabout)"

            elif (self.WATCH_MIN <= dist_round <= self.WATCH_MAX):
                # 1. 이미 진입 허가(TRUE)를 받았다면 더 이상 계산 안 함 (판단 잠금)
                if data['decision_mode'] == "TRUE":
                    can_go = True
                else:
                    # 2. 아직 판단 전(NONE)이거나 정지 상태(FALSE)일 때만 체크
                    is_hazard = False
                    # 기존 path_id 구분을 그대로 사용 (1,4 -> 180도 / 2,3 -> 90도)
                    target_ang = 180.0 if data['path_id'] in [1, 4] else 90.0
                    
                    for hid, hv in self.hvs.items():
                        if hv['pos'] is None: continue
                        
                        v_hv = hv['pos'] - self.round_center
                        hv_deg = np.degrees(np.arctan2(v_hv[1], v_hv[0])) % 360
                        
                        # [각도 차이 계산] HV가 타겟까지 남은 각도
                        # 수정: angle_diff = target_ang - hv_deg  (반시계)
                        angle_diff = target_ang - hv_deg
                        # [조건 A] 안전 각도 (기존 변수나 고정값 30.0 사용)
                        if abs(angle_diff) < self.SAFE_ANGLE_MARGIN: # 직접 수정 가능
                            is_hazard = True
                            break
                            
                        # [조건 B] TTC (남은 각도 기반)
                        if angle_diff > 0:
                            # 1.0 = 회전교차로 반지름
                            ttc = (np.radians(angle_diff) * 1.0) / hv['vel']
                            if ttc < self.TTC_THRESHOLD: # 직접 수정 가능 (TTC 기준)
                                is_hazard = True
                                break
                        else:
                            # 이미 지나간 경우 (angle_diff < 0)는 안전으로 간주
                            pass

                    # 3. 판단 결과 저장
                    if not is_hazard:
                        data['decision_mode'] = "TRUE"
                        can_go = True
                    else:
                        data['decision_mode'] = "FALSE"
                        can_go = False

            else:
                # FIFO 구간(2번 코드 그대로)
                data['decision_mode'] = "NONE" # 2.0m 밖으로 나가면 초기화
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
                                # 상단에서 정의한 self.SAFE_CROSS(2.0)와 self.SAFE_FOLLOW(0.18) 사용
                                safe_margin = self.SAFE_CROSS if is_crossing else self.SAFE_FOLLOW
                                
                                if dist < safe_margin:
                                    can_go = False
                                    # 로그용 데이터 저장: 나를 막고 있는 앞차 ID와 거리 박제
                                    data['target_cav'] = front_cid
                                    data['current_dist'] = dist
                                    break

            # 터미널 로그 (Zone 정보 유지 + Target 정보 추가)
            if data['last_signal'] != can_go:
                status = "GO" if can_go else "STOP"
                
                # 타겟 식별 (HV가 없으면 CAV를 확인)
                target = data['target_hv'] if data['target_hv'] != "None" else data['target_cav']
                
                # [수정] 존 정보와 타겟 정보를 함께 출력
                log_msg = f"[{cid}] {status} | Zone: {data['current_zone']} | Target: {target}"
                self.get_logger().info(log_msg)
                
                # 신호 상태 업데이트
                data['last_signal'] = can_go

            # CSV 기록 (19개 컬럼 순서 엄수)
            # 순서: Time, ID, X, Y, Path_ID, Zone, Can_go, Signal, Target_HV, Target_CAV, 
            #       Cmd_Vel, Actual_Vel, Calc_Vel, HV_Deg, TTC, Dist, Min_TTC_Rec, Min_Dist_Rec, Rebound_Stat
            
            # Rebound_Stat은 True/False보다 1/0이 데이터 분석 시 훨씬 편합니다.
            rebound_val = 1 if data['rebound_released'] else 0
            can_go_val = 1 if can_go else 0

            # [CAV 부분]
            cav_row = (f"{now_ts:.3f},{cid},{data['pos'][0]:.3f},{data['pos'][1]:.3f},"
                       f"{data['raw_pos'][0]:.3f},{data['raw_pos'][1]:.3f},"
                       f"{data['path_id']},{data['current_zone']},{can_go_val},{'GO' if can_go else 'STOP'},"
                       f"{data['target_hv']},{data['target_cav']},{data['actual_vel']:.2f}," # Velocity 통합
                       f"{data['hv_deg']:.1f},{data['current_ttc']:.2f},{data['current_dist']:.2f},"
                       f"{data['decision_mode']}\n")
            self.log_queue.put(cav_row)

            # 신호 전송
            msg = Bool()
            msg.data = can_go
            data['pub'].publish(msg)

    def destroy_node(self):
        self.stop_logging = True
        if hasattr(self, 'logging_thread'):
            self.logging_thread.join()
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
