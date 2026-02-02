import numpy as np
import time

#### 계산

# [1] 기하 및 거리 계산
def get_distance(p1, p2):
    """두 점 사이의 유클리드 거리 (사용자 원본 방식)"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# [2] 각도 및 회전 변환
def quat_to_yaw(q):
    """쿼터니언(x, y, z, w)을 라디안 Yaw로 변환"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

def wrap_angle(angle, reference):
    """
    위상 도약(Phase Jump) 방지용 언래핑
    reference(이전 Yaw)를 기준으로 현재 각도를 +-PI 범위 내로 보정
    """
    if reference is None:
        return angle
    
    while angle - reference > np.pi:
        angle -= 2.0 * np.pi
    while angle - reference < -np.pi:
        angle += 2.0 * np.pi
    return angle

# [3] 제어 신호 보정
def apply_deadzone(value, threshold):
    """
    연속형 데드존: 임계값 이내면 0, 넘어가면 그 차이만큼만 출력
    CTE, D-항, Yaw 오차 보정에 공통 사용
    """
    if abs(value) < threshold:
        return 0.0
    return value - (np.sign(value) * threshold)

# [4] 데이터 신선도(Freshness) 관리
def get_age(last_time, current_time):
    """마지막 데이터 수신 후 경과 시간(sec) 산출"""
    if last_time is None:
        return 999.0  # 데이터가 한 번도 안 들어온 경우 아주 오래된 것으로 간주
    
    # ROS2 Time 객체 간의 차이를 초 단위로 변환
    diff = current_time - last_time
    return diff.nanoseconds / 1e9

#### 필터

class AdvancedKalman:
    def __init__(self, q=0.1, r=0.1):
        self.q = q  # 프로세스 노이즈 (예측 모델 신뢰도)
        self.r = r  # 측정 노이즈 (센서 신뢰도)
        self.x = None
        self.p = 1.0

    def step(self, measurement, prediction_offset=0.0, gate=None):
        """
        measurement: 센서 입력값
        prediction_offset: 물리 모델 기반 이동 예측량 (dx, dy 등)
        gate: 센서 튐 방지 임계값 (m)
        """
        if self.x is None:
            self.x = measurement
            return self.x

        # 1. 예측 단계 (Prediction)
        x_prior = self.x + prediction_offset
        p_prior = self.p + self.q

        # 2. 게이트 로직 (Gate Logic)
        # 센서 데이터가 예측치에서 너무 멀리 떨어지면 센서를 무시하고 예측값 유지
        if gate is not None and abs(measurement - x_prior) > gate:
            self.x = x_prior
            self.p = p_prior
            return self.x

        # 3. 업데이트 단계 (Update)
        k_gain = p_prior / (p_prior + self.r)
        self.x = x_prior + k_gain * (measurement - x_prior)
        self.p = (1.0 - k_gain) * p_prior
        return self.x

class MovingAverageFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = [0.0] * window_size

    def step(self, value):
        """
        value: 새로운 입력 데이터 (target_v 등)
        return: 이동평균 결과값 (v_smoothed)
        """
        self.buffer.pop(0)
        self.buffer.append(value)
        return sum(self.buffer) / self.window_size
    
   
####### 인덱스

def find_nearest_index(pos, path, prev_ni, start_time, halfway_passed):
    """
    pos: [x, y] 현재 좌표
    path: [[x,y], ...] 전체 경로
    prev_ni: 이전 루프의 인덱스
    start_time: 주행 시작 시각 (time.time())
    halfway_passed: 반환점 통과 여부 플래그
    """
    path_len = len(path)
    elapsed_time = time.time() - start_time
    px, py = pos[0], pos[1]

    # [1] 탐색 범위 결정
    if prev_ni is None or elapsed_time < 5.0:
        indices = np.arange(path_len)
    else:
        # 주행 중에는 이전 인덱스 전후 100개(약 2m)만 탐색
        indices = np.arange(prev_ni - 100, prev_ni + 100) % path_len

    # [2] 최단 거리 인덱스 추출 (제곱 거리 사용)
    search_pts = path[indices]
    dists_sq = np.sum((search_pts - [px, py])**2, axis=1)
    ni = indices[np.argmin(dists_sq)]

    # [3] 랩 카운팅 및 플래그 관리 로직
    lap_increment = False
    
    # --- 핵심 수정: 반환점 플래그 활성화 구간 한정 ---
    # 결승선 근처(90%~10%)에서 뒤로 튀어도 플래그가 다시 켜지지 않도록
    # 트랙의 중간 영역(40% ~ 80%)을 지날 때만 halfway_passed를 True로 설정합니다.
    if path_len * 0.4 < ni < path_len * 0.8:
        halfway_passed = True

    # 결승선 통과 판정 (5초 이후 + 반환점을 찍고 왔을 때만)
    if elapsed_time > 5.0 and halfway_passed:
        # 90% 이상 지점에서 10% 미만 지점으로 넘어가는 순간
        if prev_ni is not None and prev_ni > path_len * 0.9 and ni < path_len * 0.1:
            lap_increment = True
            halfway_passed = False  # 바퀴 수 올리자마자 플래그 OFF (인터록)

    return ni, halfway_passed, lap_increment


def get_control_metrics(pos, ni, path):
    """
    pos: [px, py] (보통 예측된 위치)
    ni: 현재 인덱스
    path: 전체 경로 데이터
    return: path_yaw, cte
    """
    path_len = len(path)
    px, py = pos[0], pos[1]
    
    # [1] 국부 윈도우 추출 (전후 5개씩 총 11개 포인트)
    indices = [(ni + i) % path_len for i in range(-5, 6)]
    pts = path[indices]
    
    # [2] PCA 연산 (주성분 추출)
    center = np.mean(pts, axis=0)
    norm_pts = pts - center
    cov = np.dot(norm_pts.T, norm_pts) # 2x2 공분산 행렬
    
    # 고윳값(val)과 고유벡터(vec) 계산
    val, vec = np.linalg.eigh(cov)
    tangent = vec[:, np.argmax(val)] # 가장 큰 고윳값의 벡터 선택
    
    # [3] 진행 방향(path_yaw) 결정
    path_yaw = np.arctan2(tangent[1], tangent[0])
    
    # 경로의 다음 점 방향과 대조하여 방향 반전 여부 확인 (역주행 방지)
    next_idx = (ni + 1) % path_len
    if np.dot(tangent, path[next_idx] - path[ni]) < 0:
        path_yaw += np.pi
        # -PI ~ PI 범위로 재조정
        path_yaw = (path_yaw + np.pi) % (2 * np.pi) - np.pi
                
    # [4] 횡방향 이탈 오차(CTE) 계산
    dx, dy = px - path[ni][0], py - path[ni][1]
    
    # 진행 방향의 수직 벡터와 오차 벡터를 내적하여 좌/우 이탈량 산출
    # 왼쪽 이탈이 (+), 오른쪽 이탈이 (-)
    cte = -np.sin(path_yaw) * dx + np.cos(path_yaw) * dy
    
    return path_yaw, cte


def get_curvature(ni, window, path):
    """
    3점 기하학 기반 곡률 계산
    코드 원본: p1, p2, p3를 이용한 벡터 사이각 산출 방식
    """
    path_len = len(path)
    p1 = path[ni]
    p2 = path[(ni + window // 2) % path_len]
    p3 = path[(ni + window) % path_len]
    
    v1, v2 = p2 - p1, p3 - p2
    
    # 두 벡터 사이의 각도 변화량 계산
    ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    
    # -PI ~ PI 범위로 정규화
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    
    # 두 끝점 사이의 직선 거리
    dist = np.linalg.norm(p3 - p1)
    
    if dist < 0.01: 
        return 0.0
        
    return ang / dist

def get_max_future_curvature(ni, look_ahead_dist, window, path):
    """
    현재 위치부터 전방 n포인트까지 루프를 돌며 최대 곡률 탐색
    코드 원본: for i in range(look_ahead_dist) 루프 로직
    """
    path_len = len(path)
    max_curv = 0.0
    
    for i in range(look_ahead_dist):
        target_idx = (ni + i) % path_len
        # 각 지점의 곡률 계산 (절대값 기준)
        curv = abs(get_curvature(target_idx, window, path))
        
        if curv > max_curv:
            max_curv = curv
            
    return max_curv


def get_motion_yaw(curr_p, prev_p, last_motion_yaw, path_yaw):
    """
    차량의 이동 좌표(XY) 변화를 분석하여 실제 주행 방향을 계산
    
    curr_p: 현재 필터링된 [x, y]
    prev_p: 이전 루프의 [x, y]
    last_motion_yaw: 직전 루프에서 계산된 모션 야
    path_yaw: 현재 인덱스에서의 경로 방향 (초기값용)
    """
    # 1. 이동 거리 계산
    dx = curr_p[0] - prev_p[0]
    dy = curr_p[1] - prev_p[1]
    
    # 연산 효율을 위해 제곱값으로 비교 (0.02m = 2cm)
    # $dx^2 + dy^2 > 0.0004$ ($0.02^2$)
    dist_sq = dx**2 + dy**2
    
    # 2. 2cm 이상 움직였을 때만 방향 갱신
    if dist_sq > 0.0004:
        return np.arctan2(dy, dx)
    
    # 3. 안 움직였으면 이전 값을 유지하거나, 아예 처음이면 경로 방향이라도 반환
    if last_motion_yaw is not None:
        return last_motion_yaw
    else:
        return path_yaw
    
    
def predict_pose(p, v, yaw, dt):
    """
    현재 위치와 속도, 방향을 기반으로 dt초 후의 예상 위치를 계산
    코드 원본: pred_px = filt_px + (v * cos(yaw) * dt)
    
    p: 현재 [x, y]
    v: 현재 속도 (actual_v 또는 v_smoothed)
    yaw: 현재 방향 (motion_yaw)
    dt: 예측할 시간 간격 (pose_dt 또는 loop_dt)
    """
    pred_x = p[0] + (v * np.cos(yaw) * dt)
    pred_y = p[1] + (v * np.sin(yaw) * dt)
    
    return [pred_x, pred_y]

