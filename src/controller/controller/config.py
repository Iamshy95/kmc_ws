# config.py

# [1] 로그 헤더 (사용자 실차 베이스 + 신규 파라미터 2개 포함, 총 44개)
LOG_HEADERS = [
    'time', 'ni', 'lap_count', 'dt', 'pose_dt',                    # Metadata (5)
    'raw_px', 'raw_py', 'raw_yaw', 'filt_px', 'filt_py',           # Pose (5)
    'motion_yaw', 'path_yaw', 'pred_px', 'pred_py',                # Analysis (4)
    'cte', 'curvature',                                            # Metrics (2)
    'omega_pid', 'omega_ff', 'omega_yaw', 'is_flip',               # Control Breakdown (4)
    'target_v', 'v_smoothed', 'actual_v',                          # Velocity States (3)
    'cmd_v', 'cmd_w', 'echo_v', 'echo_w',                          # Commands (4)
    'p_kp', 'p_ki', 'p_kd', 'p_kd_deadzone', 'p_steer_deadzone',   # Gains 1 (5)
    'p_yaw_deadzone', 'p_ff_gain', 'p_ff_window', 'p_kyaw',        # Gains 2 (4)
    'p_v_max', 'p_v_min', 'p_look_ahead', 'p_v_curve_gain',        # Gains 3 (4)
    'p_v_cte_gain', 'p_kf_q_pose', 'p_kf_r_pose', 'kf_mode',       # Gains 4 (4)
    'battery', 'actual_v_age', 'raw_allstate'                      # Health (3)
]

# [2] 통합 제어 파라미터 (Gains & Look-ahead)
PARAMS = {
    "p_kp": 4.5, "p_ki": 2.5, "p_kd": 4.5,
    "p_kd_deadzone": 0.01,
    "p_steer_deadzone": 0.005,
    "p_yaw_deadzone": 3.0,
    "p_ff_gain": 2.0, "p_ff_window": 10, "p_kyaw": 2.0,
    "p_v_max": 1.5, "p_v_min": 0.5,
    "p_look_ahead": 40,            # 선감속 탐색 거리 (pts)
    "p_v_curve_gain": 0.8, "p_v_cte_gain": 0.1,
    "p_kf_q_pose": 0.1, "p_kf_r_pose": 0.1
}

# [3] 주행 환경별 통신 및 모드 설정
# car_id는 노드 초기화 시 전달받아 f-string으로 처리 예정
ENV_CONFIG = {
    "REAL": {
        "dt": 0.02,
        "topic_pose": "pose",      # 실차는 로컬 네임스페이스 사용 가능
        "topic_cmd": "cmd_vel",
        "msg_type": "Twist",
        "use_ma_input": False      # 실차는 명령단 MA 미적용
    },
    "SIM": {
        "dt": 0.05,
        "topic_pose": "/CAV_{id:02d}",        # 두 자리 포맷팅 템플릿
        "topic_cmd": "/CAV_{id:02d}_accel",   # 두 자리 포맷팅 템플릿
        "msg_type": "Accel",
        "use_ma_input": True                  # 시뮬은 명령단 MA 적용
    }
}

USE_PREDICTION = True