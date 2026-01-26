import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # <--- 이 녀석이 있는지 확인해 주세요!
import glob
import shutil
from datetime import datetime

# [1] 설정 구역 - 이 변수들이 파일 상단에 정의되어 있어야 합니다.
HOME = os.path.expanduser("~")
LOG_ROOT = os.path.join(HOME, "kmc_ws/src/controller/logs")
PATH_DIR = os.path.join(HOME, "kmc_ws/src/controller/path")
RESULT_ROOT = os.path.join(LOG_ROOT, "analysis_results")
# 차량 및 도로 제원 (Rigid Body 기준)
CAR_L, CAR_W, LANE_HALF_WIDTH = 0.33, 0.16, 0.12


# 분석 기준
VIBRATION_WINDOW, VIBRATION_THRESHOLD = 1.0, 3  # 1초 내 3회 flip


def analyze_file(file_path):
    print(f"\n🚀 분석 시도: {os.path.basename(file_path)}")
    fname = os.path.basename(file_path)
    parts = fname.replace(".csv", "").split("_")
    
    if len(parts) < 4:
        print(f"⚠️ 파일명 형식 미달(log_{{경로}}_{{환경}}_{{시간}}...): {fname}")
        return
    
    path_name = parts[1]
    env = parts[2]
    
    try:
        # 데이터 로드 및 컬럼명 공백 제거
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if df.empty: return
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return

    # [0] 파라미터 미리 추출 (스냅샷 및 계산용)
    p = df.iloc[0]
    
    # 컬럼 매핑 (사용자 로그 구성 반영: current_v, final_omega, battery_voltage)
    v_cmd_col = 'cmd_v'
    w_cmd_col = 'cmd_w'
    v_act_col = 'actual_v'
    batt_col = 'battery'

    # 경로 파일 로드 (배경 시각화용)
    ref_path_file = os.path.join(PATH_DIR, f"{path_name}.csv")
    ref_df = pd.read_csv(ref_path_file) if os.path.exists(ref_path_file) else None

    # [1] 강체 모델 이탈 분석 (0% 무결성 기준)
    # L=0.33, W=0.16 기준으로 차체 귀퉁이의 최대 도달 거리 계산
    yaw_diff = df['motion_yaw'] - df['path_yaw']
    yaw_err = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    df['corner_reach'] = df['cte'].abs() + (CAR_L/2)*np.abs(np.sin(yaw_err)) + (CAR_W/2)*np.abs(np.cos(yaw_err))
    df['is_out'] = df['corner_reach'] > LANE_HALF_WIDTH
    total_out_count = df['is_out'].sum()
    status_str = "PASS" if total_out_count == 0 else "FAIL"

    # [2] 유해 진동 분석 (Sliding Window 1s / 3회 이상 flip)
    df['harmful_vibration'] = False
    for i in range(len(df)):
        c_time = df['time'].iloc[i]
        window = df[(df['time'] >= c_time - VIBRATION_WINDOW) & (df['time'] <= c_time)]
        if window['is_flip'].sum() >= VIBRATION_THRESHOLD:
            df.at[i, 'harmful_vibration'] = True
    vibration_ratio = (df['harmful_vibration'].sum() / len(df)) * 100

    # --- [3] 데이터 심층 진단 (환경 분기 및 로직 수정) ---
    p = df.iloc[0]
    
    # [A] 환경에 따른 속도 기준 설정 (시뮬레이터는 cmd_v, 실제는 actual_v)
    v_ref_col = 'cmd_v' if env == 'sim' else 'actual_v'
    
    # 1. 기초 주행 및 시간 통계
    total_time = df['time'].iloc[-1] - df['time'].iloc[0]
    total_dist = np.sum(np.sqrt(df['filt_px'].diff()**2 + df['filt_py'].diff()**2).dropna())
    avg_v = df[v_ref_col].mean()
    max_v = df[v_ref_col].max()
    v_target_ratio = (df[v_ref_col] >= p.get('p_v_max', 2.0) * 0.95).sum() / len(df) * 100
    dt_mean = df['dt'].mean()
    dt_delay_count = (df['dt'] > dt_mean * 1.5).sum()

    # 2. 구간별 정밀도 (직선/곡선 분리)
    is_curve = df['path_yaw'].diff().abs() > 0.005
    df_straight, df_curve = df[~is_curve], df[is_curve]
    rmse_straight = np.sqrt(np.mean(df_straight['cte']**2)) if not df_straight.empty else 0
    rmse_curve = np.sqrt(np.mean(df_curve['cte']**2)) if not df_curve.empty else 0

    # 3. 제어 안정성 (에너지 배분: 절댓값 합산 기반 비율)
    # 제곱(x^2) 대신 절댓값(|x|)을 사용하여 성분별 기여 강도를 왜곡 없이 계산
    e_pid = df['omega_pid'].abs().sum()
    e_ff = df['omega_ff'].abs().sum()
    e_yaw = df['omega_yaw'].abs().sum()
    e_total = e_pid + e_ff + e_yaw if (e_pid + e_ff + e_yaw) > 0 else 1
    r_pid, r_ff, r_yaw = (e_pid/e_total)*100, (e_ff/e_total)*100, (e_yaw/e_total)*100
    avg_slew_rate = (df['cmd_w'].diff().abs() / df['dt']).mean()

    # 4. 하드웨어 및 시스템 진단 (환경별 분기)
    if env == 'sim':
        latency_ms, slip_ratio, volt_drop, volt_cte_corr = 0.0, 1.0, 0.0, 0.0
    else:
        corrs = [df['cmd_v'].corr(df['actual_v'].shift(i)) for i in range(15)]
        latency_ms = np.argmax(corrs) * (dt_mean * 1000)
        slip_ratio = (df_curve['actual_v'] / df_curve['cmd_v']).mean() if not df_curve.empty else 1.0
        volt_drop = df['battery'].max() - df['battery'].min()
        volt_cte_corr = df['battery'].corr(df['cte'].abs())

    # 5. 센서 Yaw 신뢰도 (각도 차이 정규화 후 MAE 계산)
    # 단순히 뺀 게 아니라, -pi ~ pi 사이로 정규화하여 실제 '방향 이격'을 계산함
    yaw_diff_raw = df['filt_yaw'] - df['motion_yaw']
    yaw_diff_norm = np.arctan2(np.sin(yaw_diff_raw), np.cos(yaw_diff_raw))
    yaw_reliability = np.abs(yaw_diff_norm).mean() # 평균 절대 오차 (MAE)

    # --- [4] 폴더 생성 (수정된 avg_v 반영) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rmse_cte = np.sqrt(np.mean(df['cte']**2))
    folder_name = f"[{status_str}]_V{avg_v:.2f}_RMSE{rmse_cte:.3f}_{timestamp}"
    save_dir = os.path.join(RESULT_ROOT, env, path_name, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- [5] 시각화 (Matplotlib) ---
    # 01_Trajectory Map: 경로, 주행 궤적, 이탈 및 진동 지점 통합 시각화
    plt.figure(figsize=(12, 10))
    if ref_df is not None:
        plt.plot(ref_df.iloc[:,0].values, ref_df.iloc[:,1].values, 'k--', alpha=0.4, label='Reference Path', linewidth=1)
    
    plt.plot(df['filt_px'].values, df['filt_py'].values, 'b-', label='Actual Trajectory', alpha=0.7)
    
    # 강체 이탈 지점 (Orange X)
    out_pts = df[df['is_out']]
    if not out_pts.empty:
        plt.scatter(out_pts['filt_px'].values, out_pts['filt_py'].values, 
                    c='orange', marker='x', s=30, label=f'Lane Departure ({total_out_count} pts)', zorder=5)
    
    # 유해 진동 발생 지점 (Red Dots)
    vib_pts = df[df['harmful_vibration']]
    if not vib_pts.empty:
        plt.scatter(vib_pts['filt_px'].values, vib_pts['filt_py'].values, 
                    c='red', s=15, label=f'Harmful Vibration ({vibration_ratio:.1f}%)', zorder=6)
    
    plt.title(f"Trajectory Analysis - Status: {status_str}\n(Rigid Body Margin: {LANE_HALF_WIDTH}m)", fontsize=14)
    plt.xlabel("Global X (m)"); plt.ylabel("Global Y (m)")
    plt.axis('equal'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.savefig(os.path.join(save_dir, "01_map.png"), dpi=150); plt.close()

    # 02_Phase Portrait: CTE vs dCTE (제어 안정성 판별)
    df['dCTE'] = df['cte'].diff() / df['dt']
    plt.figure(figsize=(8, 8))
    plt.plot(df['cte'].values, df['dCTE'].values, 'g-', alpha=0.6, linewidth=1)
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    plt.title("Control Phase Portrait (CTE vs dCTE)", fontsize=12)
    plt.xlabel("Cross Track Error (m)"); plt.ylabel("CTE Rate of Change (m/s)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, "02_phase.png"), dpi=150); plt.close()

    # 03_Control Series: 속도 추종성 및 제어 성분 분석
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top: Velocity Tracking
    ax1.plot(df['time'].values, df['cmd_v'].values, 'r--', label='Target (Cmd V)', linewidth=1.5)
    ax1.plot(df['time'].values, df['actual_v'].values, 'b-', label='Actual (Echo V)', alpha=0.8)
    ax1.set_title(f"Velocity Tracking Performance (Env: {env})", fontsize=12)
    ax1.set_ylabel("Linear Velocity (m/s)"); ax1.legend(loc='lower right'); ax1.grid(True, alpha=0.4)

    # Bottom: Control Components (PID, FF, Yaw)
    ax2.plot(df['time'].values, df['omega_pid'].values, color='tab:blue', label='PID (Error Correction)', alpha=0.8)
    ax2.plot(df['time'].values, df['omega_ff'].values, color='tab:orange', label='FF (Feed-Forward)', alpha=0.8)
    ax2.plot(df['time'].values, df['omega_yaw'].values, color='tab:green', label='Yaw (Heading Corr)', alpha=0.8)
    ax2.set_title("Steering Command Components Magnitude", fontsize=12)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.legend(loc='upper right', ncol=3); ax2.grid(True, alpha=0.4)
    
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "03_series.png"), dpi=150); plt.close()

    # 04_CTE Histogram: 정밀도 분포 분석
    plt.figure(figsize=(8, 5))
    plt.hist(df['cte'].values, bins=60, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(rmse_cte, color='red', linestyle='--', label=f'RMSE: {rmse_cte:.4f}m')
    plt.title("Cross Track Error Distribution", fontsize=12)
    plt.xlabel("CTE (m)"); plt.ylabel("Frequency (Frames)")
    plt.legend(); plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "04_hist.png"), dpi=150); plt.close()

    # --- [6] 심층 Markdown 리포트 생성 (지표 보강 및 스냅샷 정밀화) ---
    total_frames = len(df)
    out_ratio = (total_out_count / total_frames) * 100
    total_vib_frames = df['harmful_vibration'].sum()
    
    report_md = f"""# 📊 주행 심층 분석 보고서 ({status_str})

## 1. 기본 주행 및 시간 통계
- **분석 환경:** {env.upper()} (기준 속도: {v_ref_col})
- **완주 시간:** {total_time:.2f} s | **총 주행 거리:** {total_dist:.2f} m
- **속도 통계:** 평균 {avg_v:.2f} m/s | 최고 {max_v:.2f} m/s
- **목표 속도 도달율:** {v_target_ratio:.1f} % | **루프 지연:** {dt_delay_count}회

## 2. 경로 추적 정밀도 (Accuracy)
- **전체 RMSE:** {rmse_cte:.4f} m (직선 {rmse_straight:.4f} / 곡선 {rmse_curve:.4f})
- **최대 이탈 폭 (Rigid Body):** {df['corner_reach'].max():.4f} m (기준 {LANE_HALF_WIDTH}m)
- **최종 판정:** {status_str} (총 {total_out_count} 프레임 이탈 / 전체의 {out_ratio:.2f}%)

## 3. 제어 안정성 (Stability)
- **유해 진동 비중:** {vibration_ratio:.2f} % (총 {total_vib_frames} 프레임 발생)
- **에너지 배분 (절댓값 합산):** PID {r_pid:.1f}% | FF {r_ff:.1f}% | Yaw보정 {r_yaw:.1f}%
- **평균 조향 변화율:** {avg_slew_rate:.4f} rad/s²

## 4. 하드웨어 및 시스템 진단 (Deep)
- **시스템 응답 지연:** {latency_ms:.1f} ms | **곡선 구간 슬립:** {slip_ratio*100:.1f} %
- **전압 변동:** {volt_drop:.3f} V (CTE 상관계수: {volt_cte_corr:.3f})
- **센서 신뢰도 (Yaw MAE):** {yaw_reliability:.4f} rad (Filt vs Motion)

## 5. 제어 파라미터 스냅샷 (Control Parameters)
```python
self.params = {{
    # 1. 조향 PID 제어
    "p_kp": {p.get('p_kp', 0)},
    "p_ki": {p.get('p_ki', 0)},
    "p_kd": {p.get('p_kd', 0)},
    "p_steer_deadzone": {p.get('p_steer_deadzone', 0)},

    # 2. 피드포워드(FF) 및 방향(Yaw) 보정
    "p_ff_gain": {p.get('p_ff_gain', 0)},
    "p_ff_window": {p.get('p_ff_window', 0)},
    "p_kyaw": {p.get('p_kyaw', 0)},

    # 3. 속도 프로파일 및 가감속 제약
    "p_v_max": {p.get('p_v_max', 0)},
    "p_v_min": {p.get('p_v_min', 0)},
    "p_v_accel": {p.get('p_v_accel', 0)},
    "p_v_decel": {p.get('p_v_decel', 0)},

    # 4. 동적 속도 페널티 계수
    "p_v_curve_gain": {p.get('p_v_curve_gain', 0)},
    "p_v_cte_gain": {p.get('p_v_cte_gain', 0)},

    # 5. 칼만 필터 게인 세분화
    "p_kf_q_pose": {p.get('p_kf_q_pose', 0)},
    "p_kf_r_pose": {p.get('p_kf_r_pose', 0)},
    "p_kf_q_yaw": {p.get('p_kf_q_yaw', 0)},
    "p_kf_r_yaw": {p.get('p_kf_r_yaw', 0)}
}}
"""
    with open(os.path.join(save_dir, "report.md"), "w", encoding="utf-8") as f: f.write(report_md)
    
    # [7] 원본 로그 이동 및 정리
    shutil.move(file_path, os.path.join(save_dir, fname))
    print(f"✅ 분석 완료 및 이동: {save_dir}")
    

if __name__ == "__main__":
    # sim과 real 폴더 모두 감시하여 CSV 파일 탐색
    for target in ["sim", "real"]:
        search_path = os.path.join(LOG_ROOT, target, "*.csv")
        files = glob.glob(search_path)
        print(f"🔍 {target} 폴더 검색 중... 발견된 파일: {len(files)}개")
        for f in files:
            analyze_file(f)