#!/bin/bash

# 1. 환경 설정 적용 (절대 경로)
source /opt/ros/foxy/setup.bash
source ~/kmc_ws/install/setup.bash

# 2. 시뮬레이터 실행 위치로 이동
cd ~/kmc_ws/src/mobility_challenge_simulator

# 3. Domain ID 설정
export ROS_DOMAIN_ID=100

# 4. 시뮬레이터 실행
ros2 launch simulator_launch simulator_launch.py
EOF