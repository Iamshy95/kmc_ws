#!/bin/bash
# 에러 발생 시 즉시 중단
set -e

# 1. ROS2 Foxy 기본 환경 설정
source "/opt/ros/foxy/setup.bash"

# 2. 내 워크스페이스 환경 설정 (빌드된 게 있다면)
if [ -f "/home/user/kmc_ws/install/setup.bash" ]; then
    source "/home/user/kmc_ws/install/setup.bash"
fi

# 3. DDS 및 네트워크 설정 (통신 결함 방지)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=100
export ROS_LOCALHOST_ONLY=0

# 4. 컨테이너 실행 시 입력받은 명령어를 그대로 실행 (없으면 bash 실행)
exec "$@"