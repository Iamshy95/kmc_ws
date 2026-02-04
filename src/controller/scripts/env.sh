#!/bin/bash
# ROS 2 Humble 환경 소싱 (버전이 다를 경우 humble을 foxy 등으로 수정)
source /opt/ros/foxy/setup.bash

# 워크스페이스 설치 환경 소싱
source ~/kmc_ws/install/setup.bash

# 도메인 ID 설정 (차량 번호 22)
export ROS_DOMAIN_ID=100

echo " ROS 2 Environment Sourced (Domain ID: $ROS_DOMAIN_ID)"