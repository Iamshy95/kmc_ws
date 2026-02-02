#!/bin/bash

# 1. Domain ID 설정
export ROS_DOMAIN_ID=100

# 2. 전체 시스템 런치 파일 실행
ros2 launch controller sim_all.launch.py