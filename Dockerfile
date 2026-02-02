# 1단계: 베이스 이미지 설정 (ROS 2 Foxy)
FROM osrf/ros:foxy-desktop

# 2단계: 필수 시스템 패키지 및 CycloneDDS 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    libserial-dev \
    ros-foxy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*


# 3단계: 파이썬 의존성 설치 (Numpy 강제 업데이트 포함)
RUN pip3 install --upgrade pip && \
    pip3 install "numpy>=1.21.0" "pandas<2.0.0"

# 4단계: 환경 변수 설정 (DDS 통신 및 도메인 고정)
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=100
ENV ROS_LOCALHOST_ONLY=0
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 5단계: 작업 디렉토리 설정
WORKDIR /home/user/kmc_ws

# 6단계: 소스 코드 및 엔트리포인트 복사
COPY ./src ./src
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 7단계: 도커 내부용 Alias(별칭) 설정
RUN echo "alias sb='source /opt/ros/foxy/setup.bash && source /home/user/kmc_ws/install/setup.bash'" >> /root/.bashrc && \
    echo "alias cb='colcon build --symlink-install'" >> /root/.bashrc

# 8단계: 초기 빌드 실행
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && \
    colcon build --symlink-install"

# 기본 실행 환경 (엔트리포인트 적용)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]