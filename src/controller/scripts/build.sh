#!/bin/bash
# 1. 워크스페이스 최상위로 이동
cd ~/kmc_ws

# 2. SDK 빌드 (기존에 따로 빌드해야 했던 부분 자동화)
echo "🛠️ Building SDK..."
cd src/KAIST_Mobility_Challenge_SDK
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 3. 워크스페이스 최상위로 복귀 후 ROS 2 패키지 빌드
echo "📦 Building ROS 2 Packages..."
cd ~/kmc_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "✅ All Build Processes Complete!"