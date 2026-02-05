cmake_minimum_required(VERSION 3.5)
project(kmc_hardware)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. ROS 2 기본 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# 2. SDK 헤더 파일 경로 연결
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../KAIST_Mobility_Challenge_SDK/inc
)

# ---------------------------------------------------------
# 3. 기존 주행 노드 빌드 (driver_node.cpp)
# ---------------------------------------------------------
add_executable(kmc_driver_node src/driver_node.cpp)
target_link_libraries(kmc_driver_node
  ${CMAKE_CURRENT_SOURCE_DIR}/../KAIST_Mobility_Challenge_SDK/build/libKMC_HARDWARE_uart.a
)
ament_target_dependencies(kmc_driver_node
  rclcpp
  geometry_msgs
  std_msgs
  nav_msgs
)


# 5. 설치 경로 설정 (driver_stop 제거)
install(TARGETS
  kmc_driver_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()