cmake_minimum_required(VERSION 3.8)
project(kmc_hardware_driver_node)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

if(NOT DEFINED KMC_HARDWARE_UART_SDK_ROOT)
  get_filename_component(KMC_HARDWARE_UART_SDK_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()

set(KMC_HARDWARE_UART_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${KMC_HARDWARE_UART_SDK_ROOT}
  ${CMAKE_CURRENT_BINARY_DIR}/kmc_hardware_uart_sdk)

add_executable(kmc_hardware_driver_demo_node src/driver_demo_node.cpp)
target_link_libraries(kmc_hardware_driver_demo_node KMC_HARDWARE_uart)
ament_target_dependencies(kmc_hardware_driver_demo_node rclcpp geometry_msgs std_msgs)
target_compile_features(kmc_hardware_driver_demo_node PUBLIC cxx_std_17)

add_executable(kmc_hardware_driver_observe_node src/driver_observe_node.cpp)
target_link_libraries(kmc_hardware_driver_observe_node KMC_HARDWARE_uart)
ament_target_dependencies(kmc_hardware_driver_observe_node rclcpp geometry_msgs std_msgs)
target_compile_features(kmc_hardware_driver_observe_node PUBLIC cxx_std_17)

add_executable(kmc_hardware_driver_read_allstate_node src/driver_read_allstate_node.cpp)
target_link_libraries(kmc_hardware_driver_read_allstate_node KMC_HARDWARE_uart)
ament_target_dependencies(kmc_hardware_driver_read_allstate_node rclcpp geometry_msgs std_msgs)
target_compile_features(kmc_hardware_driver_read_allstate_node PUBLIC cxx_std_17)

add_executable(kmc_hardware_high_rate_control_node src/high_rate_control_node.cpp)
target_link_libraries(kmc_hardware_high_rate_control_node KMC_HARDWARE_uart)
ament_target_dependencies(kmc_hardware_high_rate_control_node rclcpp geometry_msgs std_msgs)
target_compile_features(kmc_hardware_high_rate_control_node PUBLIC cxx_std_17)


#### Joystick Control Node
install(PROGRAMS
  joystick/kmc_web_joystick_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME kmc_web_joystick_node
)

install(DIRECTORY joystick/web
  DESTINATION share/${PROJECT_NAME}
)
####

install(TARGETS
  kmc_hardware_driver_demo_node
  kmc_hardware_driver_observe_node
  kmc_hardware_driver_read_allstate_node
  kmc_hardware_high_rate_control_node

  DESTINATION lib/${PROJECT_NAME})

ament_package()
