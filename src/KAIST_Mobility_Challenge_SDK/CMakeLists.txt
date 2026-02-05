cmake_minimum_required(VERSION 3.16)
project(KMC_HARDWARE_uart_sdk LANGUAGES CXX)

option(KMC_HARDWARE_UART_BUILD_EXAMPLES "Build SDK examples" ON)

find_package(Threads REQUIRED)

add_library(KMC_HARDWARE_uart
  src/KMC_serial_port.cpp
  src/KMC_uart_client.cpp
  src/KMC_driver.cpp
)

target_include_directories(KMC_HARDWARE_uart
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:inc>
)

target_compile_features(KMC_HARDWARE_uart PUBLIC cxx_std_17)
target_link_libraries(KMC_HARDWARE_uart PUBLIC Threads::Threads)

if(KMC_HARDWARE_UART_BUILD_EXAMPLES)
  # Driver
  add_executable(driver_demo examples/Driver_Basic/driver_demo.cpp)
  target_link_libraries(driver_demo KMC_HARDWARE_uart)

  add_executable(driver_observe examples/Driver_Intermediate/driver_observe.cpp)
  target_link_libraries(driver_observe KMC_HARDWARE_uart)

  add_executable(driver_read_allstate examples/Driver_Intermediate/driver_read_allstate.cpp)
  target_link_libraries(driver_read_allstate KMC_HARDWARE_uart)

  add_executable(high_rate_control examples/Driver_Intermediate/high_rate_control.cpp)
  target_link_libraries(high_rate_control KMC_HARDWARE_uart)

  # UartClient
  add_executable(read_speed examples/UartClient_Advanced/read_speed.cpp)
  target_link_libraries(read_speed KMC_HARDWARE_uart)

  add_executable(read_battery examples/UartClient_Advanced/read_battery.cpp)
  target_link_libraries(read_battery KMC_HARDWARE_uart)

  add_executable(read_allstate examples/UartClient_Advanced/read_allstate.cpp)
  target_link_libraries(read_allstate KMC_HARDWARE_uart)

  add_executable(teleop_and_speed examples/UartClient_Advanced/teleop_and_speed.cpp)
  target_link_libraries(teleop_and_speed KMC_HARDWARE_uart)
endif()

