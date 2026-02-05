# Driver ROS2 Examples

ROS2 package that mirrors the Driver-based SDK examples with ROS topics.

## Build (colcon)

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2
colcon build --symlink-install
source install/setup.bash
```

```bash
colcon build --symlink-install \
  --cmake-args -DKMC_HARDWARE_UART_SDK_ROOT=/path/to/KAIST_Mobility_Challenge_SDK
```

## Nodes

- `kmc_hardware_driver_demo_node`: cmd_vel -> Driver setCommand, echo command.
- `kmc_hardware_driver_observe_node`: cmd_vel -> Driver, publish speed + battery.
- `kmc_hardware_driver_read_allstate_node`: cmd_vel -> Driver, publish allstate.
- `kmc_hardware_high_rate_control_node`: high-rate control stream, publish speed + cmd updates.

## Run

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 run kmc_hardware_driver_node kmc_hardware_driver_demo_node \
  --ros-args -p port:=/dev/ttyKMC -p baud:=115200

ros2 run kmc_hardware_driver_node kmc_hardware_driver_observe_node \
  --ros-args -p port:=/dev/ttyKMC -p baud:=115200 -p battery_hz:=1.0

ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
  --ros-args -p port:=/dev/ttyKMC -p baud:=115200 -p allstate_hz:=10.0

ros2 run kmc_hardware_driver_node kmc_hardware_high_rate_control_node \
  --ros-args -p port:=/dev/ttyKMC -p baud:=1000000
```

## Topics

### kmc_hardware_driver_demo_node
- Subscribe: `/cmd_vel` (`geometry_msgs/msg/Twist`)
- Publish: `/cmd_echo` (`geometry_msgs/msg/Twist`)

### kmc_hardware_driver_observe_node
- Subscribe: `/cmd_vel` (`geometry_msgs/msg/Twist`)
- Publish: `/vehicle_speed` (`std_msgs/msg/Float32`)
- Publish: `/battery_voltage` (`std_msgs/msg/Float32`)

### kmc_hardware_driver_read_allstate_node
- Subscribe: `/cmd_vel` (`geometry_msgs/msg/Twist`)
- Publish: `/allstate_text` (`std_msgs/msg/String`)

### kmc_hardware_high_rate_control_node
- Subscribe: `/cmd_vel` (`geometry_msgs/msg/Twist`)
- Publish: `/vehicle_speed` (`std_msgs/msg/Float32`)
- Publish: `/cmd_updates` (`std_msgs/msg/UInt32`)


## Common Parameters

- `port` (string, default `/dev/ttyKMC`)
- `baud` (int, default `115200` or `1000000` depending on node)
- `control_rate_hz` (double)
- `vehicle_speed_rate_hz` (double)
- `command_timeout_ms` (int)
- `command_refresh_hz` (double)
- `realtime_priority` (int)
- `cpu_affinity` (int)

## Node-Specific Parameters

- `kmc_hardware_driver_observe_node`: `battery_hz`
- `kmc_hardware_driver_read_allstate_node`: `allstate_hz`, `motor_left`, `motor_right`
- `kmc_hardware_high_rate_control_node`: `command_timeout_ms` set to `-1` auto-computes from `command_refresh_hz`

## Troubleshooting

If `ros2 run` shows "Package not found":

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 pkg list | grep kmc_hardware_driver_node
```



# Joystick

## Nodes
- `kmc_web_joystick_node`: Web joystick -> cmd_vel, publish tx/rx rates.

## Run
```bash
ros2 run kmc_hardware_driver_node kmc_web_joystick_node \
  --ros-args -p bind_address:=0.0.0.0 -p http_port:=8080
```

### Web joystick (kmc_web_joystick_node)

Start the web joystick server and open the page:

```bash
ros2 run kmc_hardware_driver_node kmc_web_joystick_node \
  --ros-args -p bind_address:=0.0.0.0 -p http_port:=8080
```

Open `http://localhost:8080` in a browser.

Useful parameters:
- `bind_address` (string, default `127.0.0.1`)
- `http_port` (int, default `8080`)
- `cmd_topic` (string, default `cmd_vel`)
- `command_timeout_ms` (int, default `200`)
- `linear_limit` / `angular_limit` (double)
- `deadzone` (double, default `0.05`)
- `web_root` (string, optional override for the `web` folder)

### kmc_web_joystick_node
- Publish: `/cmd_vel` (`geometry_msgs/msg/Twist`)
- Publish: `/cmd_vel_tx_rate_hz` (`std_msgs/msg/Float32`)
- Publish: `/driver_rx_rate_hz` (`std_msgs/msg/Float32`)
- Subscribe: `/vehicle_speed` (`std_msgs/msg/Float32`)
