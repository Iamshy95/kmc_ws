# Tutorial: [Driver ROS2] driver_demo 노드 구성

이 튜토리얼은 ROS2 노드를 직접 구성해 `cmd_vel`을 `KMC_HARDWARE::Driver`로 전달하는 방식을 다룹니다.
기본 SDK의 `driver_demo`를 ROS2 토픽으로 옮긴 버전입니다.

## 목표
1. ROS2 노드에서 UART 포트를 열고 Driver 시작
2. `/cmd_vel`을 subscribe해 `setCommand(v, omega)` 호출
3. 마지막 명령을 주기적으로 재전송 
4. 현재 명령을 `/cmd_echo`로 다시 publish

---

## 준비물
- ROS2 환경 사용 가능
- `KMC_HARDWARE::Driver` 기본 동작 이해
- UART 장치 접근 권한 확보 (`/dev/ttyKMC`)

---

## 구현 가이트

### 헤더, 클래스 선언
Driver + ROS2 노드 + Twist 메시지를 씁니다.

```cpp
#include "KMC_driver.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <chrono>
#include <optional>
#include <string>
```

---

ROS2 노드 클래스는 아래처럼 둡니다.

```cpp
class KmcHardwareDriverDemoNode final : public rclcpp::Node {
public:
  KmcHardwareDriverDemoNode() : Node("kmc_hardware_driver_demo_node") {
  }

  ~KmcHardwareDriverDemoNode() override { driver_.stop(); }

private:
  KMC_HARDWARE::Driver driver_;
};
```

---

### 멤버 변수
파라미터와 토픽 핸들을 멤버로 보관합니다.

```cpp
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_echo_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;

  std::string port_;
  int baud_{115200};
  double control_rate_hz_{100.0};
  double vehicle_speed_rate_hz_{1.0};
  int command_timeout_ms_{200};
  double command_refresh_hz_{50.0};
  int realtime_priority_{-1};
  int cpu_affinity_{-1};

  float last_cmd_v_{0.0f};
  float last_cmd_w_{0.0f};
  std::optional<rclcpp::Time> last_cmd_received_;
```

---

### 파라미터 선언
Driver 옵션에 필요한 값들을 ROS 파라미터로 받습니다.

```cpp
    port_ = declare_parameter<std::string>("port", "/dev/ttyKMC");
    baud_ = declare_parameter<int>("baud", 115200);
    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 100.0);
    vehicle_speed_rate_hz_ = declare_parameter<double>("vehicle_speed_rate_hz", 1.0);
    command_timeout_ms_ = declare_parameter<int>("command_timeout_ms", 200);
    command_refresh_hz_ = declare_parameter<double>("command_refresh_hz", 50.0);
    realtime_priority_ = declare_parameter<int>("realtime_priority", -1);
    cpu_affinity_ = declare_parameter<int>("cpu_affinity", -1);
```

---

### Driver 옵션 구성 + 시작
ROS 파라미터를 `Driver::Options`에 매핑하고 `start()`를 호출합니다.

```cpp
    KMC_HARDWARE::Driver::Options opt;
    opt.port = port_;
    opt.serial.baudrate = baud_;
    opt.serial.hw_flow_control = true;
    opt.control_rate_hz = control_rate_hz_;
    opt.vehicle_speed_rate_hz = vehicle_speed_rate_hz_;
    opt.command_timeout_ms = command_timeout_ms_;
    opt.realtime_priority = realtime_priority_;
    opt.cpu_affinity = cpu_affinity_;

    if (!driver_.start(opt)) {
      throw std::runtime_error("Failed to open UART port: " + port_);
    }
```

---

### 토픽 (cmd_vel subscribe + cmd_echo publish)
`cmd_vel`을 받아 `setCommand(v, omega)`에 전달합니다.

```cpp
    cmd_echo_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_echo", 10);
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          const float v = static_cast<float>(msg->linear.x);
          const float w = static_cast<float>(msg->angular.z);
          last_cmd_v_ = v;
          last_cmd_w_ = w;
          last_cmd_received_ = now();
          driver_.setCommand(v, w);
          cmd_echo_pub_->publish(*msg);
        });
```

매핑 :
- `linear.x` -> `v` (m/s)
- `angular.z` -> `omega` (rad/s)

---

### 명령 재전송 타이머 (선택)
명령이 끊기지 않게 마지막 값을 주기적으로 업데이트합니다.

```cpp
    if (command_refresh_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / command_refresh_hz_));
      cmd_timer_ = create_wall_timer(period, [this]() {
        if (!last_cmd_received_.has_value()) return;
        driver_.setCommand(last_cmd_v_, last_cmd_w_);
        geometry_msgs::msg::Twist msg;
        msg.linear.x = last_cmd_v_;
        msg.angular.z = last_cmd_w_;
        cmd_echo_pub_->publish(msg);
      });
    }
```

- `command_refresh_hz = 0`이면 재전송 비활성.

---

### main 함수
노드 생성 후 spin으로 유지합니다.

```cpp
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<KmcHardwareDriverDemoNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("kmc_hardware_driver_demo_node"), "Fatal: %s",
                 e.what());
  }
  rclcpp::shutdown();
  return 0;
}
```

---

전체 코드는 `examples/Driver_ROS2/src/driver_demo_node.cpp` 참고.
```cpp
#include "KMC_driver.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <chrono>
#include <optional>
#include <string>

class KmcHardwareDriverDemoNode final : public rclcpp::Node {
public:
  KmcHardwareDriverDemoNode() : Node("kmc_hardware_driver_demo_node") {
    port_ = declare_parameter<std::string>("port", "/dev/ttyKMC");
    baud_ = declare_parameter<int>("baud", 115200);
    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 100.0);
    vehicle_speed_rate_hz_ = declare_parameter<double>("vehicle_speed_rate_hz", 1.0);
    command_timeout_ms_ = declare_parameter<int>("command_timeout_ms", 200);
    command_refresh_hz_ = declare_parameter<double>("command_refresh_hz", 50.0);
    realtime_priority_ = declare_parameter<int>("realtime_priority", -1);
    cpu_affinity_ = declare_parameter<int>("cpu_affinity", -1);

    KMC_HARDWARE::Driver::Options opt;
    opt.port = port_;
    opt.serial.baudrate = baud_;
    opt.serial.hw_flow_control = true;
    opt.control_rate_hz = control_rate_hz_;
    opt.vehicle_speed_rate_hz = vehicle_speed_rate_hz_;
    opt.command_timeout_ms = command_timeout_ms_;
    opt.realtime_priority = realtime_priority_;
    opt.cpu_affinity = cpu_affinity_;

    if (!driver_.start(opt)) {
      throw std::runtime_error("Failed to open UART port: " + port_);
    }

    cmd_echo_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_echo", 10);
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          const float v = static_cast<float>(msg->linear.x);
          const float w = static_cast<float>(msg->angular.z);
          last_cmd_v_ = v;
          last_cmd_w_ = w;
          last_cmd_received_ = now();
          driver_.setCommand(v, w);
          cmd_echo_pub_->publish(*msg);
        });

    if (command_refresh_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / command_refresh_hz_));
      cmd_timer_ = create_wall_timer(period, [this]() {
        if (!last_cmd_received_.has_value()) return;
        driver_.setCommand(last_cmd_v_, last_cmd_w_);
        geometry_msgs::msg::Twist msg;
        msg.linear.x = last_cmd_v_;
        msg.angular.z = last_cmd_w_;
        cmd_echo_pub_->publish(msg);
      });
    }
  }

  ~KmcHardwareDriverDemoNode() override { driver_.stop(); }

private:
  KMC_HARDWARE::Driver driver_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_echo_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;

  std::string port_;
  int baud_{115200};
  double control_rate_hz_{100.0};
  double vehicle_speed_rate_hz_{1.0};
  int command_timeout_ms_{200};
  double command_refresh_hz_{50.0};
  int realtime_priority_{-1};
  int cpu_affinity_{-1};

  float last_cmd_v_{0.0f};
  float last_cmd_w_{0.0f};
  std::optional<rclcpp::Time> last_cmd_received_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<KmcHardwareDriverDemoNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("kmc_hardware_driver_demo_node"), "Fatal: %s",
                 e.what());
  }
  rclcpp::shutdown();
  return 0;
}

```
