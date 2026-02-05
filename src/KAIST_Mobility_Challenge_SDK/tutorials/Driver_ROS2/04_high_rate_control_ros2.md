# Tutorial: [Driver ROS2] high_rate_control 노드 구성

이 튜토리얼은 `high_rate_control` 예제를 ROS2 노드로 옮겨
1 kHz 제어 스트림과 속도 수신을 다루는 구조를 설명합니다.

## 목표
1. 고속 제어 주기로 Driver 실행
2. `/cmd_vel` subscribe으로 `setCommand(v, omega)` 업데이트
3. 속도 데이터를 `/vehicle_speed`로 publish
4. 초당 `setCommand` 호출 횟수를 `/cmd_updates`로 publish

---

## 구현 가이트

### 헤더 구성

```cpp
#include "KMC_driver.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <string>
```

---

### 노드 클래스 

```cpp
class KmcHardwareHighRateControlNode final : public rclcpp::Node {
public:
  KmcHardwareHighRateControlNode() : Node("kmc_hardware_high_rate_control_node") {
  }

  ~KmcHardwareHighRateControlNode() override { driver_.stop(); }

private:
  KMC_HARDWARE::Driver driver_;
};
```

---

### 멤버 변수

```cpp
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr cmd_update_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  std::string port_;
  int baud_{1000000};
  double control_rate_hz_{1000.0};
  double vehicle_speed_rate_hz_{100.0};
  int command_timeout_ms_{-1};
  double command_refresh_hz_{50.0};
  int realtime_priority_{70};
  int cpu_affinity_{-1};

  std::atomic<uint32_t> cmd_updates_{0};
  float last_cmd_v_{0.0f};
  float last_cmd_w_{0.0f};
  std::optional<rclcpp::Time> last_cmd_received_;
```

---

### 파라미터 선언

```cpp
    port_ = declare_parameter<std::string>("port", "/dev/ttyKMC");
    baud_ = declare_parameter<int>("baud", 1000000);
    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 1000.0);
    vehicle_speed_rate_hz_ = declare_parameter<double>("vehicle_speed_rate_hz", 100.0);
    command_timeout_ms_ = declare_parameter<int>("command_timeout_ms", -1);
    command_refresh_hz_ = declare_parameter<double>("command_refresh_hz", 50.0);
    realtime_priority_ = declare_parameter<int>("realtime_priority", 70);
    cpu_affinity_ = declare_parameter<int>("cpu_affinity", -1);
```

---

### 타임아웃 자동 계산
`command_timeout_ms=-1`이면 `command_refresh_hz` 기준으로 자동 계산합니다.

```cpp
    if (command_timeout_ms_ < 0 && command_refresh_hz_ > 0.0) {
      const double period_ms = 1000.0 / command_refresh_hz_;
      command_timeout_ms_ = static_cast<int>(std::max(50.0, period_ms * 2.0));
    }
```

---

### Driver 옵션 구성 + 시작

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

### 토픽 연결 + cmd 업데이트 카운트

```cpp
    speed_pub_ = create_publisher<std_msgs::msg::Float32>("vehicle_speed", 10);
    cmd_update_pub_ = create_publisher<std_msgs::msg::UInt32>("cmd_updates", 10);

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          const float v = static_cast<float>(msg->linear.x);
          const float w = static_cast<float>(msg->angular.z);
          last_cmd_v_ = v;
          last_cmd_w_ = w;
          last_cmd_received_ = now();
          driver_.setCommand(v, w);
          cmd_updates_.fetch_add(1, std::memory_order_relaxed);
        });
```

---

### 수신 큐 drain + 상태 타이머
속도 메시지를 publish 하고, 1초마다 업데이트 횟수를 publish합니다.

```cpp
    using namespace std::chrono_literals;
    rx_timer_ = create_wall_timer(1ms, [this]() { drainDriverQueue(); });

    if (command_refresh_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / command_refresh_hz_));
      cmd_timer_ = create_wall_timer(period, [this]() {
        if (!last_cmd_received_.has_value()) return;
        driver_.setCommand(last_cmd_v_, last_cmd_w_);
        cmd_updates_.fetch_add(1, std::memory_order_relaxed);
      });
    }

    status_timer_ = create_wall_timer(1s, [this]() {
      const uint32_t count = cmd_updates_.exchange(0, std::memory_order_relaxed);
      std_msgs::msg::UInt32 out;
      out.data = count;
      cmd_update_pub_->publish(out);
    });
```

`drainDriverQueue()`는 속도 메시지만 처리합니다.

```cpp
  void drainDriverQueue() {
    for (int i = 0; i < 200; ++i) {
      auto msg = driver_.tryPopMessage();
      if (!msg) break;
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std_msgs::msg::Float32 out;
        out.data = vs->mps;
        speed_pub_->publish(out);
      }
    }
  }
```

---

### main 함수

```cpp
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<KmcHardwareHighRateControlNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("kmc_hardware_high_rate_control_node"),
                 "Fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
```

---

전체 코드는 `examples/Driver_ROS2/src/high_rate_control_node.cpp` 참고.
```cpp
#include "KMC_driver.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <string>

class KmcHardwareHighRateControlNode final : public rclcpp::Node {
public:
  KmcHardwareHighRateControlNode() : Node("kmc_hardware_high_rate_control_node") {
    port_ = declare_parameter<std::string>("port", "/dev/ttyKMC");
    baud_ = declare_parameter<int>("baud", 1000000);
    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 1000.0);
    vehicle_speed_rate_hz_ = declare_parameter<double>("vehicle_speed_rate_hz", 100.0);
    command_timeout_ms_ = declare_parameter<int>("command_timeout_ms", -1);
    command_refresh_hz_ = declare_parameter<double>("command_refresh_hz", 50.0);
    realtime_priority_ = declare_parameter<int>("realtime_priority", 70);
    cpu_affinity_ = declare_parameter<int>("cpu_affinity", -1);

    if (command_timeout_ms_ < 0 && command_refresh_hz_ > 0.0) {
      const double period_ms = 1000.0 / command_refresh_hz_;
      command_timeout_ms_ = static_cast<int>(std::max(50.0, period_ms * 2.0));
    }

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

    speed_pub_ = create_publisher<std_msgs::msg::Float32>("vehicle_speed", 10);
    cmd_update_pub_ = create_publisher<std_msgs::msg::UInt32>("cmd_updates", 10);

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          const float v = static_cast<float>(msg->linear.x);
          const float w = static_cast<float>(msg->angular.z);
          last_cmd_v_ = v;
          last_cmd_w_ = w;
          last_cmd_received_ = now();
          driver_.setCommand(v, w);
          cmd_updates_.fetch_add(1, std::memory_order_relaxed);
        });

    using namespace std::chrono_literals;
    rx_timer_ = create_wall_timer(1ms, [this]() { drainDriverQueue(); });

    if (command_refresh_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / command_refresh_hz_));
      cmd_timer_ = create_wall_timer(period, [this]() {
        if (!last_cmd_received_.has_value()) return;
        driver_.setCommand(last_cmd_v_, last_cmd_w_);
        cmd_updates_.fetch_add(1, std::memory_order_relaxed);
      });
    }

    status_timer_ = create_wall_timer(1s, [this]() {
      const uint32_t count = cmd_updates_.exchange(0, std::memory_order_relaxed);
      std_msgs::msg::UInt32 out;
      out.data = count;
      cmd_update_pub_->publish(out);
    });
  }

  ~KmcHardwareHighRateControlNode() override { driver_.stop(); }

private:
  void drainDriverQueue() {
    for (int i = 0; i < 200; ++i) {
      auto msg = driver_.tryPopMessage();
      if (!msg) break;
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std_msgs::msg::Float32 out;
        out.data = vs->mps;
        speed_pub_->publish(out);
      }
    }
  }

private:
  KMC_HARDWARE::Driver driver_;

  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr cmd_update_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  std::string port_;
  int baud_{1000000};
  double control_rate_hz_{1000.0};
  double vehicle_speed_rate_hz_{100.0};
  int command_timeout_ms_{-1};
  double command_refresh_hz_{50.0};
  int realtime_priority_{70};
  int cpu_affinity_{-1};

  std::atomic<uint32_t> cmd_updates_{0};
  float last_cmd_v_{0.0f};
  float last_cmd_w_{0.0f};
  std::optional<rclcpp::Time> last_cmd_received_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<KmcHardwareHighRateControlNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("kmc_hardware_high_rate_control_node"),
                 "Fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}

```
