#include "KMC_driver.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <chrono>
#include <optional>
#include <sstream>
#include <string>

class KmcHardwareDriverReadAllstateNode final : public rclcpp::Node {
public:
  KmcHardwareDriverReadAllstateNode() : Node("kmc_hardware_driver_read_allstate_node") {
    port_ = declare_parameter<std::string>("port", "/dev/ttyKMC");
    baud_ = declare_parameter<int>("baud", 115200);
    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 100.0);
    vehicle_speed_rate_hz_ = declare_parameter<double>("vehicle_speed_rate_hz", 1.0);
    allstate_hz_ = declare_parameter<double>("allstate_hz", 10.0);
    motor_left_ = declare_parameter<int>("motor_left", 0);
    motor_right_ = declare_parameter<int>("motor_right", 1);
    command_timeout_ms_ = declare_parameter<int>("command_timeout_ms", 0);
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

    opt.poll_allstate = allstate_hz_ > 0.0;
    opt.allstate_rate_hz = allstate_hz_ > 0.0 ? allstate_hz_ : 10.0;
    opt.allstate_motor_left = static_cast<uint8_t>(std::clamp(motor_left_, 0, 255));
    opt.allstate_motor_right = static_cast<uint8_t>(std::clamp(motor_right_, 0, 255));

    if (!driver_.start(opt)) {
      throw std::runtime_error("Failed to open UART port: " + port_);
    }

    allstate_pub_ = create_publisher<std_msgs::msg::String>("allstate_text", 10);
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          const float v = static_cast<float>(msg->linear.x);
          const float w = static_cast<float>(msg->angular.z);
          last_cmd_v_ = v;
          last_cmd_w_ = w;
          last_cmd_received_ = now();
          driver_.setCommand(v, w);
        });

    using namespace std::chrono_literals;
    rx_timer_ = create_wall_timer(1ms, [this]() { drainDriverQueue(); });

    if (command_refresh_hz_ > 0.0) {
      const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / command_refresh_hz_));
      cmd_timer_ = create_wall_timer(period, [this]() {
        if (!last_cmd_received_.has_value()) return;
        driver_.setCommand(last_cmd_v_, last_cmd_w_);
      });
    }
  }

  ~KmcHardwareDriverReadAllstateNode() override { driver_.stop(); }

private:
  void drainDriverQueue() {
    for (int i = 0; i < 100; ++i) {
      auto msg = driver_.tryPopMessage();
      if (!msg) break;

      if (auto* st = std::get_if<KMC_HARDWARE::AllState>(&*msg)) {
        std_msgs::msg::String out;
        std::ostringstream oss;
        oss << "id=" << st->id
            << " pos_deg=" << st->position_deg
            << " rpm=" << st->speed_rpm
            << " current_A=" << st->current_A
            << " temp_C=" << st->temperature_C
            << " err=0x" << std::hex << st->error_code;
        out.data = oss.str();
        allstate_pub_->publish(out);
      }
    }
  }

private:
  KMC_HARDWARE::Driver driver_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr allstate_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;

  std::string port_;
  int baud_{115200};
  double control_rate_hz_{100.0};
  double vehicle_speed_rate_hz_{1.0};
  double allstate_hz_{10.0};
  int motor_left_{0};
  int motor_right_{1};
  int command_timeout_ms_{0};
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
    auto node = std::make_shared<KmcHardwareDriverReadAllstateNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("kmc_hardware_driver_read_allstate_node"),
                 "Fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}

