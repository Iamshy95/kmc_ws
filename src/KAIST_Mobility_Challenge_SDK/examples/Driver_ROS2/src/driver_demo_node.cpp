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

