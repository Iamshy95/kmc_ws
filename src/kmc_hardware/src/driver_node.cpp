#include "KMC_driver.hpp"
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32.hpp>
#include <chrono>
#include <mutex>

using namespace std::chrono_literals;

class KmcFastDriver : public rclcpp::Node {
public:
    KmcFastDriver() : Node("kmc_fast_driver") {
        this->declare_parameter("port", "/dev/ttyUSB0");
        this->declare_parameter("baud", 115200);
        this->declare_parameter("car_id", 22);

        std::string port = this->get_parameter("port").as_string();
        int baud = this->get_parameter("baud").as_int();
        int car_id = this->get_parameter("car_id").as_int();

        if (!driver_.start({port, baud})) {
            RCLCPP_ERROR(this->get_logger(), "포트 연결 실패: %s", port.c_str());
            return;
        }

        std::string id_str = (car_id < 10 ? "0" : "") + std::to_string(car_id);
        std::string base_topic = "/CAV_" + id_str;

        speed_pub_   = this->create_publisher<std_msgs::msg::Float32>(base_topic + "/vehicle_speed", 10);
        battery_pub_ = this->create_publisher<std_msgs::msg::Float32>(base_topic + "/battery_voltage", 10);
        cmd_echo_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(base_topic + "/cmd_echo", 10);
        
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            base_topic + "/cmd_vel", 10, std::bind(&KmcFastDriver::cmdVelCallback, this, std::placeholders::_1));

        rx_timer_ = this->create_wall_timer(50ms, std::bind(&KmcFastDriver::receiveLoop, this));
        
        // [수정 완료] 100ms마다 체크, 5초(5000ms) 동안 명령 없으면 정지
        watchdog_timer_ = this->create_wall_timer(100ms, std::bind(&KmcFastDriver::watchdogLoop, this));

        last_cmd_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "=== KMC 고성능 드라이버 가동 (Watchdog 5s) ===");
    }

    ~KmcFastDriver() { 
        driver_.setCommand(0.0f, 0.0f);
        driver_.stop(); 
    }

private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        // 이벤트 기반: 즉시 전송
        driver_.setCommand((float)msg->linear.x, (float)msg->angular.z);
        last_cmd_time_ = this->now();
        is_active_ = true;
    }

    void watchdogLoop() {
        auto now = this->now();
        std::lock_guard<std::mutex> lock(mutex_);

        // 5000ms(5초) 이상 명령이 없을 때만 작동
        if (is_active_ && (now - last_cmd_time_) > 5000ms) {
            driver_.setCommand(0.0f, 0.0f);
            is_active_ = false;
            RCLCPP_WARN(this->get_logger(), "통신 두절 감지(5s): 차량을 정지합니다.");
        }
    }

    void receiveLoop() {
        while (auto msg = driver_.tryPopMessage()) {
            if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
                std_msgs::msg::Float32 out;
                out.data = vs->mps;
                speed_pub_->publish(out);
            }
            else if (auto* bv = std::get_if<KMC_HARDWARE::BatteryVoltage>(&*msg)) {
                std_msgs::msg::Float32 out;
                out.data = bv->volt;
                battery_pub_->publish(out);
            }
        }
    }

    KMC_HARDWARE::Driver driver_;
    std::mutex mutex_;
    rclcpp::Time last_cmd_time_;
    bool is_active_{false};
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr battery_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    rclcpp::TimerBase::SharedPtr rx_timer_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KmcFastDriver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}