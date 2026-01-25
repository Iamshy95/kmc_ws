#include "KMC_driver.hpp"
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <chrono>
#include <variant>
#include <optional>  // 추가: std::optional 사용을 위함
#include <sstream>   // 추가: std::stringstream 사용을 위함

using namespace std::chrono_literals;

class KmcUnifiedDriver : public rclcpp::Node {
public:
    KmcUnifiedDriver() : Node("kmc_unified_driver") {
        // 1. 파라미터 설정
        this->declare_parameter("port", "/dev/ttyUSB0");
        this->declare_parameter("baud", 921600);
        this->declare_parameter("cmd_refresh_hz", 50.0);

        std::string port = this->get_parameter("port").as_string();
        int baud = this->get_parameter("baud").as_int();

        // 2. SDK 드라이버 시작 (구조체 인자 형식으로 수정)
        if (!driver_.start({port, baud})) {
            RCLCPP_ERROR(this->get_logger(), "포트를 열 수 없습니다: %s", port.c_str());
            return;
        }

        // 3. Publisher 설정
        speed_pub_ = this->create_publisher<std_msgs::msg::Float32>("vehicle_speed", 10);
        battery_pub_ = this->create_publisher<std_msgs::msg::Float32>("battery_voltage", 10);
        allstate_pub_ = this->create_publisher<std_msgs::msg::String>("allstate_text", 10);
        cmd_echo_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_echo", 10);

        // 4. Subscriber 설정
        cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10, std::bind(&KmcUnifiedDriver::cmdVelCallback, this, std::placeholders::_1));

        // 5. 타이머 설정
        double refresh_rate = this->get_parameter("cmd_refresh_hz").as_double();
        cmd_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / refresh_rate),
            std::bind(&KmcUnifiedDriver::sendCommandLoop, this));

        rx_timer_ = this->create_wall_timer(20ms, std::bind(&KmcUnifiedDriver::receiveLoop, this));

        RCLCPP_INFO(this->get_logger(), "통합 드라이버 노드가 시작되었습니다! (Port: %s)", port.c_str());
    }

    ~KmcUnifiedDriver() { driver_.stop(); }

private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        last_v_ = msg->linear.x;
        last_w_ = msg->angular.z;
        last_cmd_received_ = true;
        driver_.setCommand(last_v_, last_w_);
    }

    void sendCommandLoop() {
        if (last_cmd_received_) {
            driver_.setCommand(last_v_, last_w_);
            
            geometry_msgs::msg::Twist echo;
            echo.linear.x = last_v_;
            echo.angular.z = last_w_;
            cmd_echo_pub_->publish(echo);
        }
    }

    // 데이터 수신 루프 (drain_rx_queue 대신 tryPopMessage 사용으로 수정)
    void receiveLoop() {
        while (auto msg = driver_.tryPopMessage()) {
            // 1. 속도 정보
            if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
                std_msgs::msg::Float32 out;
                out.data = vs->mps;
                speed_pub_->publish(out);
            }
            // 2. 배터리 정보
            else if (auto* bv = std::get_if<KMC_HARDWARE::BatteryVoltage>(&*msg)) {
                std_msgs::msg::Float32 out;
                out.data = bv->volt;
                battery_pub_->publish(out);
            }
            // 3. 전체 상태 정보
            else if (auto* st = std::get_if<KMC_HARDWARE::AllState>(&*msg)) {
                std_msgs::msg::String out;
                std::stringstream ss;
                ss << "RPM: " << st->speed_rpm << " | Pos: " << st->position_deg 
                   << " | Curr: " << st->current_A << "A | Err: 0x" << std::hex << (int)st->error_code;
                out.data = ss.str();
                allstate_pub_->publish(out);
            }
        }
    }

    KMC_HARDWARE::Driver driver_;
    
    float last_v_{0.0f};
    float last_w_{0.0f};
    bool last_cmd_received_{false};

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speed_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr battery_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr allstate_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_echo_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    
    rclcpp::TimerBase::SharedPtr cmd_timer_;
    rclcpp::TimerBase::SharedPtr rx_timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KmcUnifiedDriver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}