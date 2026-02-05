#pragma once

#include "KMC_uart_client.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace KMC_HARDWARE {

class Driver {
public:
  struct Options {
    std::string port;
    SerialPortOptions serial;

    double control_rate_hz = 100.0;      // 0xA5 send rate (Hz)
    double vehicle_speed_rate_hz = 50.0; // 0xB3 request rate (Hz)

    bool poll_battery = false;
    double battery_rate_hz = 1.0;

    bool poll_allstate = false;
    double allstate_rate_hz = 10.0;
    uint8_t allstate_motor_left = 0;
    uint8_t allstate_motor_right = 1;

    int command_timeout_ms = 300; // 0 disables timeout
    int stop_burst_count = 3;     // repeat stop on shutdown

    // Thread hints (Linux only)
    // - realtime_priority: SCHED_FIFO priority (1~99). Use -1 to skip.
    // - cpu_affinity: pin I/O thread to this CPU index. Use -1 to skip.
    int realtime_priority = -1;
    int cpu_affinity = -1;

    size_t max_queue = 1024;
  };

  Driver();
  ~Driver();

  Driver(const Driver &) = delete;
  Driver &operator=(const Driver &) = delete;

  // Start the I/O thread and open the port.
  bool start(const Options &opt);
  // Stop the I/O thread and close the port (sends a safety stop).
  void stop();
  // Return true if the driver thread is running.
  bool isRunning() const { return running_.load(); }

  // Update the desired command (velocity, yaw rate).
  void setCommand(float velocity_mps, float omega_rps);
  // Helper for curvature-based commands.
  void setCommandCurvature(float velocity_mps, float curvature_1pm);

  // Non-blocking pop. Returns nullopt if the queue is empty.
  std::optional<Message> tryPopMessage();

  // Blocking pop with timeout. Returns true if a message was received.
  bool waitPopMessage(Message &out, int timeout_ms);

  // Request a oneshot battery read (0xAF).
  bool requestBatteryOnce(uint8_t motor_id = 0);
  // Request a oneshot all-state read (0xAF).
  bool requestAllStateOnce(uint8_t motor_id);

private:
  void pushMessage(const Message &m);

  void ioLoop();

  static double clampRate(double hz, double fallback);
  static std::chrono::nanoseconds periodFromHz(double hz, double fallback_hz);
  void applyThreadHints();

private:
  Options opt_;
  UartClient client_;

  std::atomic<bool> running_{false};

  // command state
  std::mutex cmd_mtx_;
  float cmd_v_ = 0.0f;
  float cmd_c_ = 0.0f;
  std::chrono::steady_clock::time_point cmd_last_update_;

  // queue
  std::mutex q_mtx_;
  std::condition_variable q_cv_;
  std::deque<Message> q_;

  // thread
  std::thread io_thread_;
};

} // namespace KMC_HARDWARE

