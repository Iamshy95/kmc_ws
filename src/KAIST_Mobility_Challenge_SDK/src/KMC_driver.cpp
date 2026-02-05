#include "KMC_driver.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace KMC_HARDWARE {

double Driver::clampRate(double hz, double fallback) {
  if (!std::isfinite(hz) || hz <= 0.0)
    return fallback;
  return hz;
}

std::chrono::nanoseconds Driver::periodFromHz(double hz, double fallback_hz) {
  const double clamped_hz = clampRate(hz, fallback_hz);
  const double ns = 1e9 / clamped_hz;
  const auto bounded = static_cast<long long>(std::max(1000.0, ns));
  return std::chrono::nanoseconds(bounded);
}

void Driver::applyThreadHints() {
#ifdef __linux__
  if (opt_.cpu_affinity >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(opt_.cpu_affinity), &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }

  if (opt_.realtime_priority > 0) {
    sched_param sch{};
    sch.sched_priority = opt_.realtime_priority;
    (void)pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch);
  }
#endif
}

Driver::Driver() { cmd_last_update_ = std::chrono::steady_clock::now(); }

Driver::~Driver() { stop(); }

bool Driver::start(const Options &opt) {
  stop();
  opt_ = opt;
  opt_.control_rate_hz = clampRate(opt_.control_rate_hz, 100.0);
  opt_.vehicle_speed_rate_hz = clampRate(opt_.vehicle_speed_rate_hz, 50.0);
  opt_.battery_rate_hz = clampRate(opt_.battery_rate_hz, 1.0);
  opt_.allstate_rate_hz = clampRate(opt_.allstate_rate_hz, 10.0);
  if (opt_.serial.baudrate <= 0)
    opt_.serial.baudrate = 1000000;

  if (!client_.open(opt_.port, opt_.serial)) {
    return false;
  }
  client_.flushInput();

  {
    std::lock_guard<std::mutex> lk(cmd_mtx_);
    cmd_v_ = 0.0f;
    cmd_c_ = 0.0f;
    cmd_last_update_ = std::chrono::steady_clock::now();
  }

  running_.store(true);
  io_thread_ = std::thread([this] { ioLoop(); });
  return true;
}

void Driver::stop() {
  bool was_running = running_.exchange(false);
  (void)was_running;

  if (io_thread_.joinable())
    io_thread_.join();

  if (client_.isOpen()) {
    for (int i = 0; i < std::max(1, opt_.stop_burst_count); ++i) {
      client_.sendPcControl(0.0f, 0.0f);
    }
  }
  client_.close();

  {
    std::lock_guard<std::mutex> lk(q_mtx_);
    q_.clear();
  }
}

void Driver::setCommand(float velocity_mps, float omega_rps) {
  const float abs_v = std::fabs(velocity_mps);
  const float curvature_1pm =
      (abs_v > 1e-3f) ? (omega_rps / velocity_mps) : 0.0f;
  setCommandCurvature(velocity_mps, curvature_1pm);
}

void Driver::setCommandCurvature(float velocity_mps, float curvature_1pm) {
  std::lock_guard<std::mutex> lk(cmd_mtx_);
  cmd_v_ = velocity_mps;
  cmd_c_ = curvature_1pm;
  cmd_last_update_ = std::chrono::steady_clock::now();
}

std::optional<Message> Driver::tryPopMessage() {
  std::lock_guard<std::mutex> lk(q_mtx_);
  if (q_.empty())
    return std::nullopt;
  Message m = std::move(q_.front());
  q_.pop_front();
  return m;
}

bool Driver::waitPopMessage(Message &out, int timeout_ms) {
  std::unique_lock<std::mutex> lk(q_mtx_);
  if (timeout_ms < 0)
    timeout_ms = 0;
  if (!q_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                      [this] { return !q_.empty() || !running_.load(); })) {
    return false;
  }
  if (q_.empty())
    return false;
  out = std::move(q_.front());
  q_.pop_front();
  return true;
}

bool Driver::requestBatteryOnce(uint8_t motor_id) {
  if (!running_.load())
    return false;
  return client_.requestBatteryVoltage(motor_id);
}

bool Driver::requestAllStateOnce(uint8_t motor_id) {
  if (!running_.load())
    return false;
  return client_.requestAllState(motor_id);
}

void Driver::pushMessage(const Message &m) {
  {
    std::lock_guard<std::mutex> lk(q_mtx_);
    if (q_.size() >= opt_.max_queue) {
      q_.pop_front();
    }
    q_.push_back(m);
  }
  q_cv_.notify_one();
}

void Driver::ioLoop() {
  applyThreadHints();

  const auto ctrl_period = periodFromHz(opt_.control_rate_hz, 100.0);
  const auto b3_period = periodFromHz(opt_.vehicle_speed_rate_hz, 50.0);
  const auto batt_period = opt_.poll_battery
                               ? periodFromHz(opt_.battery_rate_hz, 1.0)
                               : std::chrono::nanoseconds::max();
  const auto all_period = opt_.poll_allstate
                              ? periodFromHz(opt_.allstate_rate_hz, 10.0)
                              : std::chrono::nanoseconds::max();

  auto next_ctrl = std::chrono::steady_clock::now();
  auto next_b3 = next_ctrl;
  auto next_batt = opt_.poll_battery
                       ? next_ctrl
                       : std::chrono::steady_clock::time_point::max();
  auto next_all = opt_.poll_allstate
                      ? next_ctrl
                      : std::chrono::steady_clock::time_point::max();

  while (running_.load(std::memory_order_relaxed)) {
    auto now = std::chrono::steady_clock::now();

    // 0xA5 control
    int ctrl_sent = 0;
    while (now >= next_ctrl && ctrl_sent < 2) {
      float v, c;
      std::chrono::steady_clock::time_point last;
      {
        std::lock_guard<std::mutex> lk(cmd_mtx_);
        v = cmd_v_;
        c = cmd_c_;
        last = cmd_last_update_;
      }

      const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - last)
                              .count();
      if (opt_.command_timeout_ms > 0 && age_ms > opt_.command_timeout_ms) {
        v = 0.0f;
        c = 0.0f;
      }

      (void)client_.sendPcControl(v, c);
      next_ctrl += ctrl_period;
      ++ctrl_sent;
      now = std::chrono::steady_clock::now();
    }

    // 0xB3 request
    if (now >= next_b3) {
      (void)client_.requestVehicleSpeed();
      next_b3 += b3_period;
    }

    // 0xAF polling
    if (opt_.poll_battery && now >= next_batt) {
      (void)client_.requestBatteryVoltage(0);
      next_batt += batt_period;
    }

    if (opt_.poll_allstate && now >= next_all) {
      (void)client_.requestAllState(opt_.allstate_motor_left);
      (void)client_.requestAllState(opt_.allstate_motor_right);
      next_all += all_period;
    }

    // Drain any already buffered frames with zero wait (low latency).
    for (int i = 0; i < 16; ++i) {
      auto msg = client_.poll(0);
      if (!msg)
        break;
      pushMessage(*msg);
    }

    // Predictable sleep toward the next scheduled event (keeps 1 kHz jitter
    // small)
    const auto next_event = std::min({next_ctrl, next_b3, next_batt, next_all});
    auto wait = next_event - std::chrono::steady_clock::now();
    if (wait > std::chrono::nanoseconds::zero()) {
      if (wait > std::chrono::microseconds(200)) {
        // Leave a small margin to compensate for scheduler latency on Jetson
        std::this_thread::sleep_for(wait - std::chrono::microseconds(100));
      } else {
        // Short idle, still try to catch any just arrived bytes
        if (auto msg = client_.poll(0)) {
          pushMessage(*msg);
        }
      }
    }
  }
}

} // namespace KMC_HARDWARE

