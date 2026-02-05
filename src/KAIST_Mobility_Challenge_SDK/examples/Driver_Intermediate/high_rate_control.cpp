#include "KMC_driver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

int main(int argc, char **argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";
  int seconds = (argc > 2) ? std::atoi(argv[2]) : 5;
  if (seconds <= 0) seconds = 5;
  const int baud = (argc > 3) ? std::atoi(argv[3]) : 1000000;
  const double b3_rate = (argc > 4) ? std::atof(argv[4]) : 100.0;
  const double cmd_hz = (argc > 5) ? std::atof(argv[5]) : 50.0;

  KMC_HARDWARE::Driver::Options opt;
  opt.port = port;
  opt.serial.baudrate = baud > 0 ? baud : 1000000;
  opt.serial.hw_flow_control = true;

  opt.control_rate_hz = 1000.0;
  opt.vehicle_speed_rate_hz = b3_rate > 0 ? b3_rate : 100.0;

  if (cmd_hz > 0.0) {
    const double period_ms = 1000.0 / cmd_hz;
    opt.command_timeout_ms = static_cast<int>(std::max(50.0, period_ms * 2.0));
  } else {
    opt.command_timeout_ms = 0;
  }
  opt.stop_burst_count = 5;

  opt.realtime_priority = 70;
  opt.cpu_affinity = -1;

  KMC_HARDWARE::Driver drv;
  if (!drv.start(opt)) {
    std::fprintf(stderr, "Failed to open %s\n", port.c_str());
    return 2;
  }

  drv.setCommand(1.0f, 0.0f);

  const auto t0 = std::chrono::steady_clock::now();
  auto next_status = t0;
  auto next_cmd = t0;
  const auto cmd_period =
      (cmd_hz > 0.0)
          ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / cmd_hz))
          : std::chrono::steady_clock::duration::zero();
  size_t cmd_updates = 0;

  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(seconds)) {
    const auto now = std::chrono::steady_clock::now();

    if (cmd_hz > 0.0 && now >= next_cmd) {
      drv.setCommand(1.0f, 0.0f);
      ++cmd_updates;
      next_cmd += cmd_period;
      if (next_cmd <= now) {
        next_cmd = now + cmd_period;
      }
    }

    if (now >= next_status) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
      std::printf("[1kHz] t=%llds cmd_updates=%zu\n",
                  static_cast<long long>(elapsed), cmd_updates);
      next_status += std::chrono::seconds(1);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  drv.setCommand(0.0f, 0.0f);
  drv.stop();
  return 0;
}

