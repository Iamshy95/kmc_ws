#include "KMC_driver.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

int main(int argc, char **argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";
  int seconds = (argc > 2) ? std::atoi(argv[2]) : 5;
  if (seconds <= 0) seconds = 5;

  KMC_HARDWARE::Driver drv;
  KMC_HARDWARE::Driver::Options opt;
  opt.port = port;
  opt.serial.baudrate = 115200;

  opt.control_rate_hz = 100.0;
  opt.vehicle_speed_rate_hz = 1.0;
  opt.command_timeout_ms = 0;

  if (!drv.start(opt)) {
    std::fprintf(stderr, "Failed to open %s\n", port.c_str());
    return 2;
  }

  const auto t0 = std::chrono::steady_clock::now();
  auto next_status = t0;

  drv.setCommand(0.5f, 0.0f);

  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(seconds)) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_status) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
      std::printf("[running] t=%llds cmd=0.5 m/s\n",
                  static_cast<long long>(elapsed));
      next_status += std::chrono::seconds(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  drv.setCommand(0.0f, 0.0f);
  drv.stop();
  return 0;
}

