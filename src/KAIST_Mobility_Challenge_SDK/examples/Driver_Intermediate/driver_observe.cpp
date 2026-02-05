#include "KMC_driver.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <variant>

int main(int argc, char** argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";
  int seconds = (argc > 2) ? std::atoi(argv[2]) : 5;
  if (seconds <= 0) seconds = 5;

  KMC_HARDWARE::Driver drv;
  KMC_HARDWARE::Driver::Options opt;
  opt.port = port;
  opt.serial.baudrate = 115200;

  opt.control_rate_hz = 100.0;
  opt.vehicle_speed_rate_hz = 50.0;
  opt.poll_battery = true;
  opt.battery_rate_hz = 1.0;
  opt.command_timeout_ms = 200;

  if (!drv.start(opt)) {
    std::fprintf(stderr, "Failed to open %s\n", port.c_str());
    return 2;
  }

  const auto t0 = std::chrono::steady_clock::now();
  auto next_cmd = t0;
  const auto cmd_period =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / 50.0));

  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(seconds)) {
    const auto now = std::chrono::steady_clock::now();

    if (now >= next_cmd) {
      drv.setCommand(0.5f, 0.0f);
      next_cmd += cmd_period;
      if (next_cmd <= now) {
        next_cmd = now + cmd_period;
      }
    }

    while (auto msg = drv.tryPopMessage()) {
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std::printf("VehicleSpeed: %.3f m/s\n", vs->mps);
      } else if (auto* bv = std::get_if<KMC_HARDWARE::BatteryVoltage>(&*msg)) {
        std::printf("BatteryVoltage: %.2f V\n", bv->volt);
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  drv.setCommand(0.0f, 0.0f);
  drv.stop();
  return 0;
}

