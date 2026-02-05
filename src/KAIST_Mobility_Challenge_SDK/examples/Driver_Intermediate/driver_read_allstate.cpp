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
  opt.vehicle_speed_rate_hz = 1.0;

  opt.poll_allstate = true;
  opt.allstate_rate_hz = 10.0;
  opt.allstate_motor_left = 0;
  opt.allstate_motor_right = 1;

  opt.command_timeout_ms = 0;

  if (!drv.start(opt)) {
    std::fprintf(stderr, "Failed to open %s\n", port.c_str());
    return 2;
  }

  drv.setCommand(0.0f, 0.0f);

  const auto t0 = std::chrono::steady_clock::now();

  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(seconds)) {
    while (auto msg = drv.tryPopMessage()) {
      if (auto* st = std::get_if<KMC_HARDWARE::AllState>(&*msg)) {
        std::printf(
            "AllState: id=%u pos=%.2f deg rpm=%.1f current=%.2f A temp=%.1f C "
            "err=0x%08X\n",
            st->id,
            st->position_deg,
            st->speed_rpm,
            st->current_A,
            st->temperature_C,
            st->error_code);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  drv.setCommand(0.0f, 0.0f);
  drv.stop();
  return 0;
}

