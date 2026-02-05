#include "KMC_uart_client.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <variant>

int main(int argc, char **argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";
  int seconds = (argc > 2) ? std::atoi(argv[2]) : 5;
  if (seconds <= 0) seconds = 5;
  double rate_hz = (argc > 3) ? std::atof(argv[3]) : 20.0;
  if (rate_hz <= 0.0) rate_hz = 20.0;

  KMC_HARDWARE::UartClient client;
  if (!client.open(port, 115200)) {
    std::fprintf(stderr, "Failed to open port: %s\n", port.c_str());
    return 1;
  }
  client.flushInput();

  const auto period = std::chrono::duration<double>(1.0 / rate_hz);
  const auto t0 = std::chrono::steady_clock::now();

  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(seconds)) {
    const auto cycle_start = std::chrono::steady_clock::now();
    client.requestVehicleSpeed();

    auto msg = client.poll(50);
    if (msg) {
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std::printf("VehicleSpeed: %.3f m/s\n", vs->mps);
      }
    } else {
      std::printf("VehicleSpeed: (no response)\n");
    }

    const auto elapsed = std::chrono::steady_clock::now() - cycle_start;
    if (elapsed < period) {
      std::this_thread::sleep_for(period - elapsed);
    }
  }

  return 0;
}

