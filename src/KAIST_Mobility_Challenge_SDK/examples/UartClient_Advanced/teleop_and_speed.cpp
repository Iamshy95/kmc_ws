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

  KMC_HARDWARE::UartClient client;
  if (!client.open(port, 115200)) {
    std::fprintf(stderr, "Failed to open port: %s\n", port.c_str());
    return 1;
  }
  client.flushInput();

  const auto t0 = std::chrono::steady_clock::now();
  auto next_ctrl = t0;
  auto next_b3 = t0;
  const float v = 0.5f;
  const float k = 0.0f;

  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(seconds)) {
    const auto now = std::chrono::steady_clock::now();

    if (now >= next_ctrl) {
      client.sendPcControl(v, k);
      next_ctrl += std::chrono::milliseconds(10);
    }

    if (now >= next_b3) {
      client.requestVehicleSpeed();
      next_b3 += std::chrono::milliseconds(50);
    }

    if (auto msg = client.poll(0)) {
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std::printf("VehicleSpeed: %.3f m/s\n", vs->mps);
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  client.sendPcControl(0.0f, 0.0f);
  client.close();
  return 0;
}

