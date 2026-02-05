#include "KMC_uart_client.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <variant>

int main(int argc, char **argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";

  KMC_HARDWARE::UartClient client;
  if (!client.open(port, 115200)) {
    std::fprintf(stderr, "Failed to open port: %s\n", port.c_str());
    return 1;
  }
  client.flushInput();

  while (true) {
    const auto cycle_start = std::chrono::steady_clock::now();
    client.requestBatteryVoltage();

    const auto deadline = cycle_start + std::chrono::milliseconds(200);
    bool printed = false;
    while (std::chrono::steady_clock::now() < deadline) {
      auto msg = client.poll(50);
      if (!msg) {
        continue;
      }
      if (auto* bv = std::get_if<KMC_HARDWARE::BatteryVoltage>(&*msg)) {
        std::printf("BatteryVoltage: %.2f V\n", bv->volt);
        printed = true;
        break;
      }
    }
    if (!printed) {
      std::printf("BatteryVoltage: (no response)\n");
    }

    const auto elapsed = std::chrono::steady_clock::now() - cycle_start;
    const auto period = std::chrono::seconds(1);
    if (elapsed < period) {
      std::this_thread::sleep_for(period - elapsed);
    }
  }

  return 0;
}

