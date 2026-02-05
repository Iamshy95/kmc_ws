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
    client.requestAllState(0); // Left
    client.requestAllState(1); // Right

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    bool printed = false;
    while (std::chrono::steady_clock::now() < deadline) {
      auto msg = client.poll(20);
      if (!msg) {
        continue;
      }
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
        printed = true;
      }
    }
    if (!printed) {
      std::printf("AllState: (no response)\n");
    }
  }

  return 0;
}

