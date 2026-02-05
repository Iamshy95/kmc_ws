# Tutorial: [UartClient Advanced] read_speed (B3 요청 및 출력)

이 튜토리얼에서는 low-level class `UartClient`를 사용하여 차량의 현재 속도(0xB3)를 요청하고 응답을 출력하는 방법을 다룹니다.
요청 전송 -> `poll()` 수신 -> 출력을 구현합니다.

## 목표
1. B3 요청을 전송
2. `poll()`로 `VehicleSpeed`를 받아 출력
3. 지정한 주기로 반복

---

## 구현 가이드

### 헤더와 기본 구조
```cpp
#include "KMC_uart_client.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <variant>

int main(int argc, char **argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";
  // ... (인자 파싱)
```

### 포트 열기 + flush
`UartClient`는 lowlevel 클래스이므로 포트를 직접 열고 초기화해야 합니다.

```cpp
  KMC_HARDWARE::UartClient client;
  if (!client.open(port, 115200)) {
    std::fprintf(stderr, "Failed to open port: %s\n", port.c_str());
    return 1;
  }
  client.flushInput();
```

### 요청/수신 루프
이 예제에서는 정해진 주기(`period`)마다 요청을 보냅니다.

```cpp
  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(seconds)) {
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
```

---

전체 코드는 `examples/UartClient_Advanced/read_speed.cpp` 참고.

