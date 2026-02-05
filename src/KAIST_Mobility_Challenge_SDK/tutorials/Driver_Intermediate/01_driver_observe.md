# Tutorial: [Driver Intermediate] driver_observe (Driver 큐 활용)

이 예제는 `Driver`가 내부에서 **B3 요청과 배터리 폴링**을 수행하도록 두고, `tryPopMessage()`로 메시지를 꺼내 출력합니다.

## 목표
- `Driver`로 A5 스트림 유지
- 내부에서 B3 요청 + 배터리 폴링 활성화
- 큐에서 `VehicleSpeed`/`BatteryVoltage`를 받아 출력

--- 

## 구현 가이드

### 옵션 구성 (B3 + 배터리 폴링)
`Driver`에서 B3 요청과 배터리 폴링 주기를 설정합니다.

```cpp
  KMC_HARDWARE::Driver::Options opt;
  // ... 기본 포트 설정 ...

  opt.control_rate_hz = 100.0;
  
  // 속도 요청 주기 설정 (50Hz)
  opt.vehicle_speed_rate_hz = 50.0;
  
  // 배터리 폴링 활성화 (1Hz)
  opt.poll_battery = true;
  opt.battery_rate_hz = 1.0;
  
  // 안전 타임아웃 설정
  opt.command_timeout_ms = 200;
```

### 명령 업데이트 주기
실제 주행 시나리오처럼, 명령을 주기적으로(50Hz) 업데이트하는 코드를 추가합니다.

```cpp
  const auto cmd_period =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / 50.0));
  
  // ... 루프 내부 ...
  if (now >= next_cmd) {
    drv.setCommand(0.5f, 0.0f); // 계속 0.5m/s로 업데이트
    next_cmd += cmd_period;
  }
```

### 큐 처리
`Driver` 내부 스레드가 수신한 데이터는 큐에 쌓입니다. 메인 루프에서는 이를 꺼냅니다.

```cpp
    // 큐에 있는 모든 메시지 처리
    while (auto msg = drv.tryPopMessage()) {
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std::printf("VehicleSpeed: %.3f m/s\n", vs->mps);
      } else if (auto* bv = std::get_if<KMC_HARDWARE::BatteryVoltage>(&*msg)) {
        std::printf("BatteryVoltage: %.2f V\n", bv->volt);
      }
    }
```

---

전체 코드는 `examples/Driver_Intermediate/driver_observe.cpp` 참고.
```cpp

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

```
