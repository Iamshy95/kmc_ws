# Tutorial: [Driver Intermediate] high_rate_control (1 kHz 고속 제어)

이 예제는 Jetson 환경에서 1 kHz 제어 스트림(0xA5)을 유지하기 위한 옵션/스케줄링/출력 설정을 다룹니다.
최대 송수신 속도는 상위 알고리즘의 동작 여부, 다른 프로그램의 동작 등 하드웨어 상태에 따라 다릅니다. 이를 테스트하기 위한 튜토리얼입니다.

## 목표
- 0xA5(주행 명령) 송신을 1 kHz로 안정적으로 유지
- 0xB3(차량 속도) 요청을 가능한 범위까지 끌어올려 통신 여유/성능 확인
- 실제 시스템처럼 `setCommand(v, omega)`를 주기적으로 업데이트(ex. 50 Hz)

---

## 중요사항
- 1 kHz 제어 스트림은 충분한 baud가 필요합니다(예: 921600, 1000000).
- 0xB3 응답 속도는 보드/펌웨어 상태에 따라 달라질 수 있습니다.
- 상위 알고리즘은 보통 20~100 Hz로 목표를 업데이트하고, Driver가 1 kHz 스트림을 유지합니다.

### 준비물(권장)
- baud: `921600` or `1000000`
- Linux에서 지터를 더 줄이려면 `realtime_priority` 사용(권한 필요할 수 있음)

---

## 입력 파라미터

- `port`: 디바이스 경로 (예: `/dev/ttyKMC`)
- `seconds`: 실행 시간 (기본 5)
- `baud`: UART 보드레이트 (기본 1000000)
- `vehicle_speed_hz`: 0xB3 요청 주기 (기본 100)
- `command_hz`: `setCommand(v, omega)` 업데이트 주기 (기본 50)

---

## 구현 가이드

### 고속 통신 설정 (0xA5/0xB3)
```cpp
  opt.serial.baudrate = 921600; // or 1000000

  opt.control_rate_hz = 1000.0;

  opt.vehicle_speed_rate_hz = 1000.0; // sbc 통신 상황에 따라 1 kHz까지
```

### 명령 업데이트 주기
`command_hz`에 맞춰 `setCommand(v, omega)`를 주기적으로 업데이트합니다.
Driver는 내부에서 `(v, omega)`를 곡률로 변환해 0xA5를 전송합니다.

```cpp
  // 상위 목표값 업데이트: 50 Hz
  const double cmd_hz = 50.0;
  // ...
  if (now >= next_cmd) {
    drv.setCommand(1.0f, 0.0f);
    next_cmd += cmd_period;
  }
```

### `command_timeout_ms` 계산
권장:
- `command_timeout_ms` = `2 * (1000 / command_hz)`
- 최소값 50ms로 하한을 둠

```cpp
  if (cmd_hz > 0.0) {
    const double period_ms = 1000.0 / cmd_hz;
    opt.command_timeout_ms = static_cast<int>(std::max(50.0, period_ms * 2.0));
  }
```

### Linux Thread Hint (선택)
Jetson 등에서 지터를 더 줄이고 싶다면 사용합니다.

```cpp
  opt.realtime_priority = 70; // SCHED_FIFO (권한 필요할 수 있음)
  opt.cpu_affinity = -1;      // 특정 코어에 고정하려면 0,1,2... 지정
```

### 루프 구조와 출력

```cpp
if (cmd_hz > 0.0 && now >= next_cmd) {
  drv.setCommand(1.0f, 0.0f);
  ++cmd_updates;
  next_cmd += cmd_period;
  if (next_cmd <= now) next_cmd = now + cmd_period;
}

if (now >= next_status) {
  std::printf("[1kHz] t=%llds cmd_updates=%zu\n", ...);
  next_status += std::chrono::seconds(1);
}
```

- `cmd_updates`는 1초 동안의 명령 업데이트 횟수입니다.
- `next_cmd <= now` 체크로 지연 누적을 방지합니다.

---

전체 코드는 `examples/Driver_Intermediate/high_rate_control.cpp` 참고.
```cpp
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


```
