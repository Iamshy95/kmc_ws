# Tutorial: [Driver Basic] driver_demo (Driver 기반 송신)

이 튜토리얼에서는 `KMC_HARDWARE::Driver`를 사용하여 주행 명령(0xA5)을 주기적으로 송신하는 방법을 배웁니다.

## 목표
1. UART 포트를 열고
2. 내부 스레드가 0xA5/0xB3를 주기적으로 처리하게 만든 뒤
3. `setCommand()`로 고정 명령을 넣고 일정 시간 동안 송신을 유지합니다.

---

## 배경
안정적인 동작을 위해선 보통 제어(출력) 주”와 콜백을 분리하여 실행합니다.

- **주기 송신(0xA5)** 은 타이밍이 중요합니다. (executor 지터의 영향을 최소화)
- **주기 요청(0xB3/0xAF)** 은 필요 주기에 맞춰 보낼 수 있습니다.

`KMC_HARDWARE::Driver`는 UART I/O를 전용 스레드에서 처리하고, `setCommand()`로 목표값을 업데이트합니다.

---

## 준비물 (코드 작성 전에)
- 포트 문자열 예: `/dev/ttyKMC` (환경마다 다름)
- 보드가 사용하는 baud (예: `115200`, `921600`)
- RTS/CTS 설정 및 권한 설정은 `README_QUICKSTART.md` 참고

---

## 구현 가이드

### 헤더와 기본 선언
`Driver`는 SDK 내부에서 스레드를 돌리고 메시지 큐도 제공합니다.

```cpp
#include "KMC_driver.hpp"
```

이 시점에서 여러분이 얻게 되는 것:
- `KMC_HARDWARE::Driver`
- `KMC_HARDWARE::Driver::Options`

---

### 실행 인자와 기본값
예제는 포트와 실행 시간을 입력받도록 구성합니다.

```cpp
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  const std::string port = (argc > 1) ? argv[1] : "/dev/ttyKMC";
  int seconds = (argc > 2) ? std::atoi(argv[2]) : 5;
  if (seconds <= 0) seconds = 5;

  return 0;
}
```

- `/dev/ttyKMC`인지 `/dev/ttyUSB*`인지 환경에 따라 바꿉니다.
- 몇 초간 테스트할지 설정합니다.

---

### Driver 옵션 구성
`Driver`는 실행 중에 다음을 설정해야 합니다.
- Port
- baud/RTS/CTS 같은 시리얼 설정
- 제어(0xA5), 속도 요청(0xB3), 보조 요청(0xAF)을 몇 Hz로 돌릴지
- timeout 시간

이 설정을 전달하기 위해 `Options` 구조체가 있습니다.

```cpp
  KMC_HARDWARE::Driver drv;
  KMC_HARDWARE::Driver::Options opt;
  opt.port = port;
  
  // 시리얼 설정
  opt.serial.baudrate = 115200;           // 보드 설정에 맞추기
  opt.serial.hw_flow_control = true;      // RTS/CTS 사용
  opt.serial.rts_always_on = true;        // RTS assert 유지(기본값)

  // 루프 주기(Hz) 설정
  opt.control_rate_hz = 100.0;        // 0xA5 송신 주기 (100 Hz)
  opt.vehicle_speed_rate_hz = 1.0;    // 0xB3 요청 주기 (최소화)

  // 데모에서는 한번 setCommand()하는 명령을 구현하는 튜토리얼이므로, timeout을 끕니다.
  opt.command_timeout_ms = 0; 
```

- 더 빠른 제어주기는 부드러운 동작을 가능하게 하지만, 높은 송신 주기는 통신을 불안정하게 합니다. 
- B3는 피드백 요청이므로 필요한 주기에 맞춰 설정합니다. 여기서는 기본값으로 낮게 둡니다.

---

### Driver 시작

```cpp
  if (!drv.start(opt)) {
    std::fprintf(stderr, "Failed to open %s\n", port.c_str());
    return 2;
  }
```

`start()`는 포트 open 후 내부 I/O 스레드를 시작합니다.

---

### 명령 업데이트: `setCommand(v, omega)`
`Driver`는 명령을 **내부 스레드에서 주기적으로 송신**합니다.

```cpp
  drv.setCommand(0.5f, 0.0f);
```

ROS2에서의 적용:
- (예) cmd_vel 콜백에서 `setCommand()` 호출
- 또는 별도 제어 타이머(예: 200 Hz)에서 현재 목표값을 `setCommand()`로 업데이트

---

### 루프와 종료 처리

```cpp
  #include <thread>
  #include <chrono>

  // ... (main 내부)

  const auto t0 = std::chrono::steady_clock::now();
  auto next_status = t0;

  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(seconds)) {
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
```

`stop()`은 스레드 정리와 포트 close를 수행합니다.

---

전체 코드는 `examples/Driver_Basic/driver_demo.cpp` 참고.
```cpp
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
```
