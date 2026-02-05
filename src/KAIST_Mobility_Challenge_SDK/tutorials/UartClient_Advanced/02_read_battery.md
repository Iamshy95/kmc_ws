# Tutorial: [UartClient Advanced] read_battery (0xAF 배터리 전압 읽기)

이 튜토리얼은 `read_speed`에서 다룬 포트 open/flush, 루프 구조를 그대로 사용하고,
0xAF 배터리 요청/응답 처리만 집중합니다.

## 목표
- 0xAF 배터리 요청을 주기적으로 전송
- `poll()`로 `BatteryVoltage`를 받아 출력

---

## 구현 가이드

### 포트 open/flush
`read_speed`와 동일합니다.
```cpp
KMC_HARDWARE::UartClient client;
client.open(port, 115200);
client.flushInput();
```

### 배터리 요청 -> 응답 대기 -> 출력

```cpp
const auto cycle_start = std::chrono::steady_clock::now();
client.requestBatteryVoltage();

const auto deadline = cycle_start + std::chrono::milliseconds(200);
bool printed = false;
while (std::chrono::steady_clock::now() < deadline) {
  auto msg = client.poll(50);
  if (!msg) continue;
  if (auto* bv = std::get_if<KMC_HARDWARE::BatteryVoltage>(&*msg)) {
    std::printf("BatteryVoltage: %.2f V\n", bv->volt);
    printed = true;
    break;
  }
}
```

### 주기 유지 (1 Hz)

```cpp
const auto elapsed = std::chrono::steady_clock::now() - cycle_start;
const auto period = std::chrono::seconds(1);
if (elapsed < period) {
  std::this_thread::sleep_for(period - elapsed);
}
```

---

전체 코드는 `examples/UartClient_Advanced/read_battery.cpp` 참고.
