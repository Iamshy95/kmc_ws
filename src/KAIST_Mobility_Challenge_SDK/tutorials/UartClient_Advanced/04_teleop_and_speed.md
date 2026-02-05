# Tutorial: [UartClient Advanced] teleop_and_speed (직접 스케줄링)

이 튜토리얼은 `UartClient`로 직접 스케줄링하여 **A5 송신 + B3 요청/출력**을 구현합니다.

## 목표
- `UartClient`로 A5 송신과 B3 요청/출력을 구현
- 스케줄링/타이밍을 직접 관리하는 루프 구성

---

## 구현 가이드

### 포트 열기
`UartClient`는 저수준이기 때문에 먼저 포트를 열고, 초기 입력을 비웁니다.

```cpp
  KMC_HARDWARE::UartClient client;
  if (!client.open(port, 115200)) {
    return 1;
  }
  client.flushInput();
```

### 스케줄링 타이머 설정
여기서 다음에 보낼 시간을 잡고, 루프에서 시간이 되면 동작합니다.

```cpp
  const auto t0 = std::chrono::steady_clock::now();
  auto next_ctrl = t0;
  auto next_b3 = t0;
```

### 메인 루프 구현 (직접 주기 관리)

```cpp
  while (...) {
    const auto now = std::chrono::steady_clock::now();

    if (now >= next_ctrl) {
      client.sendPcControl(v, k);
      next_ctrl += std::chrono::milliseconds(10); // 10ms = 100Hz
    }

    if (now >= next_b3) {
      client.requestVehicleSpeed();
      next_b3 += std::chrono::milliseconds(50); // 50ms = 20Hz
    }

    if (auto msg = client.poll(0)) {
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        std::printf("VehicleSpeed: %.3f m/s\n", vs->mps);
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
```

---

전체 코드는 `examples/UartClient_Advanced/teleop_and_speed.cpp` 참고.
