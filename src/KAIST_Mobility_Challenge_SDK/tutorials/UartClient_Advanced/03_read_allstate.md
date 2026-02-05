# Tutorial: [UartClient Advanced] read_allstate (0xAF AllState 읽기)

이 튜토리얼은 `read_speed`에서 다룬 포트 open/flush, 루프 구조를 재사용합니다.
여기서는 0xAF AllState 요청과 응답 처리만 정리합니다.

## 목표
- 좌/우 모터에 대해 0xAF AllState 요청 전송
- `poll()`로 `AllState`를 받아 출력

---

## 구현 가이드

### 포트 open/flush
`read_speed`와 동일합니다.
```cpp
KMC_HARDWARE::UartClient client;
client.open(port, 115200);
client.flushInput();
```

### 두 모터 연속 요청

```cpp
client.requestAllState(0); // Left
client.requestAllState(1); // Right
```

### 수신 루프에서 AllState 출력

```cpp
const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
bool printed = false;
while (std::chrono::steady_clock::now() < deadline) {
  auto msg = client.poll(20);
  if (!msg) continue;
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
```

---

전체 코드는 `examples/UartClient_Advanced/read_allstate.cpp` 참고.
