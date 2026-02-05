# Quick Guide 

---

## 1) 빌드
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 2) 포트 권한 (Linux/Jetson)
```bash
sudo usermod -aG dialout $USER
# 로그아웃/로그인 후 다시 실행
```

---

## 3) 가장 안전한 테스트
```bash
./build/driver_demo /dev/ttyKMC 5
```

정상이라면 `[running] t=...` 형태의 상태 출력이 1초마다 보입니다.

---

## 4) 1 kHz 성능 테스트
```bash
./build/high_rate_control /dev/ttyKMC 5 921600 1000 50
```

`cmd_updates`가 초당 `command_hz` 근처로 증가하면 정상입니다.

---

## 5) 자주 막히는 것

- **RTS/CTS 설정 확인** : RTS, CTS를 사용해야합니다. 
- `setCommand()` 업데이트가 없으면 속도 값이 0으로 떨어짐

---

더 자세한 내용은:
- SDK 전체 설명: `README_SDK.md`
- 예제별 튜토리얼: `README.md`
