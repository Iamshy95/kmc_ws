# KAIST Mobility Challenge Hardware UART SDK 

이 SDK는 **PC <-> KAIST Mobility Challenge Hardware** 간 UART 프로토콜을 C++로 구현한 SDK입니다.

- **메인 제어/피드백:** `0xA5`(주행 입력), `0xB3`(차량 속도 읽기)
- **보조 도구:** `0xAF`(모터 1개 단위 상태/전압 조회 및 제한적 Read/Write)

---

## 1) 빌드

### Linux (Ubuntu)

필요: `cmake`

```bash
sudo apt update
sudo apt install -y build-essential cmake

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 2) 실행 예제

상세한 코드 작성 가이드와 단계별 가이드는 [tutorials/README.md](./tutorials/README.md)를 참고하세요.

### (Beginner) 메인 루프 데모: `driver_demo`

- 내부에서 `0xA5`를 주기적으로 송신합니다.
```bash
./build/driver_demo /dev/ttyKMC 5
```

- 첫 번째 인자: 포트 (예: `/dev/ttyKMC`, `/dev/ttyACM0`)
- 두 번째 인자: 실행 시간(default 5초)

정상 동작이면 실행 중에 아래처럼 상태 출력이 이어집니다.

```
[running] t=0s cmd=0.5 m/s
[running] t=1s cmd=0.5 m/s
[running] t=2s cmd=0.5 m/s
```

### (Beginner) 속도 읽기: `read_speed`

```bash
./build/read_speed /dev/ttyKMC 5 20
```

`UartClient`로 B3 요청을 보내고 `VehicleSpeed`를 출력합니다.

### (Beginner) 보조 도구: `read_allstate`, `read_battery`

```bash
./build/read_allstate /dev/ttyKMC
./build/read_battery  /dev/ttyKMC
```

두 예제 모두 **요청 후 응답을 `poll()`로 받아 출력**합니다.

### (Intermediate) 저수준 데모: `teleop_and_speed`

```bash
./build/teleop_and_speed /dev/ttyKMC 5
```

`UartClient`로 A5 송신 + B3 요청/출력을 함께 수행하는 데모입니다.

### (Intermediate) Driver 큐 읽기: `driver_observe`

```bash
./build/driver_observe /dev/ttyKMC 5
```

`Driver`의 메시지 큐에서 VehicleSpeed/BatteryVoltage를 출력합니다.

### (Advanced) 1 kHz 고속 제어 (Jetson 추천): `high_rate_control`

```bash
./build/high_rate_control /dev/ttyKMC 5 1000000 100 50
```

- 인자: `[port] [seconds=5] [baud=1000000] [vehicle_speed_hz=100] [command_hz=50]`
- 기본 1 kHz 제어/100 Hz 속도 요청, 1 Mbps UART, RTS/CTS 사용
- `Driver::Options`의 `realtime_priority`(Linux)로 스레드 우선순위를 올려 지터를 줄입니다.
- 필요하면 `cpu_affinity`를 원하는 코어 인덱스로 설정할 수 있습니다.
- Jetson의 통신 속도가 부적하다면, `./build/high_rate_control /dev/ttyKMC 5 115200 50 50` 처럼 baudrate를 낮춰 실행하세요.

---

## 3) 자주 막히는 것들

### 3.1 Linux 포트 권한

```bash
sudo usermod -aG dialout $USER
# 로그아웃/로그인 후 다시 실행
```

### 3.2 RTS/CTS (하드웨어 플로우컨트롤)

이 보드는 RTS/CTS 라인이 연결되어 있어야 정상 통신합니다.
특히 **보드가 송신하기 위해서는 호스트의 RTS가 ready 상태여야** 할 수 있습니다.

- USB-UART 어댑터가 **RTS/CTS를 지원**하는지 확인하세요.
- 케이블 결선이 맞는지 확인하세요(PC RTS <-> 보드 CTS, PC CTS <-> 보드 RTS).

> 이 SDK는 기본적으로 RTS/CTS(하드웨어 플로우컨트롤)를 사용하도록 설정되어 있으며, Linux에서도 RTS를 명시적으로 ready상태로 두도록 처리되어 있습니다.

---

## 4) 고속 & Jetson 최적화 필요시

- UART Baud: 1,000,000 bps 이상 사용 (`Driver::Options::serial.baudrate = 1000000`).
- 하드웨어 플로우컨트롤: `SerialPortOptions::hw_flow_control = true`, `rts_always_on = true`(기본값).
- 스레드 힌트: `realtime_priority`(예: 60~80), 필요 시 `cpu_affinity`로 I/O 스레드 코어 고정.
- `command_timeout_ms`를 50~100 ms 정도로 두어 입력이 끊기면 빠르게 정지.
- `CMAKE_BUILD_TYPE=Release`로 빌드(or `--config Release`).

---

## 5) ROS2 구현

ROS2 노드에서는 보통 `KMC_HARDWARE::Driver`를 사용합니다.

- SDK 내부 스레드가 A5/B3 루프를 유지
- 노드는 `waitPopMessage()` or `tryPopMessage()`로 메시지를 받아 publish
- ROS2 의존성은 SDK에 없습니다. 노드에서 SDK 라이브러리를 링크해 사용합니다.

ROS2 예제 패키지 빌드는 아래 명령으로 충분합니다.

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2
colcon build --symlink-install
source install/setup.bash
```

노드/토픽/파라미터는 `examples/Driver_ROS2/README.md` 참고.

예시:

```cpp
// rclcpp 노드
KMC_HARDWARE::Driver drv;
KMC_HARDWARE::Driver::Options opt;
opt.port = "/dev/ttyKMC";
opt.serial.baudrate = 1000000;
opt.control_rate_hz = 1000.0;
opt.vehicle_speed_rate_hz = 100.0;
opt.realtime_priority = 70; // Linux only
drv.start(opt);

auto pub_speed = node->create_publisher<std_msgs::msg::Float32>("vehicle_speed", 10);
using namespace std::chrono_literals;
auto timer = node->create_wall_timer(1ms, [&]{
  if (auto msg = drv.tryPopMessage()) {
    if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
      std_msgs::msg::Float32 out;
      out.data = vs->mps;
      pub_speed->publish(out);
    }
  }
});

// cmd_vel subscribe 콜백에서 drv.setCommand(linear, omega);
```
