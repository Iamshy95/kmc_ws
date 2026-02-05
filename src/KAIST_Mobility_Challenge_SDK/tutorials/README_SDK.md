# KAIST Mobility Challenge Hardware UART SDK 튜토리얼


---

## 0. SDK 설명

- UART로 **주행 명령(0xA5)** 을 보내고 **차량 속도(0xB3)** 를 받습니다.
- 보조로 **배터리 전압 / 모터 상태(0xAF)** 를 읽습니다.

---

## 1. 하드웨어 준비

### 1) RTS/CTS 플로우 컨트롤
**RTS/CTS 하드웨어 플로우컨트롤**이 연결되어야 안정적으로 동작합니다.

> 수신이 전혀 없다면 거의 대부분 RTS/CTS 설정, baudrate 설정문제입니다.

### 2) 포트 권한
```bash
sudo usermod -aG dialout $USER
# 로그아웃/로그인 후 다시 실행
```

---

## 2. 빌드

```bash
sudo apt update
sudo apt install -y build-essential cmake

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## 3. SDK 구조

### 1) Low-level: `UartClient`
- 프로토콜 프레임을 직접 송수신
- 원하는 타이밍에 명령/요청을 직접 보내고 파싱
- 예: `sendPcControl()`, `requestVehicleSpeed()`, `poll()`

### 2) High-level: `Driver`
- 내부 스레드가 일정 주기로 0xA5/0xB3 루프를 유지
- 메시지를 큐에 쌓고 `tryPopMessage()`로 가져오는 구조

---

## 4. 사용 방법

### 1) 기본 순서
1. `Driver::Options` 설정
2. `Driver::start()`
3. 주기적으로 `setCommand()` 업데이트
4. `tryPopMessage()` or `waitPopMessage()`로 수신 처리
5. 종료 시 `stop()`

### 2) 고속 운용 시 설정
- `control_rate_hz = 1000`
- `vehicle_speed_rate_hz`는 하드웨어 응답 한계에 맞춤 (100~1000)
- UART baud: `921600` or `1000000`
- `realtime_priority` 설정 
- 필요 시 `cpu_affinity`로 I/O 스레드 고정

---

## 5. `Driver` 최소 코드 

```cpp
#include "KMC_driver.hpp"

int main() {
  KMC_HARDWARE::Driver drv;
  KMC_HARDWARE::Driver::Options opt;
  opt.port = "/dev/ttyKMC";
  opt.serial.baudrate = 921600;
  opt.control_rate_hz = 1000.0;
  opt.vehicle_speed_rate_hz = 100.0;
  opt.realtime_priority = 70; // Linux only

  if (!drv.start(opt)) return 1;

  // 주기적으로 업데이트해야 안전
  drv.setCommand(1.0f, 0.0f);

  // 수신 처리
  while (true) {
    if (auto msg = drv.tryPopMessage()) {
      if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
        // vs->mps 사용
      }
    }
  }

  drv.stop();
  return 0;
}
```

---

## 6. ROS2 예제 패키지 사용

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2
colcon build --symlink-install
source install/setup.bash
```

노드 목록/파라미터/토픽은 `examples/Driver_ROS2/README.md` 참고.

---

## 7. 성능이 안 나올 때 체크리스트

- **RTS/CTS 설정 확인**
- **baud rate**가 보드 설정과 일치하는지
- `control_rate_hz` 과도하게 높여서 하드웨어가 처리 못하는지
- `command_timeout_ms` 너무 작게 잡지 않았는지
- `/dev/tty*` 권한 문제

---

## 8. 예제
예제 튜토리얼은 `README.md`를 참고

---

## 9. SDK API 레퍼런스

> 기준 파일:
> - `inc/KMC_protocol.hpp`
> - `inc/KMC_serial_port.hpp`
> - `inc/KMC_uart_client.hpp`
> - `inc/KMC_driver.hpp`
> - `src/KMC_serial_port.cpp`
> - `src/KMC_uart_client.cpp`
> - `src/KMC_driver.cpp`

---

### 9.1 namespace/용어

- `namespace KMC_HARDWARE`: SDK의 최상위 namespace
- `namespace KMC_HARDWARE::protocol`: 프로토콜 상수/바이트 변환 유틸
- “A5”: `0xA5` Header(PC->보드) 주행 제어 패킷
- “B3”: `0xB3` Header(PC->보드 요청, 보드->PC 응답) 차량 속도 패킷
- “AF”: `0xAF` Header(일반 읽기/쓰기 프레임) 보조 상태/전압 등

---

### 9.2 `inc/KMC_protocol.hpp` (프로토콜 상수/유틸)

#### Packet header 상수
- `protocol::HEADER_GENERAL = 0xAF`  
  - AF(General) 프레임의 시작 바이트입니다.
  - 이 SDK는 수신 버퍼에서 `0xAF`를 찾아 프레임을 재동기화합니다.

- `protocol::HEADER_PC_CONTROL = 0xA5`  
  - 주행 명령 프레임(A5) 시작 바이트입니다.
  - **응답 프레임은 없음**

- `protocol::HEADER_VEHICLE_SPEED = 0xB3`  
  - 차량 속도 프레임(B3) 시작 바이트입니다.
  - PC는 1바이트 `0xB3`를 보내 요청하고, 보드가 5바이트 응답을 돌려줍니다.

#### 프레임 포맷(바이트 레이아웃)
SDK가 실제로 송수신하는 프레임의 구조는 아래처럼 생각하면 됩니다. (float32는 little-endian)

**A5 (PC->Board, control)**
```
[0]   0xA5
[1:4] velocity_mps   (float32 LE)
[5:8] curvature_1pm  (float32 LE)  // 보통 [1/m] 의미로 사용
총 9바이트
```

**B3 (PC->Board request, Board->PC response)**
```
요청:  [0]   0xB3            (총 1바이트)
응답:  [0]   0xB3
       [1:4] speed_mps (float32 LE) (총 5바이트)
```

**AF (General)**
```
PC->Board (read 요청):
[0] 0xAF
[1] motor_id
[2] 0x00 (RW_READ)
[3] n_id
[4..] ids (n_id 바이트)

Board->PC (read 응답은 write 형태로 데이터 포함):
[0] 0xAF
[1] motor_id
[2] 0x01 (RW_WRITE)
[3] n_id
[4..] ids (n_id 바이트)
[... ] data (n_id * 4 바이트, float32 LE로 해석 가능)
```

#### RW 상수
- `protocol::RW_READ = 0x00`  
  - AF 프레임에서 읽기 요청에 사용합니다(PC->보드).

- `protocol::RW_WRITE = 0x01`  
  - AF 프레임에서 쓰기 or 읽기 응답에 사용됩니다(보드->PC).
  - 이 프로토콜에서는 **read 요청에 대한 응답도 `RW_WRITE` 형태로 데이터가 포함**되는 방식을 사용합니다.

#### 지원하는 AF ID 상수(제외 기능 반영)
이 SDK는 파라미터 튜닝/CAN/current-tuning 관련 ID를 의도적으로 포함하지 않습니다.

- `protocol::ID_SYSTEM_RESET = 0x00`  
  - 시스템 리셋 명령 ID
- `protocol::ID_SPEED = 0x03`  
  - 속도 관련 ID(저수준에서 쓰기 가능: `writeSpeedERPM`)
- `protocol::ID_SERVO_PULSE = 0x05`  
  - 서보 펄스 관련 ID(저수준에서 쓰기 가능: `writeServoPulseUs`)
- `protocol::ID_ALL_STATE = 0x06`  
  - AllState(상태 묶음) 읽기 ID
- `protocol::ID_BATTERY_VOLTAGE = 0x07`  
  - 배터리 전압 읽기 ID

#### 기타 상수
- `protocol::MAX_IDS = 16`  
  - AF 프레임 한 번에 실을 수 있는 ID 개수 상한

- `protocol::ALL_STATE_FIELD_COUNT = 9`  
  - AllState 응답이 9개 필드로 올 때의 개수
  - SDK는 이 패턴을 감지하면 `KMC_HARDWARE::AllState`로 디코딩합니다.

#### 바이트/엔디언 유틸(LE)
- `appendU8(out, v)`  
  - 벡터 끝에 1바이트 추가

- `appendU32LE(out, v)`  
  - `uint32_t`를 little-endian 4바이트로 추가

- `readU32LE(p)`  
  - `p[0..3]`에서 little-endian `uint32_t`를 읽음

- `appendFloatLE(out, value)`  
  - `float`를 IEEE754 32-bit로 간주하고 little-endian 4바이트로 추가
  - `static_assert(sizeof(float)==4)`로 float32 환경을 전제

- `readFloatLE(p)`  
  - little-endian float32를 읽어 `float`로 반환

---

### 9.3 `inc/KMC_serial_port.hpp` (OS별 시리얼 I/O 추상화)

이 레이어는 바이트 스트림 read/write만 담당합니다.  
프로토콜(프레이밍/파싱)은 `UartClient`가 담당합니다.

#### `struct KMC_HARDWARE::SerialPortOptions`

- `int baudrate = 115200`  
  - 포트 속도(bps)
  - Jetson 고속 운용 시 `921600` or `1000000` 사용을 권장합니다.

- `bool hw_flow_control = true`  
  - RTS/CTS 하드웨어 플로우컨트롤 사용 여부
  - `true`: CTS가 없으면 TX가 막힐 수 있음
  - `false`: CTS 무시

- `bool rts_always_on = true`  
  - 호스트 RTS 라인을 ready로 유지할지
  - 보드가 호스트 RTS를 요구하는 경우가 있어 기본값은 `true`

#### `class KMC_HARDWARE::SerialPort`

public 요약 :
```cpp
class SerialPort {
public:
  SerialPort();
  ~SerialPort();

  bool open(const std::string& port_name, const SerialPortOptions& opts);
  bool open(const std::string& port_name, int baudrate);

  void close();
  bool isOpen() const;

  void flushInput();
  int writeAll(const uint8_t* data, size_t len);
  int readSome(uint8_t* data, size_t max_len, int timeout_ms);
};
```

public method:
- `SerialPort()` / `~SerialPort()`  
  - 생성/파괴. 소멸자는 `close()`를 호출합니다.

- `bool open(const std::string& port_name, const SerialPortOptions& opts)`  
  - 포트를 열고(기존 열려있으면 닫고) 옵션을 적용합니다.
  - Linux에서는 `termios`를 raw 모드로 설정하고, baud/RTS/CTS를 설정합니다.
  - Windows에서는 DCB/COMMTIMEOUTS를 설정합니다.

- `bool open(const std::string& port_name, int baudrate)`  
  - 하위호환 편의 함수. 내부에서 `SerialPortOptions`를 만들고 위 `open()`을 호출합니다.

- `void close()`  
  - 열린 포트를 닫습니다.

- `bool isOpen() const`  
  - 핸들/FD가 유효한지로 열림 여부 판단

- `void flushInput()`  
  - 수신 버퍼의 잔여 바이트를 best-effort로 폐기합니다.
  - 프로토콜 동기화가 꼬였을 때 초기화 용도로 자주 씁니다.

- `int writeAll(const uint8_t* data, size_t len)`  
  - `len` 바이트를 **모두 쓸 때까지** 반복 write 합니다(블로킹/베스트 에포트).
  - 성공 시 `len` 반환, 실패 시 `-1` 반환
  - Linux는 `EAGAIN/EWOULDBLOCK`을 내부에서 재시도합니다.

- `int readSome(uint8_t* data, size_t max_len, int timeout_ms)`  
  - 최대 `max_len`까지 읽습니다.
  - 반환:
    - `>0`: 읽은 바이트 수
    - `0`: 타임아웃(혹은 현재 읽을 게 없음)
    - `-1`: 오류

---

### 9.4 `inc/KMC_uart_client.hpp` (low level protocol client)

`UartClient`는 byte stream 위에 프로토콜을 올리고,
`Message(std::variant)`로 파싱 결과를 돌려줍니다.

#### 메시지 타입들

- `struct KMC_HARDWARE::VehicleSpeed`
  - `float mps`: 차량 속도 [m/s]

- `struct KMC_HARDWARE::BatteryVoltage`
  - `float volt`: 배터리 전압 [V]

- `struct KMC_HARDWARE::AllState`
  - AllState(상태 묶음) 결과입니다.
  - 필드:
    - `uint32_t id`: 모터/디바이스 식별값(프로토콜 정의에 따름)
    - `float position_deg`: 위치 [deg]
    - `float speed_rpm`: 기계적 회전수 [rpm]
    - `float current_A`: 전류 [A]
    - `float temperature_C`: 온도 [°C]
    - `uint32_t error_code`: 에러 코드
    - `float reserved_0/reserved_1/reserved_2`: 프로토콜 확장/예약 필드

- `struct KMC_HARDWARE::AfResponse`
  - 해석되지 않은 원본 AF 응답을 담는 타입입니다.
  - 필드:
    - `uint8_t motor_id`: 응답에 포함된 motor id
    - `uint8_t rw`: `RW_READ`/`RW_WRITE`
    - `std::vector<uint8_t> ids`: 포함된 ID 목록
    - `std::vector<float> data_f32`: 4바이트 단위 데이터를 float32로 해석한 값
    - `std::vector<uint32_t> data_u32`: 같은 4바이트를 u32로도 같이 해석한 값

- `using KMC_HARDWARE::Message = std::variant<VehicleSpeed, BatteryVoltage, AllState, AfResponse>;`
  - `poll()`은 이 타입을 반환합니다.
  - `std::visit` or `std::get_if`로 분기 처리합니다.

#### `class KMC_HARDWARE::UartClient` (Public API)

public method:
```cpp
class UartClient {
public:
  bool open(const std::string& port_name, int baudrate = 115200);
  bool open(const std::string& port_name, const SerialPortOptions& opts);
  void close();
  bool isOpen() const;
  void flushInput();

  bool sendPcControl(float velocity_mps, float curvature_1pm);
  bool requestVehicleSpeed();

  bool requestAllState(uint8_t motor_id);
  bool requestBatteryVoltage(uint8_t motor_id = 0);

  bool writeSpeedERPM(uint8_t motor_id, float erpm);
  bool writeServoPulseUs(uint8_t motor_id, float pulse_us);
  bool systemReset(uint8_t motor_id);

  std::optional<Message> poll(int timeout_ms);
};
```

스레드 안전성:
- 모든 Public method는 내부 `mtx_`로 serialize됩니다. 호출은 thread-safe

포트 제어:
- `bool open(const std::string& port_name, int baudrate = 115200)`  
  - 포트를 열고 수신 버퍼(`rx_buffer_`)를 초기화합니다.

- `bool open(const std::string& port_name, const SerialPortOptions& opts)`  
  - `SerialPortOptions`로 포트를 엽니다.

- `void close()`  
  - 포트 닫고, 내부 수신 버퍼를 비웁니다.

- `bool isOpen() const`  
  - 포트가 열렸는지 확인합니다.

- `void flushInput()`  
  - OS 수신 버퍼 + 내부 `rx_buffer_`를 비웁니다.

주요 송신:
- `bool sendPcControl(float velocity_mps, float curvature_1pm)`  
  - A5 프레임을 송신합니다.
  - 페이로드: float32 2개(velocity, curvature)

- `bool requestVehicleSpeed()`  
  - 1바이트 `0xB3`를 송신해 차량 속도를 요청합니다.
  - 응답은 `poll()`로 읽어 `VehicleSpeed`로 받습니다.

보조(AF) 요청:
- `bool requestAllState(uint8_t motor_id)`  
  - AF read 요청으로 `ID_ALL_STATE`를 요청합니다.
  - 응답 패턴이 AllState(9필드)면 `AllState`로 디코딩됩니다.

- `bool requestBatteryVoltage(uint8_t motor_id = 0)`  
  - AF read 요청으로 `ID_BATTERY_VOLTAGE`를 요청합니다.
  - 응답 패턴이 배터리 전압 1개면 `BatteryVoltage`로 디코딩됩니다.

저수준 쓰기(필요 시만):
- `bool writeSpeedERPM(uint8_t motor_id, float erpm)`  
  - AF write로 속도 관련 값을 씁니다(ACK 없음).
  - 반영되었는지 확인하려면 read-back을 별도로 해야 합니다.

- `bool writeServoPulseUs(uint8_t motor_id, float pulse_us)`  
  - AF write로 서보 펄스(us)를 씁니다(ACK 없음).
  - 디바이스 측 구현상 첫 ID가 ServoPulse일 때만 처리되는 제약이 있어 SDK가 그 형태로 전송합니다.

- `bool systemReset(uint8_t motor_id)`  
  - AF write로 reset을 요청합니다.

수신/파싱:
- `std::optional<Message> poll(int timeout_ms)`  
  - 내부 버퍼에 이미 완전한 프레임이 있으면 즉시 파싱해 반환합니다.
  - 없으면 `SerialPort::readSome(..., timeout_ms)`로 바이트를 읽고,
    다시 파싱을 시도합니다.
  - 반환:
    - `Message`: 프레임 1개 파싱 성공
    - `std::nullopt`: 타임아웃/부분 프레임/에러(에러도 nullop)

#### `UartClient` 내부 함수/멤버

- `bool sendGeneralRead(uint8_t motor_id, const std::vector<uint8_t>& ids)`  
  - AF read 프레임을 구성해 전송
  - 형식: `[0xAF][motor_id][0x00][n_id][ids...]`

- `bool sendGeneralWrite(uint8_t motor_id, const std::vector<uint8_t>& ids, const std::vector<float>& data_f32)`  
  - AF write 프레임을 구성해 전송
  - 형식: `[0xAF][motor_id][0x01][n_id][ids...][data...float32 LE]`

- `bool tryParseOne(Message& out)`  
  - `rx_buffer_` 안에서 프레임 1개를 찾고(Header 기반 재동기화),
    완전하면 `out`에 채우고 소비합니다.
  - 구현 핵심:
    - `0xAF`와 `0xB3`를 동시에 탐색해 가장 앞에 있는 Header 기준으로 파싱
    - B3는 5바이트 고정(Header 1 + float32 4)
    - AF는 길이가 가변이며 `rw/n_id`에 따라 총 길이를 계산
    - 손상 프레임이면 1바이트씩 버리며 재동기화

멤버 변수:
- `mutable std::mutex mtx_`  
  - thread-safe serialize 용도
- `SerialPort port_`  
  - OS 시리얼 포트 핸들
- `std::vector<uint8_t> rx_buffer_`  
  - readSome으로 읽은 바이트를 누적하는 버퍼(프레임 경계가 맞지 않아도 유지)

---

### 9.5 `inc/KMC_driver.hpp` 

`Driver`는 UART I/O thread를 내부에서 돌립니다.

#### `struct KMC_HARDWARE::Driver::Options`

- `std::string port`  
  - 열 포트 경로(예: `/dev/ttyKMC`)

- `SerialPortOptions serial`  
  - baud/RTS/CTS 등 시리얼 설정

- `double control_rate_hz = 100.0`  
  - A5(주행 명령) 송신 주기[Hz]
  - 1 kHz 운용 시 `1000.0`

- `double vehicle_speed_rate_hz = 50.0`  
  - B3 요청 주기[Hz]
  - 보드가 1 kHz 응답이 가능하면 `1000.0`도 가능

- `bool poll_battery = false` / `double battery_rate_hz = 1.0`  
  - 배터리 전압(AF) 주기 요청 on/off 및 주기[Hz]

- `bool poll_allstate = false` / `double allstate_rate_hz = 10.0`  
  - AllState(AF) 주기 요청 on/off 및 주기[Hz]

- `uint8_t allstate_motor_left = 0`, `uint8_t allstate_motor_right = 1`  
  - poll_allstate가 켜졌을 때 요청할 모터 id 두 개(좌/우)

- `int command_timeout_ms = 300`  
  - `setCommand()`가 이 시간(ms) 동안 업데이트되지 않으면,
    Driver가 자동으로 `(0,0)`을 송신합니다(안전 목적).
  - 데모처럼 한 번 명령을 주고 유지하려면 0으로 두면 됩니다.

- `int stop_burst_count = 3`  
  - `stop()` 시 `(0,0)`을 몇 번 연속으로 보낼지(best-effort)

- `int realtime_priority = -1` (Linux only, best-effort)  
  - I/O 스레드를 `SCHED_FIFO`로 올릴 때 사용할 priority(1~99)
  - 권한이 없으면 적용이 실패할 수 있으며, 실패해도 프로그램은 계속 동작합니다.

- `int cpu_affinity = -1` (Linux only, best-effort)  
  - I/O 스레드를 특정 CPU 코어로 고정할 때 사용할 CPU index

- `size_t max_queue = 1024`  
  - Driver 내부 메시지 큐 크기 상한
  - 초과 시 가장 오래된 메시지부터 drop 합니다.

#### `class KMC_HARDWARE::Driver` (public API)

public method:
```cpp
class Driver {
public:
  struct Options { /* ... */ };

  Driver();
  ~Driver();

  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;

  bool start(const Options& opt);
  void stop();
  bool isRunning() const;

  void setCommand(float velocity_mps, float omega_rps);
  void setCommandCurvature(float velocity_mps, float curvature_1pm);
  std::optional<Message> tryPopMessage();
  bool waitPopMessage(Message& out, int timeout_ms);

  bool requestBatteryOnce(uint8_t motor_id = 0);
  bool requestAllStateOnce(uint8_t motor_id);
};
```

- `Driver()` / `~Driver()`  
  - 생성 시 마지막 명령 시간(`cmd_last_update_`)을 현재로 초기화합니다.
  - 소멸자는 `stop()`을 호출합니다(스레드/포트 정리).

- `bool start(const Options& opt)`  
  - 기존에 실행 중이면 `stop()` 후 다시 시작합니다.
  - 주요 동작:
    - 옵션 복사/정규화(Hz clamp)
    - 포트 오픈(`UartClient::open`)
    - 입력 flush
    - 내부 상태 초기화
    - `running_=true`, I/O 스레드 시작
  - 실패 시 `false` 반환(포트 오픈 실패 등)

- `void stop()`  
  - `running_=false`로 루프 종료 후 스레드 join
  - 베스트 에포트로 `(0,0)`을 여러 번 송신(stop burst)
  - 포트 close
  - 내부 큐 비움

- `bool isRunning() const`  
  - 현재 루프가 동작 중인지 확인합니다.

- `void setCommand(float velocity_mps, float omega_rps)`  
  - 스레드 안전하게 현재 목표 명령을 업데이트합니다.
  - 내부에서 `(v, omega)`를 곡률로 변환해 A5에 사용합니다.
- `void setCommandCurvature(float velocity_mps, float curvature_1pm)`  
  - 곡률 기반 명령을 직접 넣는 헬퍼입니다.

- `std::optional<Message> tryPopMessage()`  
  - 큐에서 즉시 하나 pop (없으면 nullopt)

- `bool waitPopMessage(Message& out, int timeout_ms)`  
  - 최대 `timeout_ms` 동안 큐를 기다렸다가, 있으면 pop해서 `out`에 반환
  - 반환값:
    - `true`: 메시지 pop 성공
    - `false`: 타임아웃 or stop으로 인해 깨움

- `bool requestBatteryOnce(uint8_t motor_id = 0)`  
  - 즉시 배터리 요청(AF)을 한 번 전송합니다.
  - 응답은 나중에 큐로 들어옵니다(`BatteryVoltage` or `AfResponse`).

- `bool requestAllStateOnce(uint8_t motor_id)`  
  - 즉시 AllState 요청(AF)을 한 번 전송합니다.
  - 응답은 나중에 큐로 들어옵니다(`AllState` or `AfResponse`).

#### `Driver` 내부 함수/멤버

내부 함수:
- `void pushMessage(const Message& m)`  
  - 큐에 메시지를 넣고, 조건변수를 notify 합니다.
  - 큐가 꽉 차면 앞에서 drop 합니다.

- `void ioLoop()`  
  - I/O 스레드의 메인 루프
  - A5 송신 / B3 요청 / AF 폴링 / 수신 파싱을 한 스레드에서 수행합니다.

- `static double clampRate(double hz, double fallback)`  
  - 비정상 값(NaN/<=0)인 경우 fallback 사용

- `static std::chrono::nanoseconds periodFromHz(double hz, double fallback_hz)`  
  - Hz -> period(ns) 변환
  - 과도한 입력에 대해 period가 0ns가 되지 않게 lower bound를 둡니다.

- `void applyThreadHints()` (Linux)  
  - `cpu_affinity`, `realtime_priority`를 best-effort로 적용합니다.

멤버 변수(역할):
- `Options opt_`  
  - 현재 적용된 옵션(스레드가 참조)
- `UartClient client_`  
  - 실제 UART 송수신 담당
- `std::atomic<bool> running_`  
  - 루프 실행 플래그

명령 상태:
- `std::mutex cmd_mtx_`  
  - 아래 명령 변수 보호
- `float cmd_v_`, `float cmd_c_`  
  - 최신 목표값(속도/곡률). `setCommand()` 입력은 내부에서 곡률로 변환됩니다.
- `std::chrono::steady_clock::time_point cmd_last_update_`  
  - 마지막 `setCommand()` 시간(타임아웃 판단에 사용)

큐:
- `std::mutex q_mtx_` / `std::condition_variable q_cv_`  
  - 큐 동기화/대기
- `std::deque<Message> q_`  
  - 수신 메시지 저장소

스레드:
- `std::thread io_thread_`  
  - I/O 루프 스레드 핸들

---

### 9.6 `src/*.cpp` 구현

#### 9.6.1 `src/KMC_driver.cpp`

함수별 동작:
- `Driver::start()`  
  - Hz 값들을 `clampRate()`로 정규화합니다.
  - `opt_.serial.baudrate <= 0`이면 기본값을 `1000000`으로 둡니다 (상황에 따라 baudrate는 조절).
  - 포트 open에 성공하면 내부 스레드를 시작합니다.

- `Driver::stop()`  
  - `running_`을 내리고 join한 뒤, 베스트 에포트로 (0,0)을 여러 번 송신합니다.

- `Driver::setCommand()`  
  - `(v, omega)`를 곡률로 변환한 뒤 mutex로 보호하여 저장합니다.

- `Driver::tryPopMessage()` / `Driver::waitPopMessage()`  
  - 큐에서 메시지를 꺼내는 API입니다.

- `Driver::periodFromHz()`  
  - ns 변환 시 너무 큰 Hz로 0ns가 되지 않도록 최소 1000ns 하한을 둡니다.

- `Driver::applyThreadHints()`  
  - Linux에서:
    - `pthread_setaffinity_np`로 CPU 고정 시도
    - `pthread_setschedparam(SCHED_FIFO)`로 우선순위 설정 시도
  - 권한/환경에 따라 실패할 수 있고, 실패해도 계속 동작합니다.

- `Driver::ioLoop()` (핵심 루프)
  - 다음 실행 시각(next_*)을 각각 유지합니다.
  - 루프에서:
    1) A5: `control_rate_hz` 주기로 송신  
       - `command_timeout_ms` 초과 시 자동 0,0으로 대체  
       - 지터/지연으로 너무 많이 밀렸을 때도 폭주하지 않도록 catch-up 2회 제한을 둡니다.
    2) B3: `vehicle_speed_rate_hz` 주기로 요청(1바이트 전송)
    3) AF: 옵션이 켜졌을 때만 battery/allstate 요청
    4) 수신: `client_.poll(0)`으로 버퍼에 쌓인 프레임을 여러 개(최대 16개) 빠르게 처리
    5) 다음 이벤트까지의 시간을 계산해 짧게 sleep  
       - 200us 이상 남으면 약간 덜 자서(100us 마진) 깨어나 지터를 줄임

#### 9.6.2 `src/KMC_uart_client.cpp`

- 송신은 모두 바이트 배열 생성 -> `SerialPort::writeAll()` 형태입니다.
- 수신은 `rx_buffer_`에 누적하고, `tryParseOne()`이 1프레임씩 사용합니다.

`tryParseOne()`의 프레이밍 규칙:
- 수신 버퍼에서 `0xAF`/`0xB3` 중 가장 앞에 있는 Header를 찾습니다.
- Header 앞의 쓸모없는 바이트는 버립니다.
- B3:
  - 길이 5바이트 고정(Header 1 + float32 4)
  - float32를 LE로 해석해 `VehicleSpeed` 반환
- AF:
  - `[0xAF][motor_id][rw][n_id]` 최소 4바이트를 확인
  - `n_id`와 `rw`에 따라 총 길이를 계산
  - `rw==RW_WRITE`인 경우에만 데이터(4바이트 * n_id)가 따라오므로 그때만 디코딩합니다.
  - 디코딩 결과가 배터리 1개이면 `BatteryVoltage`로,
    AllState 9필드이면 `AllState`로 변환합니다.
  - 그 외는 `AfResponse`로 그대로 돌려줍니다.

#### 9.6.3 `src/KMC_serial_port.cpp`

Windows:
- `CreateFileA`로 COM 포트 open
- Windows에서 `COM10` 이상 포트는 `"\\\\.\\COM10"` 형태가 필요해서, 내부에서 포트 이름을 보정합니다.
- DCB로 baud/RTS/CTS 설정
- `COMMTIMEOUTS`를 요청마다 적용(`readSome`에서 timeout_ms 반영)

Linux:
- `open(..., O_NONBLOCK)` 후 `termios`를 raw로 설정
- baud는 `B115200`, `B921600`, `B1000000` 등 지원되는 상수에 매핑
- `select()`로 timeout 대기 후 `read()` 수행
- `rts_always_on`이면 ioctl로 RTS를 assert

---

### 9.7  요약

- **주기/성능**  
  - `Driver::Options::control_rate_hz`, `vehicle_speed_rate_hz`, `serial.baudrate`
  - Linux에서 `realtime_priority`, `cpu_affinity`

- **safe 동작**  
  - `command_timeout_ms`, `stop_burst_count`

- **보조 상태(배터리/AllState)**  
  - `poll_battery`, `battery_rate_hz`, `poll_allstate`, `allstate_rate_hz`

- **저수준 디버깅/실험**  
  - `UartClient`를 직접 사용(`poll()`, `request...()`, `sendPcControl()`)



