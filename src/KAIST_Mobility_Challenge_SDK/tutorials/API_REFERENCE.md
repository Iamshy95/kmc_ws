# KMC_HARDWARE UART SDK API Reference

Scope: Public C++ API in `inc`

Public headers:
- `inc/KMC_protocol.hpp`
- `inc/KMC_serial_port.hpp`
- `inc/KMC_uart_client.hpp`
- `inc/KMC_driver.hpp`

Namespaces:
- `KMC_HARDWARE`
- `KMC_HARDWARE::protocol`

Units:
- velocity: m/s
- angular rate: rad/s
- curvature: 1/m
- rate: Hz
- voltage: V
- current: A
- temperature: C
- angle: deg
- speed: rpm
- time: ms, us
- baudrate: bps

---

## Driver (Main Interface)

### Driver::Options

| Field | Type | Default | Unit | Description |
| --- | --- | --- | --- | --- |
| `port` | `std::string` | `""` | - | Device path (e.g., `/dev/ttyKMC`) |
| `serial` | `SerialPortOptions` | defaults | - | Serial options (see SerialPortOptions) |
| `control_rate_hz` | `double` | `100.0` | Hz | 0xA5 control stream rate |
| `vehicle_speed_rate_hz` | `double` | `50.0` | Hz | 0xB3 speed request rate |
| `poll_battery` | `bool` | `false` | - | Enable battery polling |
| `battery_rate_hz` | `double` | `1.0` | Hz | Battery polling rate |
| `poll_allstate` | `bool` | `false` | - | Enable all-state polling |
| `allstate_rate_hz` | `double` | `10.0` | Hz | All-state polling rate |
| `allstate_motor_left` | `uint8_t` | `0` | - | Left motor ID |
| `allstate_motor_right` | `uint8_t` | `1` | - | Right motor ID |
| `command_timeout_ms` | `int` | `300` | ms | If no `setCommand()` update within this time, send (0,0) |
| `stop_burst_count` | `int` | `3` | count | Repeat count of (0,0) on `stop()` |
| `realtime_priority` | `int` | `-1` | - | Linux SCHED_FIFO priority (1-99), -1 disables |
| `cpu_affinity` | `int` | `-1` | - | Linux CPU affinity index, -1 disables |
| `max_queue` | `size_t` | `1024` | count | Max receive queue size (drop oldest on overflow) |

### Driver API

- `bool start(const Options& opt)`
  - Open port, flush input, start I/O thread.
- `void stop()`
  - Stop I/O thread, send safety stop, close port.
- `bool isRunning() const`
  - I/O thread running state.
- `void setCommand(float velocity_mps, float omega_rps)`
  - Input: v (m/s), omega (rad/s).
  - Internal: `curvature = omega / v` (|v| <= 1e-3 -> curvature = 0).
  - 0xA5 uses `(v, curvature)`.
- `void setCommandCurvature(float velocity_mps, float curvature_1pm)`
  - Input: v (m/s), curvature (1/m).
- `std::optional<Message> tryPopMessage()`
  - Non-blocking queue pop.
- `bool waitPopMessage(Message& out, int timeout_ms)`
  - Blocking pop with timeout (ms).
- `bool requestBatteryOnce(uint8_t motor_id = 0)`
  - Send one 0xAF battery request.
- `bool requestAllStateOnce(uint8_t motor_id)`
  - Send one 0xAF all-state request.

### Driver I/O Loop Behavior (Implementation Detail)

- 0xA5 at `control_rate_hz`.
- 0xB3 at `vehicle_speed_rate_hz`.
- 0xAF auxiliary/diagnostic requests when `poll_battery` / `poll_allstate` enabled.
- Parsed frames are pushed to the queue.
- If `command_timeout_ms` expires, (0,0) is sent instead of the last command.

---

## Data Structures (Messages)

`Message` type:
```
std::variant<VehicleSpeed, BatteryVoltage, AllState, AfResponse>
```

### VehicleSpeed

| Field | Type | Unit | Description |
| --- | --- | --- | --- |
| `mps` | `float` | m/s | Vehicle speed |

### BatteryVoltage

| Field | Type | Unit | Description |
| --- | --- | --- | --- |
| `volt` | `float` | V | Battery voltage |

### AllState

| Field | Type | Unit | Description |
| --- | --- | --- | --- |
| `id` | `uint32_t` | - | Motor ID |
| `position_deg` | `float` | deg | Motor position |
| `speed_rpm` | `float` | rpm | Mechanical speed |
| `current_A` | `float` | A | Current |
| `temperature_C` | `float` | C | Temperature |
| `error_code` | `uint32_t` | - | Error bitmask |
| `reserved_0` | `float` | - | Reserved |
| `reserved_1` | `float` | - | Reserved |
| `reserved_2` | `float` | - | Reserved |

### AfResponse (Raw 0xAF)

| Field | Type | Unit | Description |
| --- | --- | --- | --- |
| `motor_id` | `uint8_t` | - | Response motor ID |
| `rw` | `uint8_t` | - | RW_READ / RW_WRITE |
| `ids` | `std::vector<uint8_t>` | - | ID list |
| `data_f32` | `std::vector<float>` | - | float32 LE decode |
| `data_u32` | `std::vector<uint32_t>` | - | Same bytes as u32 LE |

---

## UartClient (Low-Level)

Thread safety: internal mutex.

### Open/Close

- `bool open(const std::string& port_name, int baudrate = 115200)`
- `bool open(const std::string& port_name, const SerialPortOptions& opts)`
- `void close()`
- `bool isOpen() const`
- `void flushInput()`

### Main Interface (0xA5/0xB3)

- `bool sendPcControl(float velocity_mps, float curvature_1pm)`
  - Input: v (m/s), curvature (1/m).
  - Send 0xA5 frame. No response expected.
- `bool requestVehicleSpeed()`
  - Send 0xB3 request (1 byte). Response is asynchronous.

### Auxiliary/Diagnostic (0xAF)

- `bool requestAllState(uint8_t motor_id)`
- `bool requestBatteryVoltage(uint8_t motor_id = 0)`
- `bool writeSpeedERPM(uint8_t motor_id, float erpm)`
  - Input: electrical RPM.
- `bool writeServoPulseUs(uint8_t motor_id, float pulse_us)`
  - Input: pulse width (us).
- `bool systemReset(uint8_t motor_id)`

### Polling

- `std::optional<Message> poll(int timeout_ms)`
  - `timeout_ms = 0` is non-blocking.
  - Parse internal RX buffer; read from OS buffer if needed.

### Parsing Rules

- Sync on `0xAF` or `0xB3` header.
- `0xB3`: fixed 5 bytes, return `VehicleSpeed`.
- `0xAF`: parse to `AfResponse`, then map to `BatteryVoltage` or `AllState` on known patterns.

---

## SerialPort (Internal)

### SerialPortOptions

| Field | Type | Default | Unit | Description |
| --- | --- | --- | --- | --- |
| `baudrate` | `int` | `115200` | bps | UART baudrate |
| `hw_flow_control` | `bool` | `true` | - | RTS/CTS enabled |
| `rts_always_on` | `bool` | `true` | - | Keep RTS asserted |

### SerialPort API

- `bool open(const std::string& port_name, const SerialPortOptions& opts)`
- `bool open(const std::string& port_name, int baudrate)`
- `void close()`
- `bool isOpen() const`
- `void flushInput()`
- `int writeAll(const uint8_t* data, size_t len)`
  - Returns bytes written; -1 on error.
- `int readSome(uint8_t* data, size_t max_len, int timeout_ms)`
  - Returns bytes read; 0 on timeout; -1 on error.

Thread safety: no internal synchronization; guarded by `UartClient` mutex.

---

## Protocol (Frames and Constants)

### Frame IDs (Header Bytes)

| ID | Meaning |
| --- | --- |
| `0xA5` | PC control command frame (velocity + curvature) |
| `0xB3` | Vehicle speed request/response frame |
| `0xAF` | Auxiliary/diagnostic/utility read/write frame |

### Constants

Packet headers (1 byte):

| Name | Value | Description |
| --- | --- | --- |
| `HEADER_GENERAL` | `0xAF` | Auxiliary/diagnostic/utility read/write |
| `HEADER_PC_CONTROL` | `0xA5` | Control command |
| `HEADER_VEHICLE_SPEED` | `0xB3` | Speed request/response |

Read/Write flags:

| Name | Value | Description |
| --- | --- | --- |
| `RW_READ` | `0x00` | Read request |
| `RW_WRITE` | `0x01` | Write request and read response |

Supported 0xAF IDs:

| Name | Value | Description |
| --- | --- | --- |
| `ID_SYSTEM_RESET` | `0x00` | Device reset |
| `ID_SPEED` | `0x03` | Speed command (ERPM) |
| `ID_SERVO_PULSE` | `0x05` | Servo pulse width (us) |
| `ID_ALL_STATE` | `0x06` | All-state |
| `ID_BATTERY_VOLTAGE` | `0x07` | Battery voltage |

Size limits:

| Name | Value | Description |
| --- | --- | --- |
| `MAX_IDS` | `16` | Max IDs per 0xAF frame |
| `ALL_STATE_FIELD_COUNT` | `9` | All-state field count |

### Frame Formats (little-endian)

All float values are float32 little-endian.

`0xA5` control frame:

```
[0]  HEADER_PC_CONTROL (0xA5)
[1]  velocity_mps (float32 LE, m/s)
[5]  curvature_1pm (float32 LE, 1/m)
```

`0xB3` request/response:

- Request: single byte `HEADER_VEHICLE_SPEED`.
- Response: 5 bytes `header + float32 speed`.

```
[0]  HEADER_VEHICLE_SPEED (0xB3)
[1]  speed_mps (float32 LE, m/s)
```

`0xAF` (auxiliary/diagnostic read/write):

```
[0]  HEADER_GENERAL (0xAF)
[1]  motor_id
[2]  RW_READ or RW_WRITE
[3]  n_id
[4..] ids[0..n_id-1]
[4+n_id..] data_f32[0..n_id-1] (only when RW_WRITE)
```

Notes:

- Read requests: `RW_READ`, no data.
- Read responses: `RW_WRITE`, include data.

### Helpers

- `appendU8(out, v)`
- `appendU32LE(out, v)`
- `readU32LE(p)`
- `appendFloatLE(out, value)`
- `readFloatLE(p)`

---

## Notes

- All floats are float32 little-endian.
- Writes are best-effort; no ACKs.

---

## Minimal Usage

Driver (v, omega):

```cpp
KMC_HARDWARE::Driver drv;
KMC_HARDWARE::Driver::Options opt;
opt.port = "/dev/ttyKMC";
opt.control_rate_hz = 100.0;
drv.start(opt);
drv.setCommand(0.5f, 0.2f); // v=0.5 m/s, omega=0.2 rad/s
```

Driver (v, curvature):

```cpp
drv.setCommandCurvature(0.5f, 0.4f); // k=0.4 1/m
```

UartClient:

```cpp
KMC_HARDWARE::UartClient client;
client.open("/dev/ttyKMC", 115200);
client.sendPcControl(0.5f, 0.4f); // v, curvature
client.requestVehicleSpeed();
if (auto msg = client.poll(50)) {
  if (auto* vs = std::get_if<KMC_HARDWARE::VehicleSpeed>(&*msg)) {
    // use vs->mps
  }
}
```



