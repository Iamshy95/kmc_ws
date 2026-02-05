#pragma once
#include "KMC_protocol.hpp"
#include "KMC_serial_port.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace KMC_HARDWARE {

// Raw 0xAF response payload.
struct AfResponse {
  uint8_t motor_id = 0;           // Motor ID in the response.
  uint8_t rw = 0;                 // RW_READ or RW_WRITE.
  std::vector<uint8_t> ids;       // Requested ID list.
  std::vector<float> data_f32;    // float32 LE.
  std::vector<uint32_t> data_u32; // same bytes as u32 LE.
};

// Parsed 0xB3 vehicle speed message.
struct VehicleSpeed {
  float mps = 0.0f; // m/s
};

// Parsed battery voltage message.
struct BatteryVoltage {
  float volt = 0.0f; // V
};

// Decoded all state payload (0xAF id 0x06).
struct AllState {
  uint32_t id = 0;            // motor ID
  float position_deg = 0.0f;  // deg
  float speed_rpm = 0.0f;     // mechanical RPM
  float current_A = 0.0f;     // A
  float temperature_C = 0.0f; // C
  uint32_t error_code = 0;    // bitmask
  float reserved_0 = 0.0f;
  float reserved_1 = 0.0f;
  float reserved_2 = 0.0f;
};

// Message variant returned by poll().
using Message =
    std::variant<VehicleSpeed, BatteryVoltage, AllState, AfResponse>;

// Low-level UART client
class UartClient {
public:
  bool open(const std::string &port_name, int baudrate = 115200);
  bool open(const std::string &port_name, const SerialPortOptions &opts);
  void close();
  bool isOpen() const;

  // Flush unread bytes from the OS buffer and internal buffer.
  void flushInput();

  // --- Main interface ---
  // Send PC control (0xA5): velocity and curvature
  bool sendPcControl(float velocity_mps, float curvature_1pm);
  // Request vehicle speed (0xB3)
  bool requestVehicleSpeed();

  // --- Auxiliary interface (0xAF) ---
  // Request all state for the given motor ID.
  bool requestAllState(uint8_t motor_id);
  // Request battery voltage for the given motor ID (default: 0).
  bool requestBatteryVoltage(uint8_t motor_id = 0);

  // Optional low-level writes (no ACK)
  // Write speed command in ERPM (0xAF write).
  bool writeSpeedERPM(uint8_t motor_id, float erpm);
  // Write servo pulse width (0xAF write).
  bool writeServoPulseUs(uint8_t motor_id, float pulse_us);
  // Trigger a device reset (0xAF write).
  bool systemReset(uint8_t motor_id);

  // Poll for one parsed message
  std::optional<Message> poll(int timeout_ms);

private:
  bool sendGeneralRead(uint8_t motor_id, const std::vector<uint8_t> &ids);
  bool sendGeneralWrite(uint8_t motor_id, const std::vector<uint8_t> &ids,
                        const std::vector<float> &data_f32);

  bool tryParseOne(Message &out);

private:
  mutable std::mutex mtx_;
  SerialPort port_;
  std::vector<uint8_t> rx_buffer_;
};

} // namespace KMC_HARDWARE


