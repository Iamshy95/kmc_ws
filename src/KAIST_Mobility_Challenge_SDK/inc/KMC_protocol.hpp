#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace KMC_HARDWARE::protocol {

// Packet headers
constexpr uint8_t HEADER_GENERAL = 0xAF;
constexpr uint8_t HEADER_PC_CONTROL = 0xA5;
constexpr uint8_t HEADER_VEHICLE_SPEED = 0xB3;

// RW
constexpr uint8_t RW_READ = 0x00;
constexpr uint8_t RW_WRITE = 0x01;

// Supported 0xAF IDs
constexpr uint8_t ID_SYSTEM_RESET = 0x00;
constexpr uint8_t ID_SPEED = 0x03;
constexpr uint8_t ID_SERVO_PULSE = 0x05;
constexpr uint8_t ID_ALL_STATE = 0x06;
constexpr uint8_t ID_BATTERY_VOLTAGE = 0x07;

constexpr uint8_t MAX_IDS = 16;
constexpr uint8_t ALL_STATE_FIELD_COUNT = 9;

// Helpers (le)
inline void appendU8(std::vector<uint8_t> &out, uint8_t v) { out.push_back(v); }

inline void appendU32LE(std::vector<uint8_t> &out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

inline uint32_t readU32LE(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0])) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

inline void appendFloatLE(std::vector<uint8_t> &out, float value) {
  uint32_t raw = 0;
  static_assert(sizeof(float) == sizeof(uint32_t), "float32 required");
  std::memcpy(&raw, &value, sizeof(float));
  appendU32LE(out, raw);
}

inline float readFloatLE(const uint8_t *p) {
  uint32_t raw = readU32LE(p);
  float v = 0.0f;
  std::memcpy(&v, &raw, sizeof(float));
  return v;
}

} // namespace KMC_HARDWARE::protocol
