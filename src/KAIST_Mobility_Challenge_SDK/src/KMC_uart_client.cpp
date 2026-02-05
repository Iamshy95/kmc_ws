#include "KMC_uart_client.hpp"

#include <algorithm>

namespace KMC_HARDWARE {

bool UartClient::open(const std::string &port_name, int baudrate) {
  std::lock_guard<std::mutex> lk(mtx_);
  rx_buffer_.clear();
  return port_.open(port_name, baudrate);
}

bool UartClient::open(const std::string &port_name,
                      const SerialPortOptions &opts) {
  std::lock_guard<std::mutex> lk(mtx_);
  rx_buffer_.clear();
  return port_.open(port_name, opts);
}

void UartClient::close() {
  std::lock_guard<std::mutex> lk(mtx_);
  port_.close();
  rx_buffer_.clear();
}

bool UartClient::isOpen() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return port_.isOpen();
}

void UartClient::flushInput() {
  std::lock_guard<std::mutex> lk(mtx_);
  port_.flushInput();
  rx_buffer_.clear();
}

bool UartClient::sendPcControl(float velocity_mps, float curvature_1pm) {
  std::lock_guard<std::mutex> lk(mtx_);
  std::vector<uint8_t> out;
  out.reserve(9);
  protocol::appendU8(out, protocol::HEADER_PC_CONTROL);
  protocol::appendFloatLE(out, velocity_mps);
  protocol::appendFloatLE(out, curvature_1pm);
  return port_.writeAll(out.data(), out.size()) == static_cast<int>(out.size());
}

bool UartClient::requestVehicleSpeed() {
  std::lock_guard<std::mutex> lk(mtx_);
  const uint8_t b = protocol::HEADER_VEHICLE_SPEED;
  return port_.writeAll(&b, 1) == 1;
}

bool UartClient::sendGeneralRead(uint8_t motor_id,
                                 const std::vector<uint8_t> &ids) {
  if (ids.size() > protocol::MAX_IDS)
    return false;

  std::vector<uint8_t> out;
  out.reserve(4 + ids.size());
  protocol::appendU8(out, protocol::HEADER_GENERAL);
  protocol::appendU8(out, motor_id);
  protocol::appendU8(out, protocol::RW_READ);
  protocol::appendU8(out, static_cast<uint8_t>(ids.size()));
  out.insert(out.end(), ids.begin(), ids.end());

  return port_.writeAll(out.data(), out.size()) == static_cast<int>(out.size());
}

bool UartClient::sendGeneralWrite(uint8_t motor_id,
                                  const std::vector<uint8_t> &ids,
                                  const std::vector<float> &data_f32) {
  if (ids.size() != data_f32.size())
    return false;
  if (ids.size() > protocol::MAX_IDS)
    return false;

  std::vector<uint8_t> out;
  out.reserve(4 + ids.size() + 4 * ids.size());
  protocol::appendU8(out, protocol::HEADER_GENERAL);
  protocol::appendU8(out, motor_id);
  protocol::appendU8(out, protocol::RW_WRITE);
  protocol::appendU8(out, static_cast<uint8_t>(ids.size()));
  out.insert(out.end(), ids.begin(), ids.end());
  for (float v : data_f32) {
    protocol::appendFloatLE(out, v);
  }

  return port_.writeAll(out.data(), out.size()) == static_cast<int>(out.size());
}

bool UartClient::requestAllState(uint8_t motor_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  return sendGeneralRead(motor_id, {protocol::ID_ALL_STATE});
}

bool UartClient::requestBatteryVoltage(uint8_t motor_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  return sendGeneralRead(motor_id, {protocol::ID_BATTERY_VOLTAGE});
}

bool UartClient::writeSpeedERPM(uint8_t motor_id, float erpm) {
  std::lock_guard<std::mutex> lk(mtx_);
  return sendGeneralWrite(motor_id, {protocol::ID_SPEED}, {erpm});
}

bool UartClient::writeServoPulseUs(uint8_t motor_id, float pulse_us) {
  std::lock_guard<std::mutex> lk(mtx_);
  return sendGeneralWrite(motor_id, {protocol::ID_SERVO_PULSE}, {pulse_us});
}

bool UartClient::systemReset(uint8_t motor_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  std::vector<uint8_t> out;
  out.reserve(5);
  protocol::appendU8(out, protocol::HEADER_GENERAL);
  protocol::appendU8(out, motor_id);
  protocol::appendU8(out, protocol::RW_WRITE);
  protocol::appendU8(out, 1);
  protocol::appendU8(out, protocol::ID_SYSTEM_RESET);
  return port_.writeAll(out.data(), out.size()) == static_cast<int>(out.size());
}

std::optional<Message> UartClient::poll(int timeout_ms) {
  std::lock_guard<std::mutex> lk(mtx_);

  Message msg;
  if (tryParseOne(msg)) {
    return msg;
  }

  uint8_t buf[512];
  int n = port_.readSome(buf, sizeof(buf), timeout_ms);
  if (n < 0)
    return std::nullopt;
  if (n > 0) {
    rx_buffer_.insert(rx_buffer_.end(), buf, buf + n);
  }

  if (tryParseOne(msg)) {
    return msg;
  }
  return std::nullopt;
}

bool UartClient::tryParseOne(Message &out) {
  while (true) {
    if (rx_buffer_.empty())
      return false;

    auto it_af = std::find(rx_buffer_.begin(), rx_buffer_.end(),
                           protocol::HEADER_GENERAL);
    auto it_b3 = std::find(rx_buffer_.begin(), rx_buffer_.end(),
                           protocol::HEADER_VEHICLE_SPEED);

    if (it_af == rx_buffer_.end() && it_b3 == rx_buffer_.end()) {
      rx_buffer_.clear();
      return false;
    }

    auto header_it = it_af;
    if (header_it == rx_buffer_.end() ||
        (it_b3 != rx_buffer_.end() && it_b3 < header_it)) {
      header_it = it_b3;
    }

    if (header_it != rx_buffer_.begin()) {
      rx_buffer_.erase(rx_buffer_.begin(), header_it);
    }

    uint8_t header = rx_buffer_[0];

    // B3 response 5 bytes
    if (header == protocol::HEADER_VEHICLE_SPEED) {
      if (rx_buffer_.size() < 5)
        return false;

      VehicleSpeed vs;
      vs.mps = protocol::readFloatLE(&rx_buffer_[1]);

      rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + 5);
      out = vs;
      return true;
    }

    // AF response least 4 bytes
    if (header == protocol::HEADER_GENERAL) {
      if (rx_buffer_.size() < 4)
        return false;

      uint8_t motor_id = rx_buffer_[1];
      uint8_t rw = rx_buffer_[2];
      uint8_t n_id = rx_buffer_[3];

      if (rw != protocol::RW_READ && rw != protocol::RW_WRITE) {
        // corrupted; drop header and resync
        rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + 1);
        continue;
      }

      if (n_id > protocol::MAX_IDS) {
        rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + 1);
        continue;
      }

      size_t total_len = 4 + static_cast<size_t>(n_id);
      if (rw == protocol::RW_WRITE) {
        total_len += 4 * static_cast<size_t>(n_id);
      }

      if (rx_buffer_.size() < total_len)
        return false;

      AfResponse resp;
      resp.motor_id = motor_id;
      resp.rw = rw;
      resp.ids.assign(rx_buffer_.begin() + 4, rx_buffer_.begin() + 4 + n_id);

      if (rw == protocol::RW_WRITE && n_id > 0) {
        resp.data_f32.resize(n_id);
        resp.data_u32.resize(n_id);
        size_t data_off = 4 + n_id;
        for (size_t i = 0; i < n_id; ++i) {
          const uint8_t *p = &rx_buffer_[data_off + i * 4];
          resp.data_f32[i] = protocol::readFloatLE(p);
          resp.data_u32[i] = protocol::readU32LE(p);
        }
      }

      rx_buffer_.erase(rx_buffer_.begin(),
                       rx_buffer_.begin() + static_cast<long>(total_len));

      // decoding
      if (resp.rw == protocol::RW_WRITE && resp.ids.size() == 1 &&
          resp.ids[0] == protocol::ID_BATTERY_VOLTAGE &&
          resp.data_f32.size() == 1) {
        BatteryVoltage bv;
        bv.volt = resp.data_f32[0];
        out = bv;
        return true;
      }

      if (resp.rw == protocol::RW_WRITE &&
          resp.ids.size() == protocol::ALL_STATE_FIELD_COUNT &&
          resp.data_f32.size() == protocol::ALL_STATE_FIELD_COUNT &&
          resp.data_u32.size() == protocol::ALL_STATE_FIELD_COUNT) {
        bool all6 = true;
        for (uint8_t id : resp.ids) {
          if (id != protocol::ID_ALL_STATE) {
            all6 = false;
            break;
          }
        }
        if (all6) {
          AllState st;
          st.id = resp.data_u32[0];
          st.position_deg = resp.data_f32[1];
          st.speed_rpm = resp.data_f32[2];
          st.current_A = resp.data_f32[3];
          st.temperature_C = resp.data_f32[4];
          st.error_code = resp.data_u32[5];
          st.reserved_0 = resp.data_f32[6];
          st.reserved_1 = resp.data_f32[7];
          st.reserved_2 = resp.data_f32[8];
          out = st;
          return true;
        }
      }

      out = resp;
      return true;
    }

    // Unknown header
    rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + 1);
  }
}

} // namespace KMC_HARDWARE

