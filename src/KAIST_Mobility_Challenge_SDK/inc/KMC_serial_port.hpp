#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace KMC_HARDWARE {

// Serial port options for open().
struct SerialPortOptions {
  int baudrate = 115200;

  // Enable RTS/CTS flow control. enabled -> TX wait for CTS
  bool hw_flow_control = true;

  // Keep RTS ready.
  bool rts_always_on = true;
};

class SerialPort {
public:
  SerialPort();
  ~SerialPort();

  SerialPort(const SerialPort &) = delete;
  SerialPort &operator=(const SerialPort &) = delete;

  bool open(const std::string &port_name, const SerialPortOptions &opts);

  bool open(const std::string &port_name, int baudrate);

  void close();
  bool isOpen() const;

  // Flush unread bytes from the OS buffer (best-effort)
  void flushInput();

  // Blocking write, returns bytes written or -1 on error
  int writeAll(const uint8_t *data, size_t len);

  // Read with timeout; returns bytes read (0 on timeout) or -1 on error.
  int readSome(uint8_t *data, size_t max_len, int timeout_ms);

private:
#ifdef _WIN32
  void *handle_;
#else
  int fd_;
#endif
};

} // namespace KMC_HARDWARE
