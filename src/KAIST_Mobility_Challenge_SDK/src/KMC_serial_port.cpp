#include "KMC_serial_port.hpp"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#endif

namespace KMC_HARDWARE {

SerialPort::SerialPort()
#ifdef _WIN32
    : handle_(INVALID_HANDLE_VALUE)
#else
    : fd_(-1)
#endif
{
}

SerialPort::~SerialPort() { close(); }

bool SerialPort::isOpen() const {
#ifdef _WIN32
  return handle_ != INVALID_HANDLE_VALUE;
#else
  return fd_ >= 0;
#endif
}

void SerialPort::close() {
#ifdef _WIN32
  if (handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(static_cast<HANDLE>(handle_));
    handle_ = INVALID_HANDLE_VALUE;
  }
#else
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
#endif
}

#ifdef _WIN32
static std::string winPortName(const std::string &port_name) {
  // "COM10" and above require "\\\\.\\COM10" syntax
  if (port_name.rfind("\\\\.\\", 0) == 0) {
    return port_name;
  }
  if (port_name.rfind("COM", 0) == 0 && port_name.size() > 4) {
    return "\\\\.\\" + port_name;
  }
  return port_name;
}
#endif

bool SerialPort::open(const std::string &port_name, int baudrate) {
  SerialPortOptions opts;
  opts.baudrate = baudrate;
  return open(port_name, opts);
}

bool SerialPort::open(const std::string &port_name,
                      const SerialPortOptions &opts) {
  close();

#ifdef _WIN32
  std::string name = winPortName(port_name);
  handle_ = CreateFileA(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                        OPEN_EXISTING, 0, nullptr);
  if (handle_ == INVALID_HANDLE_VALUE) {
    return false;
  }

  DCB dcb{};
  dcb.DCBlength = sizeof(DCB);
  if (!GetCommState(static_cast<HANDLE>(handle_), &dcb)) {
    close();
    return false;
  }

  dcb.BaudRate = static_cast<DWORD>(opts.baudrate);
  dcb.ByteSize = 8;
  dcb.Parity = NOPARITY;
  dcb.StopBits = ONESTOPBIT;
  dcb.fBinary = TRUE;
  dcb.fParity = FALSE;

  dcb.fOutxCtsFlow = opts.hw_flow_control ? TRUE : FALSE;
  dcb.fOutxDsrFlow = FALSE;
  dcb.fDsrSensitivity = FALSE;
  dcb.fTXContinueOnXoff = TRUE;
  dcb.fOutX = FALSE;
  dcb.fInX = FALSE;
  dcb.fErrorChar = FALSE;
  dcb.fNull = FALSE;
  dcb.fAbortOnError = FALSE;
  dcb.fDtrControl = DTR_CONTROL_ENABLE;

  if (opts.hw_flow_control) {
    dcb.fRtsControl = RTS_CONTROL_HANDSHAKE;
  } else {
    dcb.fRtsControl =
        opts.rts_always_on ? RTS_CONTROL_ENABLE : RTS_CONTROL_DISABLE;
  }

  if (!SetCommState(static_cast<HANDLE>(handle_), &dcb)) {
    close();
    return false;
  }

  COMMTIMEOUTS timeouts{};
  timeouts.ReadIntervalTimeout = MAXDWORD;
  timeouts.ReadTotalTimeoutMultiplier = 0;
  timeouts.ReadTotalTimeoutConstant = 0;
  timeouts.WriteTotalTimeoutMultiplier = 0;
  timeouts.WriteTotalTimeoutConstant = 0;
  SetCommTimeouts(static_cast<HANDLE>(handle_), &timeouts);

  return true;

#else
  fd_ = ::open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    return false;
  }

  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    close();
    return false;
  }

  // 8N1
  cfmakeraw(&tty);
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= ~(PARENB | CSTOPB);

  // Flow control
#ifdef CRTSCTS
  if (opts.hw_flow_control) {
    tty.c_cflag |= CRTSCTS;
  } else {
    tty.c_cflag &= ~CRTSCTS;
  }
#endif

  speed_t speed;
  switch (opts.baudrate) {
  case 9600:
    speed = B9600;
    break;
  case 19200:
    speed = B19200;
    break;
  case 38400:
    speed = B38400;
    break;
  case 57600:
    speed = B57600;
    break;
  case 115200:
    speed = B115200;
    break;
#ifdef B230400
  case 230400:
    speed = B230400;
    break;
#endif
#ifdef B460800
  case 460800:
    speed = B460800;
    break;
#endif
#ifdef B500000
  case 500000:
    speed = B500000;
    break;
#endif
#ifdef B921600
  case 921600:
    speed = B921600;
    break;
#endif
#ifdef B1000000
  case 1000000:
    speed = B1000000;
    break;
#endif
#ifdef B1500000
  case 1500000:
    speed = B1500000;
    break;
#endif
  default:
    speed = B115200;
    break;
  }
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);

  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    close();
    return false;
  }

  if (opts.rts_always_on) {
#ifdef TIOCMBIS
    int flag = TIOCM_RTS;
    (void)ioctl(fd_, TIOCMBIS, &flag);
#else
    int status = 0;
    if (ioctl(fd_, TIOCMGET, &status) == 0) {
      status |= TIOCM_RTS;
      (void)ioctl(fd_, TIOCMSET, &status);
    }
#endif
  }

  return true;
#endif
}

void SerialPort::flushInput() {
#ifdef _WIN32
  if (!isOpen())
    return;
  PurgeComm(static_cast<HANDLE>(handle_), PURGE_RXCLEAR);
#else
  if (!isOpen())
    return;
  tcflush(fd_, TCIFLUSH);
#endif
}

int SerialPort::writeAll(const uint8_t *data, size_t len) {
  if (!isOpen())
    return -1;

#ifdef _WIN32
  size_t total = 0;
  while (total < len) {
    DWORD written = 0;
    if (!WriteFile(static_cast<HANDLE>(handle_), data + total,
                   static_cast<DWORD>(len - total), &written, nullptr)) {
      return -1;
    }
    if (written == 0)
      return -1;
    total += static_cast<size_t>(written);
  }
  return static_cast<int>(total);
#else
  size_t total = 0;
  while (total < len) {
    ssize_t n = ::write(fd_, data + total, len - total);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return -1;
    }
    total += static_cast<size_t>(n);
  }
  return static_cast<int>(total);
#endif
}

int SerialPort::readSome(uint8_t *data, size_t max_len, int timeout_ms) {
  if (!isOpen())
    return -1;

#ifdef _WIN32
  DWORD bytes_read = 0;
  COMMTIMEOUTS timeouts{};
  timeouts.ReadIntervalTimeout = MAXDWORD;
  timeouts.ReadTotalTimeoutMultiplier = 0;
  timeouts.ReadTotalTimeoutConstant =
      static_cast<DWORD>(timeout_ms < 0 ? 0 : timeout_ms);
  SetCommTimeouts(static_cast<HANDLE>(handle_), &timeouts);

  if (!ReadFile(static_cast<HANDLE>(handle_), data, static_cast<DWORD>(max_len),
                &bytes_read, nullptr)) {
    return -1;
  }
  return static_cast<int>(bytes_read);

#else
  fd_set set;
  FD_ZERO(&set);
  FD_SET(fd_, &set);

  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;

  int rv =
      select(fd_ + 1, &set, nullptr, nullptr, timeout_ms >= 0 ? &tv : nullptr);
  if (rv < 0) {
    return -1;
  }
  if (rv == 0) {
    return 0; // timeout
  }

  ssize_t n = ::read(fd_, data, max_len);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return 0;
    }
    return -1;
  }
  return static_cast<int>(n);
#endif
}

} // namespace KMC_HARDWARE


