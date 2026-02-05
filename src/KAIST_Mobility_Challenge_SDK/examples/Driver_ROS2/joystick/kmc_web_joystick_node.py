#!/usr/bin/env python3
import base64
import hashlib
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


class _NodeHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, node):
        super().__init__(server_address, handler_cls)
        self.node = node


class _Handler(BaseHTTPRequestHandler):
    server_version = "KMCWebJoystick/1.0"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        node = self.server.node
        path = urlparse(self.path).path
        if path == node.ws_path:
            self._handle_websocket()
            return
        if path in ("/", "/index.html"):
            body = node.index_page
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        node = self.server.node
        path = urlparse(self.path).path
        if path != "/cmd":
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            linear = float(payload.get("linear", 0.0))
            angular = float(payload.get("angular", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            self.send_error(400, "Bad JSON")
            return
        node.set_command(linear, angular)
        self.send_response(204)
        self.end_headers()

    def _handle_websocket(self):
        node = self.server.node
        upgrade = self.headers.get("Upgrade", "").lower()
        if upgrade != "websocket":
            self.send_error(400, "WebSocket upgrade required")
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "WebSocket key missing")
            return
        accept = self._ws_accept(key)

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        self.connection.settimeout(1.0)
        try:
            self._websocket_loop(node)
        except Exception:
            return

    @staticmethod
    def _ws_accept(key):
        guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        sha1 = hashlib.sha1((key + guid).encode("utf-8")).digest()
        return base64.b64encode(sha1).decode("utf-8")

    def _recv_exact(self, size):
        data = b""
        while len(data) < size:
            chunk = self.connection.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _recv_frame(self):
        header = self._recv_exact(2)
        if not header:
            return None
        b1, b2 = header
        fin = (b1 & 0x80) != 0
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F

        if length == 126:
            ext = self._recv_exact(2)
            if not ext:
                return None
            length = int.from_bytes(ext, "big")
        elif length == 127:
            ext = self._recv_exact(8)
            if not ext:
                return None
            length = int.from_bytes(ext, "big")

        if length > 65536:
            return None

        if masked:
            mask = self._recv_exact(4)
            if not mask:
                return None
        else:
            return None

        payload = self._recv_exact(length) if length else b""
        if payload is None:
            return None
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if not fin:
            return None
        return opcode, payload

    def _send_frame(self, opcode, payload):
        payload = payload or b""
        b1 = 0x80 | (opcode & 0x0F)
        length = len(payload)
        if length < 126:
            header = bytes([b1, length])
        elif length < 65536:
            header = bytes([b1, 126]) + length.to_bytes(2, "big")
        else:
            header = bytes([b1, 127]) + length.to_bytes(8, "big")
        self.connection.sendall(header + payload)

    def _websocket_loop(self, node):
        while not node.shutdown_event.is_set():
            try:
                frame = self._recv_frame()
            except socket.timeout:
                continue
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x8:
                break
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode != 0x1:
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
                linear = float(data.get("linear", 0.0))
                angular = float(data.get("angular", 0.0))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            node.set_command(linear, angular)


class KmcWebJoystickNode(Node):
    def __init__(self):
        super().__init__("kmc_web_joystick_node")
        self.declare_parameter("bind_address", "127.0.0.1")
        self.declare_parameter("http_port", 8080)
        self.declare_parameter("ws_path", "/ws")
        self.declare_parameter("web_root", "")
        self.declare_parameter("cmd_topic", "cmd_vel")
        self.declare_parameter("rx_topic", "vehicle_speed")
        self.declare_parameter("tx_rate_topic", "cmd_vel_tx_rate_hz")
        self.declare_parameter("rx_rate_topic", "driver_rx_rate_hz")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("rate_publish_hz", 1.0)
        self.declare_parameter("command_timeout_ms", 200)
        self.declare_parameter("linear_limit", 2.0)
        self.declare_parameter("angular_limit", 2.0)
        self.declare_parameter("deadzone", 0.05)

        self.bind_address = self.get_parameter("bind_address").value
        self.http_port = int(self.get_parameter("http_port").value)
        self.ws_path = self.get_parameter("ws_path").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.rx_topic = self.get_parameter("rx_topic").value
        self.tx_rate_topic = self.get_parameter("tx_rate_topic").value
        self.rx_rate_topic = self.get_parameter("rx_rate_topic").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.rate_publish_hz = float(self.get_parameter("rate_publish_hz").value)
        self.command_timeout_ms = int(self.get_parameter("command_timeout_ms").value)
        self.linear_limit = float(self.get_parameter("linear_limit").value)
        self.angular_limit = float(self.get_parameter("angular_limit").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.web_root = self._find_web_root(self.get_parameter("web_root").value)

        self._cmd_lock = threading.Lock()
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._last_cmd_ts = None
        self.shutdown_event = threading.Event()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.tx_rate_pub = self.create_publisher(Float32, self.tx_rate_topic, 10)
        self.rx_rate_pub = self.create_publisher(Float32, self.rx_rate_topic, 10)

        self._tx_samples = 0
        self._rx_samples = 0
        self._rate_t0 = time.monotonic()

        if self.rx_topic:
            self.rx_sub = self.create_subscription(
                Float32, self.rx_topic, self._on_speed, 10
            )

        if self.publish_hz > 0.0:
            self.create_timer(1.0 / self.publish_hz, self._publish_cmd_vel)
        if self.rate_publish_hz > 0.0:
            self.create_timer(1.0 / self.rate_publish_hz, self._publish_rate_stats)

        self.index_page = self._load_index_page()
        self.httpd = None
        self.httpd_thread = None
        if self.http_port > 0:
            self._start_http_server()

    def _find_web_root(self, override):
        if override:
            return override
        if get_package_share_directory is None:
            return ""
        try:
            share_dir = get_package_share_directory("kmc_hardware_driver_node")
        except Exception:
            return ""
        return os.path.join(share_dir, "web")

    def _load_index_page(self):
        if self.web_root:
            path = os.path.join(self.web_root, "index.html")
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()
        fallback = b"<!doctype html><title>Web UI not installed</title>"
        return fallback

    def _start_http_server(self):
        try:
            self.httpd = _NodeHTTPServer(
                (self.bind_address, self.http_port), _Handler, self
            )
        except OSError as exc:
            self.get_logger().error(f"HTTP server failed: {exc}")
            return
        self.httpd_thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.httpd_thread.start()
        self.get_logger().info(f"Web UI at http://{self.bind_address}:{self.http_port}")

    def set_command(self, linear, angular):
        with self._cmd_lock:
            self._cmd_v = float(linear)
            self._cmd_w = float(angular)
            self._last_cmd_ts = time.monotonic()

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _clamp(self, value, limit):
        if limit <= 0.0:
            return value
        return max(-limit, min(limit, value))

    def _publish_cmd_vel(self):
        now = time.monotonic()
        with self._cmd_lock:
            linear = self._cmd_v
            angular = self._cmd_w
            last_time = self._last_cmd_ts

        if last_time is None:
            linear = 0.0
            angular = 0.0
        elif self.command_timeout_ms > 0:
            timeout_s = self.command_timeout_ms / 1000.0
            if (now - last_time) > timeout_s:
                linear = 0.0
                angular = 0.0

        linear = self._clamp(self._apply_deadzone(linear), self.linear_limit)
        angular = self._clamp(self._apply_deadzone(angular), self.angular_limit)

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)
        self._tx_samples += 1

    def _on_speed(self, _msg):
        self._rx_samples += 1

    def _publish_rate_stats(self):
        now = time.monotonic()
        dt = now - self._rate_t0
        if dt <= 0.0:
            return
        tx_rate = self._tx_samples / dt
        rx_rate = self._rx_samples / dt

        tx = Float32()
        tx.data = float(tx_rate)
        rx = Float32()
        rx.data = float(rx_rate)
        self.tx_rate_pub.publish(tx)
        self.rx_rate_pub.publish(rx)

        self._tx_samples = 0
        self._rx_samples = 0
        self._rate_t0 = now

    def shutdown(self):
        self.shutdown_event.set()
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self.httpd_thread:
            self.httpd_thread.join(timeout=1.0)
            self.httpd_thread = None


def main():
    rclpy.init()
    node = KmcWebJoystickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
