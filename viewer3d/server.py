"""Máy chủ cục bộ của viewer 3D: phục vụ trang web tĩnh + một kênh WebSocket.

Chỉ dùng thư viện chuẩn (http.server + WebSocket tối giản theo RFC 6455) để không thêm phụ thuộc.
Mặc định chỉ nghe trên 127.0.0.1.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import socket
import struct
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .session import Session

WEB_DIR = Path(__file__).resolve().parent / "web"
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_QUEUED_FRAMES = 3      # khách chậm (tab chạy nền): bỏ khung cũ thay vì làm nghẽn mô phỏng
_MAX_COMMAND_BYTES = 4096

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


def _ws_frame(payload: bytes, opcode: int) -> bytes:
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < 65536:
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    return head + payload


class Client:
    """Một kết nối WebSocket: hàng đợi gửi riêng + luồng ghi riêng."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._q: deque[tuple[bytes, bool]] = deque()
        self._cond = threading.Condition()
        self.alive = True
        threading.Thread(target=self._writer, daemon=True, name="ws-writer").start()

    def offer(self, data: bytes | str, important: bool = False, opcode: int | None = None) -> None:
        """Xếp một khung vào hàng đợi; chỉ luồng ghi gửi lên socket để các khung không bị xen nhau."""
        if opcode is None:
            opcode = 0x1 if isinstance(data, str) else 0x2
        frame = _ws_frame(data.encode() if isinstance(data, str) else data, opcode)
        with self._cond:
            if not important and sum(1 for _, imp in self._q if not imp) >= _MAX_QUEUED_FRAMES:
                for i, (_, imp) in enumerate(self._q):
                    if not imp:
                        del self._q[i]
                        break
            self._q.append((frame, important))
            self._cond.notify()

    def _writer(self) -> None:
        try:
            while self.alive:
                with self._cond:
                    while not self._q and self.alive:
                        self._cond.wait(0.5)
                    if not self.alive:
                        break
                    frame, _ = self._q.popleft()
                self.sock.sendall(frame)
        except OSError:
            pass
        finally:
            self.alive = False

    def close(self) -> None:
        self.alive = False
        with self._cond:
            self._cond.notify()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MarioViewer3D"
    session: Session

    def log_message(self, fmt, *args) -> None:  # noqa: D102 - im lặng, tránh trôi log
        pass

    # ------------------------------------------------------------------ HTTP
    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/ws":
            return self._websocket()
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if WEB_DIR not in target.parents or not target.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/"):
            ctype += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------- WebSocket
    def _local_origin(self) -> bool:
        """Chỉ nhận kết nối từ chính trang viewer (chặn trang web lạ gọi vào cổng cục bộ)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlsplit(origin).netloc == self.headers.get("Host")

    def _recv(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if len(data) != n:
            raise ConnectionError("đóng kết nối")
        return data

    def _websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if self.headers.get("Upgrade", "").lower() != "websocket" or not key or not self._local_origin():
            return self.send_error(HTTPStatus.BAD_REQUEST)
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        client = Client(self.connection)
        self.session.add_client(client)
        try:
            text = b""
            while client.alive:
                b0, b1 = self._recv(2)
                opcode, n = b0 & 0x0F, b1 & 0x7F
                if n == 126:
                    n = struct.unpack("!H", self._recv(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", self._recv(8))[0]
                if n > _MAX_COMMAND_BYTES or not b1 & 0x80:      # khách phải che (mask) mọi khung
                    break
                mask = self._recv(4)
                payload = bytes(c ^ mask[i & 3] for i, c in enumerate(self._recv(n)))
                if opcode == 0x8:                                # đóng
                    break
                if opcode == 0x9:                                # ping -> pong (qua luồng ghi)
                    client.offer(payload, important=True, opcode=0xA)
                elif opcode in (0x0, 0x1):
                    text += payload
                    if b0 & 0x80:
                        self._command(text)
                        text = b""
        except (OSError, ConnectionError, ValueError):
            pass
        finally:
            self.session.remove_client(client)
            client.close()

    def _command(self, raw: bytes) -> None:
        try:
            cmd = json.loads(raw)
        except ValueError:
            return
        if not isinstance(cmd, dict):
            return
        if cmd.get("cmd") == "keys":                 # phím người chơi: áp dụng ngay, không qua hàng đợi
            value = cmd.get("value", 0)
            if isinstance(value, int) and not isinstance(value, bool):
                self.session.keys = value & 0xFF
        elif cmd.get("cmd") != "_join":
            self.session.cmds.put(cmd)


def serve(session: Session, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"session": session})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd
