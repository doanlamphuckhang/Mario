"""Viewer server checks without sockets, browsers or a running emulator loop."""
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viewer3d import server


class ViewerServerTests(unittest.TestCase):
    def test_commands_ignore_malformed_values(self):
        session = SimpleNamespace(keys=0, cmds=queue.Queue())
        handler = SimpleNamespace(session=session)
        for raw in (b'{"cmd": "keys", "value": [1]}', b'{"cmd": "keys", "value": true}',
                    b'{"cmd": "keys", "value": "x"}', b'[1, 2]', b'not json'):
            server.Handler._command(handler, raw)
        self.assertEqual(session.keys, 0)
        server.Handler._command(handler, b'{"cmd": "keys", "value": 385}')
        self.assertEqual(session.keys, 385 & 0xFF)
        server.Handler._command(handler, b'{"cmd": "_join", "client": 1}')
        server.Handler._command(handler, b'{"cmd": "pause", "value": true}')
        self.assertEqual(session.cmds.get_nowait(), {'cmd': 'pause', 'value': True})
        self.assertTrue(session.cmds.empty())

    def test_pong_goes_through_writer_queue(self):
        with patch.object(server.threading, 'Thread'):
            client = server.Client(sock=None)
        client.offer(b'ping-data', important=True, opcode=0xA)
        frame, important = client._q[0]
        self.assertTrue(important)
        self.assertEqual(frame, server._ws_frame(b'ping-data', 0xA))
        self.assertEqual(frame[0] & 0x0F, 0xA)

    def test_slow_client_drops_old_frames_but_keeps_events(self):
        with patch.object(server.threading, 'Thread'):
            client = server.Client(sock=None)
        client.offer('event', important=True)
        for i in range(server._MAX_QUEUED_FRAMES + 2):
            client.offer(bytes([i]))
        frames = [frame for frame, important in client._q if not important]
        self.assertEqual(len(frames), server._MAX_QUEUED_FRAMES)
        self.assertEqual(sum(important for _, important in client._q), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
