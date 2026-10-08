"""HTTP retry behaviour of Client, against a local stub server (no service)."""

import json
import logging
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cyborgdb
from cyborgdb.client.encrypted_index import EncryptedIndex


class _StubHandler(BaseHTTPRequestHandler):
    """Replays ``server.script`` in order: each entry is a (status, headers,
    body) reply, or ``None`` to drop the connection without replying."""

    protocol_version = "HTTP/1.1"
    # The client keeps connections alive; without a timeout an idle one holds
    # its handler thread open indefinitely.
    timeout = 5

    def _reply(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.server.hits += 1
        step = self.server.script.pop(0) if self.server.script else None
        if step is None:
            self.close_connection = True
            return
        status, headers, body = step
        payload = json.dumps(body).encode()
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _reply

    def log_message(self, *args):
        pass


class _StubServerTestCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.server.script = []
        self.server.hits = 0
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.client = cyborgdb.Client(
            f"http://127.0.0.1:{self.server.server_port}", api_key="k"
        )

    def _index(self):
        return EncryptedIndex(
            "idx", b"\x01" * 32, self.client.api, self.client.api_client
        )

    def _calls(self):
        """One GET and one POST, since urllib3 treats the two differently."""
        return {
            "GET list_indexes": self.client.list_indexes,
            "POST train": lambda: self._index().train(),
        }


class TestRetryAfterReachesTheCaller(_StubServerTestCase):
    """A busy reply carrying Retry-After must surface as that reply, not as
    "couldn't reach the service"."""

    CASES = [
        (429, cyborgdb.RateLimitError),
        (503, cyborgdb.ServiceError),
        (413, cyborgdb.CyborgDBError),
    ]

    def test_status_with_retry_after_raises_the_typed_error(self):
        for status, expected in self.CASES:
            for label, call in self._calls().items():
                with self.subTest(status=status, call=label):
                    self.server.hits = 0
                    self.server.script = [
                        (status, {"Retry-After": "7"}, {"detail": "busy"})
                    ]
                    with self.assertRaises(expected) as caught:
                        call()
                    self.assertNotIsInstance(caught.exception, cyborgdb.TransportError)
                    self.assertEqual(caught.exception.status_code, status)
                    self.assertEqual(caught.exception.retry_after, 7.0)
                    self.assertEqual(caught.exception.detail, "busy")
                    self.assertEqual(self.server.hits, 1, "must not be retried")

    def test_status_without_retry_after_is_unchanged(self):
        for status, expected in self.CASES:
            with self.subTest(status=status):
                self.server.script = [(status, {}, {"detail": "busy"})]
                with self.assertRaises(expected) as caught:
                    self.client.list_indexes()
                self.assertIsNone(caught.exception.retry_after)


class TestDroppedConnectionRetry(_StubServerTestCase):
    """The single replay added for idle keep-alive connections the server closed."""

    def test_one_dropped_connection_is_retried(self):
        for label, call in self._calls().items():
            with self.subTest(call=label):
                self.server.hits = 0
                ok = (
                    {"indexes": ["a"]}
                    if label.startswith("GET")
                    else {"status": "success", "message": "ok"}
                )
                self.server.script = [None, (200, {}, ok)]
                call()
                self.assertEqual(self.server.hits, 2)

    def test_a_second_drop_is_not_retried_again(self):
        self.server.script = [None, None, (200, {}, {"indexes": []})]
        with self.assertRaises(cyborgdb.TransportError):
            self.client.list_indexes()
        self.assertEqual(self.server.hits, 2)


if __name__ == "__main__":
    unittest.main()
