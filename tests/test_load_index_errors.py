"""Unit tests for load_index error messages — no running service required."""

import json
import unittest
from unittest.mock import MagicMock, patch

import urllib3.exceptions

from cyborgdb.client.client import Client
from cyborgdb.exceptions import (
    AuthenticationError,
    CyborgDBError,
    NotFoundError,
    TransportError,
)
from cyborgdb.openapi_client.exceptions import ApiException


def _api_exception(status):
    exc = ApiException(status=status, reason="synthetic")
    exc.body = json.dumps({"detail": "synthetic"})
    exc.headers = {"X-Request-Id": "req-test"}
    return exc


def _make_client():
    """Build a Client wired to a mock API — no network needed."""
    mock_api = MagicMock()
    mock_api_client = MagicMock()
    mock_api_client.configuration.api_key = {}
    client = Client.__new__(Client)
    client.api = mock_api
    client.api_client = mock_api_client
    return client


class TestLoadIndexErrorMessages(unittest.TestCase):
    def _load(self, client, name, key):
        with self.assertRaises(CyborgDBError) as ctx:
            client.load_index(name, key)
        return ctx.exception

    def test_401_raises_authentication_error_with_index_name(self):
        client = _make_client()
        key = b"\x01" * 32
        api_exc = _api_exception(401)
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=api_exc,
        ):
            err = self._load(client, "my_index", key)
        self.assertIsInstance(err, AuthenticationError)
        self.assertIn("my_index", str(err))
        self.assertNotIn("{index_name}", str(err))
        self.assertEqual(err.status_code, 401)
        self.assertIs(err.__cause__, api_exc)

    def test_404_raises_not_found_error_with_index_name(self):
        client = _make_client()
        key = b"\x02" * 32
        api_exc = _api_exception(404)
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=api_exc,
        ):
            err = self._load(client, "my_index", key)
        self.assertIsInstance(err, NotFoundError)
        self.assertIn("my_index", str(err))
        self.assertNotIn("{index_name}", str(err))
        self.assertEqual(err.status_code, 404)
        self.assertIs(err.__cause__, api_exc)

    def test_max_retry_error_raises_transport_error_with_index_name(self):
        client = _make_client()
        key = b"\x03" * 32
        transport_exc = urllib3.exceptions.MaxRetryError(
            pool=None, url="/", reason=None
        )
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=transport_exc,
        ):
            err = self._load(client, "my_index", key)
        self.assertIsInstance(err, TransportError)
        self.assertIn("my_index", str(err))

    def test_key_hex_not_in_error_message(self):
        client = _make_client()
        key = bytes(range(32))
        key_hex = key.hex()
        api_exc = _api_exception(401)
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=api_exc,
        ):
            err = self._load(client, "my_index", key)
        self.assertNotIn(key_hex, str(err))

    def test_curly_brace_name_appears_verbatim(self):
        client = _make_client()
        key = b"\x05" * 32
        api_exc = _api_exception(404)
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=api_exc,
        ):
            err = self._load(client, "a{b}", key)
        self.assertIn("a{b}", str(err))

    def test_single_quote_name_appears_verbatim(self):
        # Spec decision: names with single quotes are emitted unescaped (acceptable
        # for human-read messages); this test pins that we chose not to escape.
        client = _make_client()
        key = b"\x06" * 32
        api_exc = _api_exception(404)
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=api_exc,
        ):
            err = self._load(client, "a'b", key)
        self.assertIn("a'b", str(err))


if __name__ == "__main__":
    unittest.main()
