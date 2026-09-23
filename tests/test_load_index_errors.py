"""Unit tests for load_index error messages (no live service required)."""

import json
import unittest
from unittest.mock import patch

import urllib3.exceptions

from cyborgdb.client.client import Client
from cyborgdb.exceptions import (
    AuthenticationError,
    CyborgDBError,
    NotFoundError,
    TransportError,
)
from cyborgdb.openapi_client.exceptions import ApiException


def _api_exception(status, detail="synthetic failure"):
    exc = ApiException(status=status, reason="synthetic")
    exc.body = json.dumps({"detail": detail})
    exc.headers = {"X-Request-Id": "req-test"}
    return exc


def _make_client():
    return Client("http://localhost:8000", api_key="test-key")


class TestLoadIndexErrorMessages(unittest.TestCase):
    """load_index names the index in the exception message."""

    def _load_index_raises(self, side_effect):
        client = _make_client()
        key = Client.generate_key()
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=side_effect,
        ):
            with self.assertRaises(CyborgDBError) as ctx:
                client.load_index("my_index", key)
        return ctx.exception, key

    def test_401_raises_authentication_error_with_index_name(self):
        api_exc = _api_exception(401)
        err, key = self._load_index_raises(api_exc)
        self.assertIsInstance(err, AuthenticationError)
        self.assertIn("my_index", str(err))
        self.assertNotIn("{index_name}", str(err))
        self.assertEqual(err.status_code, 401)
        self.assertIs(err.__cause__, api_exc)
        self.assertNotIn(key.hex(), str(err))

    def test_404_raises_not_found_error_with_index_name(self):
        api_exc = _api_exception(404)
        err, key = self._load_index_raises(api_exc)
        self.assertIsInstance(err, NotFoundError)
        self.assertIn("my_index", str(err))
        self.assertNotIn("{index_name}", str(err))
        self.assertEqual(err.status_code, 404)
        self.assertIs(err.__cause__, api_exc)
        self.assertNotIn(key.hex(), str(err))

    def test_transport_error_with_index_name(self):
        transport_exc = urllib3.exceptions.MaxRetryError(None, "/v1/indexes/describe")
        err, key = self._load_index_raises(transport_exc)
        self.assertIsInstance(err, TransportError)
        self.assertIn("my_index", str(err))
        self.assertNotIn("{index_name}", str(err))
        self.assertNotIn(key.hex(), str(err))

    def test_brace_in_name_emitted_verbatim(self):
        """Names containing { or } are not re-interpolated."""
        client = _make_client()
        key = Client.generate_key()
        with patch(
            "cyborgdb.client.encrypted_index.EncryptedIndex._describe",
            side_effect=_api_exception(404),
        ):
            with self.assertRaises(CyborgDBError) as ctx:
                client.load_index("a{b}", key)
        self.assertIn("a{b}", str(ctx.exception))
        self.assertNotIn("{index_name}", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
