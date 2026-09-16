"""Conformance tests for the typed exception taxonomy
"""

import json
import unittest

import urllib3.exceptions

import cyborgdb
from cyborgdb.exceptions import (
    AuthenticationError,
    ConflictError,
    CyborgDBError,
    NotFoundError,
    RateLimitError,
    ServiceError,
    TransportError,
    ValidationError,
    translate_api_error,
)
from cyborgdb.openapi_client.exceptions import ApiException

# status -> (class, retryable).
TAXONOMY = [
    (400, ValidationError, False),
    (422, ValidationError, False),
    (401, AuthenticationError, False),
    (403, AuthenticationError, False),
    (404, NotFoundError, False),
    (409, ConflictError, False),
    (429, RateLimitError, True),
    (500, ServiceError, True),
    (502, ServiceError, True),
    (503, ServiceError, True),
    (504, ServiceError, True),
]

# Concept name -> class, for the public-surface checks.
CONCEPTS = {
    "ValidationError": ValidationError,
    "AuthenticationError": AuthenticationError,
    "NotFoundError": NotFoundError,
    "ConflictError": ConflictError,
    "RateLimitError": RateLimitError,
    "ServiceError": ServiceError,
    "TransportError": TransportError,
}


def _api_exception(status, detail="synthetic failure", headers=None):
    """Build an ApiException the way the generated client would."""
    exc = ApiException(status=status, reason="synthetic")
    exc.body = json.dumps({"detail": detail})
    exc.headers = headers if headers is not None else {"X-Request-Id": "req-abc123"}
    return exc


class TestStatusMapping(unittest.TestCase):
    """Every status the taxonomy names produces its type and retryability."""

    def test_status_mapping(self):
        for status, expected, retryable in TAXONOMY:
            with self.subTest(status=status):
                result = translate_api_error(_api_exception(status), "op failed")
                self.assertIsInstance(result, expected)
                self.assertEqual(result.retryable, retryable)
                self.assertEqual(result.status_code, status)


class TestTranslation(unittest.TestCase):
    def test_populates_context_fields(self):
        exc = _api_exception(
            503, headers={"X-Request-Id": "req-abc123", "Retry-After": "2.5"}
        )
        result = translate_api_error(exc, "op failed")
        self.assertIsInstance(result, ServiceError)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.request_id, "req-abc123")
        self.assertEqual(result.detail, "synthetic failure")
        self.assertEqual(result.retry_after, 2.5)
        self.assertTrue(result.retryable)

    def test_every_type_subclasses_the_base(self):
        for name, cls in CONCEPTS.items():
            with self.subTest(concept=name):
                self.assertTrue(issubclass(cls, CyborgDBError))

    def test_network_failures_become_transport_errors(self):
        for exc in (
            urllib3.exceptions.TimeoutError("timed out"),
            urllib3.exceptions.NewConnectionError(None, "connection refused"),
            urllib3.exceptions.ProtocolError("connection aborted"),
        ):
            with self.subTest(exc=type(exc).__name__):
                result = translate_api_error(exc, "op failed")
                self.assertIsInstance(result, TransportError)
                self.assertIsNone(result.status_code)
                self.assertTrue(result.retryable)

    def test_unnamed_status_stays_a_value_error(self):
        # 418 is not in the taxonomy: the previous untyped behavior is preserved
        # rather than inventing a type for it.
        result = translate_api_error(_api_exception(418), "op failed")
        self.assertIsInstance(result, ValueError)
        self.assertNotIsInstance(result, CyborgDBError)

    def test_non_api_exceptions_pass_through(self):
        original = KeyError("unrelated")
        self.assertIs(translate_api_error(original, "op failed"), original)

    def test_missing_headers_do_not_raise(self):
        exc = ApiException(status=500, reason="synthetic")
        result = translate_api_error(exc, "op failed")
        self.assertIsInstance(result, ServiceError)
        self.assertIsNone(result.request_id)
        self.assertIsNone(result.retry_after)


class TestPublicSurface(unittest.TestCase):
    def test_importable_from_the_exceptions_module(self):
        from cyborgdb.exceptions import (  # noqa: F401
            AuthenticationError,
            ConflictError,
            CyborgDBError,
            NotFoundError,
            RateLimitError,
            ServiceError,
            TransportError,
            ValidationError,
        )

    def test_reexported_at_the_package_top_level(self):
        for name, cls in CONCEPTS.items():
            with self.subTest(concept=name):
                self.assertIs(getattr(cyborgdb, name), cls)
        self.assertIs(cyborgdb.CyborgDBError, CyborgDBError)


if __name__ == "__main__":
    unittest.main()
