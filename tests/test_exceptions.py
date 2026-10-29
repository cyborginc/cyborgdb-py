"""Conformance tests for the typed exception taxonomy"""

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
    _ArgumentTypeError,
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


def _api_exception(status, detail="synthetic failure", headers=None, **body):
    """Build an ApiException the way the generated client would."""
    exc = ApiException(status=status, reason="synthetic")
    exc.body = json.dumps({"status_code": status, "detail": detail, **body})
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

    def test_unnamed_status_is_the_base_error(self):
        for status in (405, 413, 418):
            with self.subTest(status=status):
                result = translate_api_error(_api_exception(status), "op failed")
                self.assertIs(type(result), CyborgDBError)
                self.assertEqual(result.status_code, status)
                self.assertEqual(result.request_id, "req-abc123")
                self.assertFalse(result.retryable)

    def test_argument_type_errors_stay_type_errors(self):
        exc = _ArgumentTypeError("bad arg")
        self.assertIsInstance(exc, ValidationError)
        self.assertIsInstance(exc, TypeError)
        self.assertIsNone(exc.status_code)

    def test_non_api_exceptions_pass_through(self):
        original = KeyError("unrelated")
        self.assertIs(translate_api_error(original, "op failed"), original)

    def test_missing_headers_do_not_raise(self):
        exc = ApiException(status=500, reason="synthetic")
        result = translate_api_error(exc, "op failed")
        self.assertIsInstance(result, ServiceError)
        self.assertIsNone(result.request_id)
        self.assertIsNone(result.retry_after)


class TestKinds(unittest.TestCase):
    def test_service_kind_is_carried(self):
        for status, kind, expected in (
            (401, "wrong_key", AuthenticationError),
            (401, "unauthenticated", AuthenticationError),
            (403, "quota_exceeded", AuthenticationError),
            (403, "permission_denied", AuthenticationError),
            (400, "failed_precondition", ValidationError),
            (500, "data_loss", ServiceError),
        ):
            with self.subTest(kind=kind):
                result = translate_api_error(
                    _api_exception(status, kind=kind), "op failed"
                )
                self.assertIsInstance(result, expected)
                self.assertEqual(result.kind, kind)

    def test_unknown_kind_is_passed_through(self):
        result = translate_api_error(
            _api_exception(400, kind="some_new_kind"), "op failed"
        )
        self.assertIsInstance(result, ValidationError)
        self.assertEqual(result.kind, "some_new_kind")

    def test_kind_falls_back_to_the_status(self):
        for status, kind in (
            (400, "invalid_argument"),
            (401, "unauthenticated"),
            (403, "permission_denied"),
            (404, "not_found"),
            (405, "invalid_argument"),
            (409, "already_exists"),
            (422, "invalid_argument"),
            (429, "rate_limited"),
            (500, "internal"),
            (503, "unavailable"),
        ):
            with self.subTest(status=status):
                result = translate_api_error(_api_exception(status), "op failed")
                self.assertEqual(result.kind, kind)

    def test_kind_without_a_json_body(self):
        exc = ApiException(status=503, reason="synthetic")
        exc.body = "Service Unavailable"
        result = translate_api_error(exc, "op failed")
        self.assertEqual(result.kind, "unavailable")
        self.assertEqual(result.detail, "Service Unavailable")

    def test_field_errors_are_carried(self):
        field_errors = [
            {
                "field": "body.items.0.vector",
                "message": "Field required",
                "type": "missing",
            }
        ]
        result = translate_api_error(
            _api_exception(422, kind="invalid_argument", errors=field_errors),
            "op failed",
        )
        self.assertEqual(result.errors, field_errors)

    def test_field_errors_absent_by_default(self):
        result = translate_api_error(_api_exception(404, kind="not_found"), "op")
        self.assertIsNone(result.errors)

    def test_field_errors_none_when_no_usable_entries(self):
        for errors in ([], ["something"]):
            with self.subTest(errors=errors):
                result = translate_api_error(
                    _api_exception(422, kind="invalid_argument", errors=errors),
                    "op failed",
                )
                self.assertIsNone(result.errors)

    def test_errors_raised_by_the_sdk_have_kinds(self):
        self.assertEqual(ValidationError("bad").kind, "invalid_argument")
        self.assertEqual(_ArgumentTypeError("bad").kind, "invalid_argument")
        result = translate_api_error(
            urllib3.exceptions.TimeoutError("timed out"), "op failed"
        )
        self.assertEqual(result.kind, "transport")
        self.assertIsNone(result.errors)


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
