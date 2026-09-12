"""Unit tests for the cyborgdb exception translation layer."""

import unittest
from unittest.mock import MagicMock
import urllib3.exceptions

import cyborgdb
from cyborgdb.exceptions import (
    AuthenticationError,
    ConnectionTimeoutError,
    CyborgError,
    ServiceUnavailableError,
)
from cyborgdb.openapi_client.exceptions import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
    ServiceException,
    UnauthorizedException,
)


def _make_client():
    from cyborgdb import Client

    client = Client.__new__(Client)
    client.api = MagicMock()
    client.api_client = MagicMock()
    client.config = MagicMock()
    client.config.api_key = {}
    return client


def _make_index():
    from cyborgdb.client.encrypted_index import EncryptedIndex

    idx = EncryptedIndex.__new__(EncryptedIndex)
    idx._api = MagicMock()
    idx._api_client = MagicMock()
    idx._index_name = "test"
    idx._index_key_hex = None
    return idx


class TestImports(unittest.TestCase):
    """Case 10: import paths and top-level re-exports."""

    def test_import_from_exceptions_module(self):
        from cyborgdb.exceptions import (
            AuthenticationError,
            ConnectionTimeoutError,
            CyborgError,
            ServiceUnavailableError,
        )

        self.assertTrue(issubclass(AuthenticationError, CyborgError))
        self.assertTrue(issubclass(ServiceUnavailableError, CyborgError))
        self.assertTrue(issubclass(ConnectionTimeoutError, CyborgError))

    def test_top_level_re_exports(self):
        self.assertIs(cyborgdb.AuthenticationError, AuthenticationError)
        self.assertIs(cyborgdb.ServiceUnavailableError, ServiceUnavailableError)
        self.assertIs(cyborgdb.ConnectionTimeoutError, ConnectionTimeoutError)
        self.assertIs(cyborgdb.CyborgError, CyborgError)


class TestClientListIndexes(unittest.TestCase):
    """Exception translation for Client.list_indexes."""

    def setUp(self):
        self.client = _make_client()

    def test_unauthorized_raises_authentication_error(self):
        """Case 1: UnauthorizedException(401) -> AuthenticationError."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = (
            UnauthorizedException(status=401)
        )
        with self.assertRaises(AuthenticationError) as ctx:
            self.client.list_indexes()
        self.client.api.list_indexes_v1_indexes_list_get.assert_called_once()
        self.assertIsInstance(ctx.exception, CyborgError)
        self.assertIsInstance(ctx.exception.__cause__, UnauthorizedException)

    def test_forbidden_raises_authentication_error(self):
        """Case 2: ForbiddenException(403) -> AuthenticationError."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = (
            ForbiddenException(status=403)
        )
        with self.assertRaises(AuthenticationError) as ctx:
            self.client.list_indexes()
        self.assertIsInstance(ctx.exception.__cause__, ForbiddenException)

    def test_service_exception_503_raises_service_unavailable(self):
        """Case 3: ServiceException(503) -> ServiceUnavailableError."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = ServiceException(
            status=503
        )
        with self.assertRaises(ServiceUnavailableError) as ctx:
            self.client.list_indexes()
        self.assertIsInstance(ctx.exception, CyborgError)

    def test_service_exception_504_raises_service_unavailable(self):
        """Case 4: ServiceException(504) -> ServiceUnavailableError."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = ServiceException(
            status=504
        )
        with self.assertRaises(ServiceUnavailableError):
            self.client.list_indexes()

    def test_service_exception_500_raises_service_unavailable(self):
        """Case 5: ServiceException(500) -> ServiceUnavailableError."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = ServiceException(
            status=500
        )
        with self.assertRaises(ServiceUnavailableError):
            self.client.list_indexes()

    def test_timeout_raises_connection_timeout_error(self):
        """Case 6: urllib3 TimeoutError -> ConnectionTimeoutError."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = (
            urllib3.exceptions.TimeoutError()
        )
        with self.assertRaises(ConnectionTimeoutError) as ctx:
            self.client.list_indexes()
        self.assertIsInstance(ctx.exception, CyborgError)
        self.assertIsInstance(ctx.exception.__cause__, urllib3.exceptions.TimeoutError)

    def test_max_retry_raises_service_unavailable(self):
        """Case 7: urllib3 MaxRetryError -> ServiceUnavailableError."""
        pool = MagicMock()
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = (
            urllib3.exceptions.MaxRetryError(pool, "/")
        )
        with self.assertRaises(ServiceUnavailableError):
            self.client.list_indexes()

    def test_bad_request_raises_value_error(self):
        """Case 8: BadRequestException(400) -> ValueError (unchanged)."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = (
            BadRequestException(status=400)
        )
        with self.assertRaises(ValueError) as ctx:
            self.client.list_indexes()
        self.assertNotIsInstance(ctx.exception, CyborgError)

    def test_not_found_raises_value_error(self):
        """Case 9: NotFoundException(404) -> ValueError (unchanged)."""
        self.client.api.list_indexes_v1_indexes_list_get.side_effect = (
            NotFoundException(status=404)
        )
        with self.assertRaises(ValueError):
            self.client.list_indexes()


class TestEncryptedIndexQuery(unittest.TestCase):
    """Exception translation for EncryptedIndex.query (outer block)."""

    def setUp(self):
        self.idx = _make_index()
        self.idx._request_headers = MagicMock(return_value={})
        self.idx._key_to_hex = MagicMock(return_value=None)

    def _call_query(self):
        # Use a flat list so query() goes through the REST path
        # (numpy arrays are routed to query_binary internally)
        self.idx.query(query_vectors=[1.0, 2.0, 3.0])

    def test_unauthorized_raises_authentication_error(self):
        self.idx._api.query_vectors_v1_vectors_query_post_without_preload_content.side_effect = UnauthorizedException(
            status=401
        )
        with self.assertRaises(AuthenticationError) as ctx:
            self._call_query()
        self.assertIsInstance(ctx.exception.__cause__, UnauthorizedException)

    def test_service_exception_raises_service_unavailable(self):
        self.idx._api.query_vectors_v1_vectors_query_post_without_preload_content.side_effect = ServiceException(
            status=503
        )
        with self.assertRaises(ServiceUnavailableError):
            self._call_query()

    def test_timeout_raises_connection_timeout(self):
        self.idx._api.query_vectors_v1_vectors_query_post_without_preload_content.side_effect = urllib3.exceptions.TimeoutError()
        with self.assertRaises(ConnectionTimeoutError) as ctx:
            self._call_query()
        self.assertIsInstance(ctx.exception.__cause__, urllib3.exceptions.TimeoutError)

    def test_new_connection_error_raises_service_unavailable(self):
        conn = MagicMock()
        self.idx._api.query_vectors_v1_vectors_query_post_without_preload_content.side_effect = urllib3.exceptions.NewConnectionError(
            conn, "refused"
        )
        with self.assertRaises(ServiceUnavailableError):
            self._call_query()

    def test_cause_chain_preserved(self):
        """Case 11: __cause__ on translated exception is the originating exception."""
        cause = ServiceException(status=503)
        self.idx._api.query_vectors_v1_vectors_query_post_without_preload_content.side_effect = cause
        with self.assertRaises(ServiceUnavailableError) as ctx:
            self._call_query()
        self.assertIs(ctx.exception.__cause__, cause)


class TestEncryptedIndexUpsert(unittest.TestCase):
    """Exception translation for EncryptedIndex.upsert (dict path)."""

    def setUp(self):
        self.idx = _make_index()
        self.idx._request_headers = MagicMock(return_value={})
        self.idx._key_to_hex = MagicMock(return_value=None)

    def _call_upsert(self):
        self.idx.upsert([{"id": "a", "vector": [0.1, 0.2]}])

    def test_unauthorized_raises_authentication_error(self):
        self.idx._api.upsert_vectors_v1_vectors_upsert_post.side_effect = (
            UnauthorizedException(status=401)
        )
        with self.assertRaises(AuthenticationError):
            self._call_upsert()

    def test_service_exception_raises_service_unavailable(self):
        self.idx._api.upsert_vectors_v1_vectors_upsert_post.side_effect = (
            ServiceException(status=500)
        )
        with self.assertRaises(ServiceUnavailableError):
            self._call_upsert()

    def test_timeout_raises_connection_timeout(self):
        self.idx._api.upsert_vectors_v1_vectors_upsert_post.side_effect = (
            urllib3.exceptions.TimeoutError()
        )
        with self.assertRaises(ConnectionTimeoutError):
            self._call_upsert()

    def test_cause_chain_preserved(self):
        """Case 11: __cause__ on translated exception."""
        cause = UnauthorizedException(status=401)
        self.idx._api.upsert_vectors_v1_vectors_upsert_post.side_effect = cause
        with self.assertRaises(AuthenticationError) as ctx:
            self._call_upsert()
        self.assertIs(ctx.exception.__cause__, cause)


if __name__ == "__main__":
    unittest.main()
