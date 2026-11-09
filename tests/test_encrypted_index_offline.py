"""Offline unit tests for EncryptedIndex / Client wrapper logic (no service)."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import urllib3.exceptions

from cyborgdb.client.client import Client
from cyborgdb.client.encrypted_index import EncryptedIndex
from cyborgdb.exceptions import (
    AuthenticationError,
    CyborgDBError,
    NotFoundError,
    TransportError,
    ValidationError,
)
from cyborgdb.openapi_client.exceptions import ApiException

KEY = b"\x01" * 32


def _index(api=None):
    api_client = MagicMock()
    api_client.configuration.api_key = {}
    return EncryptedIndex("idx", KEY, api or MagicMock(), api_client)


def _describe_response(dimension=8, metric="euclidean"):
    return SimpleNamespace(
        dimension=dimension,
        metric=metric,
        n_lists=1,
        metadata_schema=None,
        bm25=None,
    )


def _not_found():
    exc = ApiException(status=404, reason="Not Found")
    exc.body = json.dumps({"detail": "index not found"})
    exc.headers = {}
    return exc


class TestUpsertBinaryContents(unittest.TestCase):
    def _sent_batch(self, contents):
        api = MagicMock()
        index = _index(api)
        index.upsert_binary(
            ["a", "b", "c"], np.zeros((3, 2), dtype=np.float32), contents=contents
        )
        request = api.upsert_vectors_binary_v1_vectors_upsert_binary_post.call_args
        return request.kwargs["binary_upsert_request"].batch.to_dict()

    def test_str_and_bytes_contents_are_accepted(self):
        batch = self._sent_batch(["hello", b"\x00\x01", bytearray(b"hi")])
        self.assertEqual(batch["contents"], ["hello", "AAE=", "aGk="])

    def test_none_contents_keep_their_position(self):
        batch = self._sent_batch([None, "second", None])
        self.assertEqual(batch["contents"], ["", "second", ""])

    def test_length_mismatch_is_a_validation_error(self):
        index = _index()
        vectors = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValidationError):
            index.upsert_binary(["a", "b"], vectors, contents=["only one"])
        with self.assertRaises(ValidationError):
            index.upsert_binary(["a", "b"], vectors, metadata=[{}])

    def test_wrong_vectors_type_is_validation_and_type_error(self):
        with self.assertRaises(ValidationError) as ctx:
            _index().upsert_binary(["a"], [[0.0, 1.0]])
        self.assertIsInstance(ctx.exception, TypeError)

    def test_zero_dimension_vectors_are_a_validation_error(self):
        api = MagicMock()
        with self.assertRaises(ValidationError) as ctx:
            _index(api).upsert_binary(["a"], np.zeros((1, 0), dtype=np.float32))
        self.assertEqual(ctx.exception.kind, "invalid_argument")
        self.assertEqual(api.method_calls, [])


class TestDimensionCache(unittest.TestCase):
    def test_zero_dimension_is_not_cached(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.side_effect = [
            _describe_response(dimension=0),
            _describe_response(dimension=8),
            _describe_response(dimension=8),
        ]
        index = _index(api)
        self.assertEqual(index.dimension, 0)
        self.assertEqual(index.dimension, 8)
        self.assertEqual(index.dimension, 8)
        self.assertEqual(api.get_index_info_v1_indexes_describe_post.call_count, 2)

    def test_load_index_on_empty_auto_dimension_index(self):
        client = Client("http://api.example.com", api_key="k")
        client.api = MagicMock()
        client.api.get_index_info_v1_indexes_describe_post.side_effect = [
            _describe_response(dimension=0),
            _describe_response(dimension=16),
        ]
        index = client.load_index("idx", KEY)
        self.assertEqual(index.dimension, 16)


class TestGetterErrors(unittest.TestCase):
    def test_getters_raise_typed_errors(self):
        for name in ("dimension", "metric", "n_lists", "metadata_schema", "bm25"):
            with self.subTest(getter=name):
                api = MagicMock()
                api.get_index_info_v1_indexes_describe_post.side_effect = _not_found()
                with self.assertRaises(NotFoundError):
                    getattr(_index(api), name)


class TestIsTrained(unittest.TestCase):
    def test_errors_propagate_instead_of_returning_false(self):
        unauthorized = ApiException(status=401, reason="Unauthorized")
        unauthorized.headers = {}
        for exc, expected in (
            (unauthorized, AuthenticationError),
            (urllib3.exceptions.NewConnectionError(None, "refused"), TransportError),
        ):
            with self.subTest(expected=expected.__name__):
                api = MagicMock()
                api.get_index_info_v1_indexes_describe_post.side_effect = exc
                with self.assertRaises(expected):
                    _index(api).is_trained()

    def test_returns_the_reported_status(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.return_value = SimpleNamespace(
            dimension=8, metric="euclidean", is_trained=True
        )
        self.assertIs(_index(api).is_trained(), True)


class TestLoadIndexErrors(unittest.TestCase):
    def test_error_message_names_the_index(self):
        client = Client("http://api.example.com", api_key="k")
        client.api = MagicMock()
        client.api.get_index_info_v1_indexes_describe_post.side_effect = _not_found()
        with self.assertRaises(NotFoundError) as ctx:
            client.load_index("my-index", KEY)
        self.assertIn("'my-index'", str(ctx.exception))
        self.assertNotIn("{index_name}", str(ctx.exception))

    def test_bad_key_is_a_validation_error(self):
        client = Client("http://api.example.com", api_key="k")
        with self.assertRaises(ValidationError):
            client.load_index("idx", b"short")
        with self.assertRaises(ValidationError):
            client.create_index("idx")


class TestTrain(unittest.TestCase):
    def test_max_memory_is_forwarded(self):
        api = MagicMock()
        _index(api).train(max_memory=512)
        request = api.train_index_v1_indexes_train_post.call_args.kwargs[
            "train_request"
        ]
        self.assertEqual(request.max_memory, 512)


class TestArgumentErrorsAreTyped(unittest.TestCase):
    def test_upsert_argument_errors(self):
        index = _index()
        with self.assertRaises(ValidationError):
            index.upsert([{"vector": [0.0]}])
        with self.assertRaises(TypeError):
            index.upsert("not a list")
        with self.assertRaises(CyborgDBError):
            index.upsert(["a"], [[0.0], [1.0]])

    def test_query_argument_errors(self):
        index = _index()
        with self.assertRaises(ValidationError):
            index.query(query_vectors=[])
        with self.assertRaises(ValidationError):
            index.query(query_vectors=np.zeros((1, 1, 1)))
        with self.assertRaises(ValidationError):
            index.query_metadata(order_by={"a": 1, "b": -1})


class TestQueryInclude(unittest.TestCase):
    def _query(self, results, include):
        api = MagicMock()
        raw = api.query_vectors_v1_vectors_query_post_without_preload_content
        raw.return_value = SimpleNamespace(
            status=200, data=json.dumps({"results": results}).encode("utf-8")
        )
        return _index(api).query(query_vectors=[0.0, 1.0], top_k=1, include=include)

    def test_distance_dropped_when_not_included(self):
        results = [{"id": "a", "distance": 0.1, "metadata": {"k": 1}}]
        self.assertEqual(self._query(results, []), [{"id": "a"}])
        self.assertEqual(
            self._query(results, ["metadata"]), [{"id": "a", "metadata": {"k": 1}}]
        )

    def test_distance_kept_when_included_or_include_omitted(self):
        results = [{"id": "a", "distance": 0.1}]
        expected = [{"id": "a", "distance": 0.1}]
        self.assertEqual(self._query(results, ["distance"]), expected)
        self.assertEqual(self._query(results, None), expected)

    def test_hybrid_score_ignores_include(self):
        results = [{"id": "a", "score": 0.7}]
        self.assertEqual(self._query(results, []), [{"id": "a", "score": 0.7}])


class TestIncludeValidation(unittest.TestCase):
    """A bad ``include`` raises ``ValidationError`` with the service's wording
    before any request is sent."""

    def test_query_rejects_unsupported_values(self):
        api = MagicMock()
        index = _index(api)
        for include in (["contents"], ["vector"], ["score"], ["distance", "bananas"]):
            for call in (
                lambda: index.query(query_vectors=[0.0, 1.0], include=include),
                lambda: index.query(query_vectors=[[0.0, 1.0]], include=include),
                lambda: index.query_binary(np.zeros(2, np.float32), include=include),
            ):
                with self.subTest(include=include):
                    with self.assertRaisesRegex(
                        ValidationError,
                        r"Valid values: 'distance', 'metadata'\. .*index\.get\(",
                    ):
                        call()
        self.assertEqual(api.method_calls, [])

    def test_get_rejects_unsupported_values(self):
        api = MagicMock()
        index = _index(api)
        for include in (["distance"], ["score"], ["Contents"]):
            with self.subTest(include=include):
                with self.assertRaisesRegex(
                    ValidationError, r"Valid values: 'vector', 'contents', 'metadata'"
                ):
                    index.get(["a"], include=include)
        self.assertEqual(api.method_calls, [])

    def test_bare_string_rejected(self):
        api = MagicMock()
        index = _index(api)
        with self.assertRaisesRegex(ValidationError, "not a single string"):
            index.get(["a"], include="metadata")
        with self.assertRaisesRegex(ValidationError, "not a single string"):
            index.query(query_vectors=[0.0, 1.0], include="distance")
        self.assertEqual(api.method_calls, [])


class TestPydanticRejectionsAreTranslated(unittest.TestCase):
    """Malformed arguments the generated request models reject must surface as
    the SDK's ValidationError (kind invalid_argument), never as bare pydantic
    errors a caller would classify as internal."""

    def assert_rejected_before_sending(self, api, fn):
        with self.assertRaises(ValidationError) as ctx:
            fn()
        self.assertEqual(ctx.exception.kind, "invalid_argument")
        self.assertIsNone(ctx.exception.status_code)
        self.assertEqual(api.method_calls, [])
        self.assertIsNone(ctx.exception.__cause__)
        self.assertIsNone(ctx.exception.__context__)

    def test_upsert_non_string_id(self):
        api = MagicMock()
        self.assert_rejected_before_sending(
            api, lambda: _index(api).upsert([{"id": 123, "vector": [0.1]}])
        )

    def test_upsert_contents_of_wrong_type(self):
        api = MagicMock()
        self.assert_rejected_before_sending(
            api,
            lambda: _index(api).upsert([{"id": "a", "vector": [0.1], "contents": 123}]),
        )

    def test_query_non_object_filter(self):
        api = MagicMock()
        self.assert_rejected_before_sending(
            api, lambda: _index(api).query(query_vectors=[0.1, 0.2], filters="nope")
        )

    def test_query_binary_non_object_filter(self):
        api = MagicMock()
        self.assert_rejected_before_sending(
            api,
            lambda: _index(api).query(
                query_vectors=np.zeros((1, 2), dtype=np.float32), filters="nope"
            ),
        )

    def test_query_without_vectors_or_contents(self):
        api = MagicMock()
        self.assert_rejected_before_sending(api, lambda: _index(api).query())

    def test_query_metadata_non_object_filter(self):
        api = MagicMock()
        self.assert_rejected_before_sending(
            api, lambda: _index(api).query_metadata(filters="nope", text="hello")
        )

    def test_create_index_non_object_schema(self):
        client = Client("http://api.example.com", api_key="k")
        client.api = MagicMock()
        self.assert_rejected_before_sending(
            client.api,
            lambda: client.create_index("idx", KEY, metadata_schema="not a dict"),
        )

    def test_create_index_non_finite_bm25_params(self):
        # nan/inf can't travel as JSON: httpx (async) raises a bare ValueError
        # serializing them, so they must be rejected client-side on both paths.
        client = Client("http://api.example.com", api_key="k")
        client.api = MagicMock()
        for kwargs in (
            {"bm25_k1": float("nan")},
            {"bm25_k1": float("inf")},
            {"bm25_b": float("nan")},
            {"bm25_b": float("-inf")},
        ):
            with self.subTest(**kwargs):
                self.assert_rejected_before_sending(
                    client.api, lambda: client.create_index("idx", KEY, **kwargs)
                )

    def test_field_errors_point_at_the_argument(self):
        with self.assertRaises(ValidationError) as ctx:
            _index().query(query_vectors=[0.1, 0.2], filters="nope")
        self.assertEqual(ctx.exception.errors[0]["field"], "filters")

    def test_rejected_value_stays_out_of_the_traceback(self):
        # The raw pydantic error renders the value, so it is not chained.
        import traceback

        with self.assertRaises(ValidationError) as ctx:
            _index().query(query_vectors=[0.1, 0.2], filters="TOP-SECRET-VALUE")
        exc = ctx.exception
        self.assertIsNone(exc.__cause__)
        rendered = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self.assertNotIn("TOP-SECRET-VALUE", rendered)


if __name__ == "__main__":
    unittest.main()
