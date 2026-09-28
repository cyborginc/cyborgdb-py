"""Offline unit tests for EncryptedIndex / Client wrapper logic (no service)."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from cyborgdb.client.client import Client
from cyborgdb.client.encrypted_index import EncryptedIndex
from cyborgdb.exceptions import CyborgDBError, NotFoundError, ValidationError
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


if __name__ == "__main__":
    unittest.main()
