"""Offline tests for EncryptedIndex.describe / get_num_vectors and their async twins."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from cyborgdb import AsyncEncryptedIndex, EncryptedIndex
from cyborgdb.exceptions import NotFoundError
from cyborgdb.openapi_client.exceptions import ApiException
from cyborgdb.openapi_client.models import (
    BM25Config,
    IndexInfoResponseModel,
    MetadataFieldPolicy,
    NumVectorsResponseModel,
)
from cyborgdb.openapi_client_async.exceptions import ApiException as AsyncApiException

KEY = b"\x01" * 32
KEY_HEX = KEY.hex()


def _describe_response(**overrides):
    fields = {
        "index_name": "idx",
        "is_trained": False,
        "index_type": "disk_ivf",
        "training_state": "untrained",
        "storage_precision": "float16",
        "dimension": 8,
        "metric": "cosine",
        "n_lists": 1,
    }
    fields.update(overrides)
    return IndexInfoResponseModel(**fields)


def _not_found(exception_class):
    exc = exception_class(status=404, reason="Not Found")
    exc.body = json.dumps({"detail": "index not found"})
    exc.headers = {}
    return exc


def _sync_index(api):
    api_client = MagicMock()
    api_client.configuration.api_key = {}
    return EncryptedIndex("idx", KEY, api, api_client)


def _async_index(api):
    api_client = MagicMock()
    api_client.configuration.api_key = {}
    return AsyncEncryptedIndex("idx", KEY, api, api_client)


class TestDescribe(unittest.TestCase):
    def test_returns_every_field_as_plain_values(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.return_value = _describe_response(
            metadata_schema={
                "title": MetadataFieldPolicy(filterable=True, pattern=True),
                "body": MetadataFieldPolicy(filterable=False, full_text=True),
            },
            bm25=BM25Config(k1=1.2, b=0.75, analyzer_version="v1"),
            embedding_model="all-MiniLM-L6-v2",
        )
        self.assertEqual(
            _sync_index(api).describe(),
            {
                "index_name": "idx",
                "index_type": "disk_ivf",
                "dimension": 8,
                "metric": "cosine",
                "storage_precision": "float16",
                "n_lists": 1,
                "training_state": "untrained",
                "metadata_schema": {
                    "title": {"filterable": True, "pattern": True, "full_text": False},
                    "body": {"filterable": False, "pattern": False, "full_text": True},
                },
                "bm25": {"k1": 1.2, "b": 0.75, "analyzer_version": "v1"},
                "embedding_model": "all-MiniLM-L6-v2",
            },
        )
        request = api.get_index_info_v1_indexes_describe_post.call_args.kwargs
        self.assertEqual(request["index_operation_request"].index_name, "idx")
        self.assertEqual(request["index_operation_request"].index_key, KEY_HEX)

    def test_defaults_when_optional_fields_are_absent(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.return_value = _describe_response()
        described = _sync_index(api).describe()
        self.assertEqual(described["metadata_schema"], {})
        self.assertIsNone(described["bm25"])
        self.assertNotIn("embedding_model", described)

    def test_training_state_is_passed_through(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.return_value = _describe_response(
            training_state="training", is_trained=False
        )
        self.assertEqual(_sync_index(api).describe()["training_state"], "training")

    def test_refreshes_the_cached_getters(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.return_value = _describe_response()
        index = _sync_index(api)
        index.describe()
        self.assertEqual((index.dimension, index.metric), (8, "cosine"))
        self.assertEqual(api.get_index_info_v1_indexes_describe_post.call_count, 1)

    def test_errors_are_typed(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post.side_effect = _not_found(
            ApiException
        )
        with self.assertRaises(NotFoundError):
            _sync_index(api).describe()


class TestGetNumVectors(unittest.TestCase):
    def test_returns_the_count(self):
        api = MagicMock()
        api.get_index_size_v1_vectors_num_vectors_post.return_value = (
            NumVectorsResponseModel(result=42)
        )
        self.assertEqual(_sync_index(api).get_num_vectors(), 42)
        request = api.get_index_size_v1_vectors_num_vectors_post.call_args.kwargs
        self.assertEqual(request["index_operation_request"].index_name, "idx")
        self.assertEqual(request["index_operation_request"].index_key, KEY_HEX)

    def test_errors_are_typed(self):
        api = MagicMock()
        api.get_index_size_v1_vectors_num_vectors_post.side_effect = _not_found(
            ApiException
        )
        with self.assertRaises(NotFoundError):
            _sync_index(api).get_num_vectors()


class TestAsync(unittest.TestCase):
    def test_describe_matches_the_sync_client(self):
        response = _describe_response(embedding_model="all-MiniLM-L6-v2")
        sync_api, async_api = MagicMock(), MagicMock()
        sync_api.get_index_info_v1_indexes_describe_post.return_value = response
        async_api.get_index_info_v1_indexes_describe_post = AsyncMock(
            return_value=response
        )
        self.assertEqual(
            asyncio.run(_async_index(async_api).describe()),
            _sync_index(sync_api).describe(),
        )

    def test_get_num_vectors(self):
        api = MagicMock()
        api.get_index_size_v1_vectors_num_vectors_post = AsyncMock(
            return_value=NumVectorsResponseModel(result=7)
        )
        self.assertEqual(asyncio.run(_async_index(api).get_num_vectors()), 7)

    def test_errors_are_typed(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post = AsyncMock(
            side_effect=_not_found(AsyncApiException)
        )
        api.get_index_size_v1_vectors_num_vectors_post = AsyncMock(
            side_effect=_not_found(AsyncApiException)
        )
        index = _async_index(api)
        with self.assertRaises(NotFoundError):
            asyncio.run(index.describe())
        with self.assertRaises(NotFoundError):
            asyncio.run(index.get_num_vectors())


if __name__ == "__main__":
    unittest.main()
