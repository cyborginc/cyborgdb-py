"""Offline tests for Client.list_embedding_models and its async twin."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from cyborgdb import AsyncClient, Client
from cyborgdb.exceptions import ServiceError
from cyborgdb.openapi_client.exceptions import ApiException
from cyborgdb.openapi_client.models import (
    EmbeddingModelInfo,
    EmbeddingModelListResponseModel,
)
from cyborgdb.openapi_client_async.exceptions import ApiException as AsyncApiException

MODELS = EmbeddingModelListResponseModel(
    models=[
        EmbeddingModelInfo(
            name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            max_seq_length=256,
        ),
        EmbeddingModelInfo(
            name="BAAI/bge-base-en-v1.5", dimension=768, max_seq_length=512
        ),
    ]
)
EXPECTED = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_seq_length": 256,
    },
    {"name": "BAAI/bge-base-en-v1.5", "dimension": 768, "max_seq_length": 512},
]


def _server_error(exception_class):
    exc = exception_class(status=500, reason="Internal Server Error")
    exc.body = json.dumps({"detail": "boom"})
    exc.headers = {}
    return exc


class TestListEmbeddingModels(unittest.TestCase):
    def test_returns_plain_dicts_in_service_order(self):
        client = Client("http://localhost:8000")
        client.api = MagicMock()
        client.api.list_embedding_models_v1_embedding_models_get.return_value = MODELS
        self.assertEqual(client.list_embedding_models(), EXPECTED)

    def test_empty_list(self):
        client = Client("http://localhost:8000")
        client.api = MagicMock()
        client.api.list_embedding_models_v1_embedding_models_get.return_value = (
            EmbeddingModelListResponseModel(models=[])
        )
        self.assertEqual(client.list_embedding_models(), [])

    def test_errors_are_typed(self):
        client = Client("http://localhost:8000")
        client.api = MagicMock()
        client.api.list_embedding_models_v1_embedding_models_get.side_effect = (
            _server_error(ApiException)
        )
        with self.assertRaises(ServiceError):
            client.list_embedding_models()

    def test_async_matches_the_sync_client(self):
        async def run():
            client = AsyncClient("http://localhost:8000")
            client.api = MagicMock()
            client.api.list_embedding_models_v1_embedding_models_get = AsyncMock(
                return_value=MODELS
            )
            try:
                return await client.list_embedding_models()
            finally:
                await client.close()

        self.assertEqual(asyncio.run(run()), EXPECTED)

    def test_async_errors_are_typed(self):
        async def run():
            client = AsyncClient("http://localhost:8000")
            client.api = MagicMock()
            client.api.list_embedding_models_v1_embedding_models_get = AsyncMock(
                side_effect=_server_error(AsyncApiException)
            )
            try:
                await client.list_embedding_models()
            finally:
                await client.close()

        with self.assertRaises(ServiceError):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
