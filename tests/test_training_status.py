"""Offline tests for Client.get_training_status and its async twin."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from cyborgdb import AsyncClient, Client
from cyborgdb.exceptions import ServiceError
from cyborgdb.openapi_client.exceptions import ApiException
from cyborgdb.openapi_client.models import IndexTrainingStatusResponseModel
from cyborgdb.openapi_client_async.exceptions import ApiException as AsyncApiException

STATUS = IndexTrainingStatusResponseModel(
    training_indexes=["a", "b"],
    retrain_threshold=65536,
    currently_training="a",
    queued_indexes=["b"],
    worker_running=True,
    worker_pid=1234,
)


def _server_error(exception_class):
    exc = exception_class(status=500, reason="Internal Server Error")
    exc.body = json.dumps({"detail": "boom"})
    exc.headers = {}
    return exc


class TestGetTrainingStatus(unittest.TestCase):
    def test_returns_only_the_documented_fields(self):
        client = Client("http://localhost:8000")
        client.api = MagicMock()
        client.api.get_training_status_v1_indexes_training_status_get.return_value = (
            STATUS
        )
        self.assertEqual(
            client.get_training_status(),
            {"training_indexes": ["a", "b"], "retrain_threshold": 65536},
        )

    def test_errors_are_typed(self):
        client = Client("http://localhost:8000")
        client.api = MagicMock()
        client.api.get_training_status_v1_indexes_training_status_get.side_effect = (
            _server_error(ApiException)
        )
        with self.assertRaises(ServiceError):
            client.get_training_status()

    def test_async_matches_the_sync_client(self):
        async def run():
            client = AsyncClient("http://localhost:8000")
            client.api = MagicMock()
            client.api.get_training_status_v1_indexes_training_status_get = AsyncMock(
                return_value=STATUS
            )
            try:
                return await client.get_training_status()
            finally:
                await client.close()

        self.assertEqual(
            asyncio.run(run()),
            {"training_indexes": ["a", "b"], "retrain_threshold": 65536},
        )

    def test_async_errors_are_typed(self):
        async def run():
            client = AsyncClient("http://localhost:8000")
            client.api = MagicMock()
            client.api.get_training_status_v1_indexes_training_status_get = AsyncMock(
                side_effect=_server_error(AsyncApiException)
            )
            try:
                await client.get_training_status()
            finally:
                await client.close()

        with self.assertRaises(ServiceError):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
