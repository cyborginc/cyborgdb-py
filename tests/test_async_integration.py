"""Integration tests for AsyncClient and AsyncEncryptedIndex.

All tests are skipped unless CYBORGDB_SERVICE_URL is set.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid

import numpy as np
import pytest

SERVICE_URL = os.getenv("CYBORGDB_SERVICE_URL")
API_KEY = os.getenv("CYBORGDB_API_KEY")

pytestmark = pytest.mark.skipif(
    not SERVICE_URL,
    reason="CYBORGDB_SERVICE_URL not set — skipping integration tests",
)

DIM = 8


def _key():
    from cyborgdb import AsyncClient

    return AsyncClient.generate_key()


def _unique_name(prefix="async-test"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 1. Basic async round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_async_round_trip():
    from cyborgdb import AsyncClient

    index_name = _unique_name()
    key = _key()

    async with AsyncClient(base_url=SERVICE_URL, api_key=API_KEY) as client:
        index = await client.create_index(
            index_name=index_name, index_key=key, dimension=DIM
        )

        items = [
            {"id": f"v{i}", "vector": np.random.rand(DIM).tolist()} for i in range(5)
        ]
        await index.upsert(items)

        results = await index.query(query_vectors=items[0]["vector"], top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

        await index.delete_index()


# ---------------------------------------------------------------------------
# 2. Descriptor methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_descriptor_methods():
    from cyborgdb import AsyncClient

    index_name = _unique_name()
    key = _key()

    async with AsyncClient(base_url=SERVICE_URL, api_key=API_KEY) as client:
        index = await client.create_index(
            index_name=index_name, index_key=key, dimension=DIM
        )

        dim = await index.dimension()
        assert dim == DIM

        metric = await index.metric()
        assert isinstance(metric, str)

        n = await index.n_lists()
        assert isinstance(n, int)

        await index.delete_index()


# ---------------------------------------------------------------------------
# 3. async with close — pool closed after exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_with_close():
    from cyborgdb import AsyncClient

    async with AsyncClient(base_url=SERVICE_URL, api_key=API_KEY) as client:
        assert client._async_api_client is not None

    assert client._async_api_client.rest_client.pool.is_closed


# ---------------------------------------------------------------------------
# 4. Concurrency scaling — N=64 concurrent queries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_scaling():
    from cyborgdb import AsyncClient

    index_name = _unique_name()
    key = _key()
    N = 64

    async with AsyncClient(base_url=SERVICE_URL, api_key=API_KEY) as client:
        index = await client.create_index(
            index_name=index_name, index_key=key, dimension=DIM
        )

        vectors = np.random.rand(20, DIM).astype(np.float32)
        ids = [f"v{i}" for i in range(20)]
        await index.upsert(ids, vectors)

        query_vec = np.random.rand(DIM).astype(np.float32)

        start = time.monotonic()
        tasks = [
            asyncio.create_task(index.query(query_vectors=query_vec, top_k=5))
            for _ in range(N)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - start

        assert all(isinstance(r, list) for r in results)

        serial_start = time.monotonic()
        await index.query(query_vectors=query_vec, top_k=5)
        serial_time = time.monotonic() - serial_start

        pool_size = 100
        expected_max = 2 * serial_time * math.ceil(N / pool_size) + 2.0
        assert wall < expected_max, (
            f"Wall time {wall:.2f}s exceeds expected max {expected_max:.2f}s — "
            "queries may be blocked by thread pool"
        )

        await index.delete_index()


# ---------------------------------------------------------------------------
# 5. Cancellation raises asyncio.CancelledError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_raises_cancelled_error():
    from cyborgdb import AsyncClient

    index_name = _unique_name()
    key = _key()

    async with AsyncClient(base_url=SERVICE_URL, api_key=API_KEY) as client:
        index = await client.create_index(
            index_name=index_name, index_key=key, dimension=DIM
        )

        query_vec = np.random.rand(DIM).astype(np.float32)
        task = asyncio.create_task(index.query(query_vectors=query_vec, top_k=5))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await index.delete_index()


# ---------------------------------------------------------------------------
# 6. Error taxonomy — wrong API key → AuthenticationError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authentication_error_taxonomy():
    from cyborgdb import AsyncClient
    from cyborgdb.exceptions import AuthenticationError

    async with AsyncClient(base_url=SERVICE_URL, api_key="wrong-key") as client:
        with pytest.raises(AuthenticationError):
            await client.list_indexes()
