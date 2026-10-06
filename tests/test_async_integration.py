"""Integration tests for AsyncClient / AsyncEncryptedIndex against a running
service at CYBORGDB_BASE_URL (default http://localhost:8000), like the sync
e2e suites.

Mirrors those suites: a fresh index per test, deleted on teardown, with
polling instead of fixed sleeps for the service's async ingestion.
"""

import asyncio
import os
import time
import uuid

import numpy as np
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from cyborgdb import AsyncClient, AsyncEncryptedIndex
from cyborgdb.exceptions import AuthenticationError, NotFoundError

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
DIMENSION = 128
POLL_INTERVAL = 0.2
DEFAULT_TIMEOUT = 30.0
# Stored vectors are quantized, so a vector's distance to itself is small but
# not zero. Distinct random 128-dim vectors in [0, 1) sit at ~4.6 apart.
SELF_DISTANCE_MAX = 0.1

pytestmark = pytest.mark.asyncio


async def wait_for(predicate, description, timeout=DEFAULT_TIMEOUT):
    """Poll an async predicate until it is truthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return
        await asyncio.sleep(POLL_INTERVAL)
    raise AssertionError(f"condition never held within {timeout}s: {description}")


async def wait_for_ids(index, expected_ids, timeout=DEFAULT_TIMEOUT):
    expected = set(expected_ids)

    async def visible():
        try:
            return expected <= {row["id"] for row in await index.query_metadata()}
        except Exception:
            # The index may not be queryable for a moment after creation.
            return False

    await wait_for(visible, f"ids {sorted(expected)} visible", timeout)


async def wait_until_gone(index, gone_ids, timeout=DEFAULT_TIMEOUT):
    gone = set(gone_ids)

    async def absent():
        return not ({row["id"] for row in await index.query_metadata()} & gone)

    await wait_for(absent, f"ids {sorted(gone)} gone", timeout)


def _vectors(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, DIMENSION).astype(np.float32)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(BASE_URL, api_key=API_KEY) as client:
        yield client


@pytest_asyncio.fixture
async def index(client):
    name = f"test_async_{uuid.uuid4().hex[:12]}"
    key = AsyncClient.generate_key()
    index = await client.create_index(
        name, key, dimension=DIMENSION, metric="euclidean"
    )
    yield index
    try:
        await index.delete_index()
    except Exception:
        pass


class TestRoundTrip:
    async def test_create_upsert_train_query_get_delete(self, client, index):
        ids = [f"item_{i}" for i in range(100)]
        vectors = _vectors(100)

        await index.upsert(ids, vectors)
        await wait_for_ids(index, ids)
        assert sorted(await index.list_ids()) == sorted(ids)

        # Below the service's AUTO_TRAIN_MIN_VECTORS (65536) train() is accepted
        # but is a no-op, as for the sync client: the index stays exhaustive.
        assert not await index.is_trained()
        await index.train(n_lists=4)
        assert await index.is_training() is False
        assert await index.n_lists() == 1

        results = await index.query(vectors[7], top_k=5, include=["distance"])
        assert len(results) == 5
        assert results[0]["id"] == "item_7"
        assert 0 <= results[0]["distance"] < SELF_DISTANCE_MAX

        batch = await index.query(vectors[:3], top_k=2)
        assert [r[0]["id"] for r in batch] == ["item_0", "item_1", "item_2"]

        json_results = await index.query(vectors[7].tolist(), top_k=1)
        assert json_results[0]["id"] == "item_7"

        got = await index.get(["item_7"], include=["vector"])
        assert got[0]["id"] == "item_7"
        np.testing.assert_allclose(got[0]["vector"], vectors[7], rtol=1e-6)

        await index.delete(["item_7"])
        await wait_until_gone(index, ["item_7"])
        assert "item_7" not in await index.list_ids()

        name = index.index_name
        await index.delete_index()
        assert name not in await client.list_indexes()
        with pytest.raises(NotFoundError):
            await client.load_index(name, index._index_key)

    async def test_dict_upsert_with_metadata_and_contents(self, index):
        vectors = _vectors(3, seed=1)
        await index.upsert(
            [
                {
                    "id": "doc1",
                    "vector": vectors[0],
                    "contents": "hello world",
                    "metadata": {"category": "greeting", "n": 1},
                },
                {"id": "doc2", "vector": vectors[1], "metadata": {"category": "other"}},
                {"id": "doc3", "vector": vectors[2].tolist(), "contents": b"\x00\x01"},
            ]
        )
        await wait_for_ids(index, ["doc1", "doc2", "doc3"])

        rows = await index.query_metadata({"category": "greeting"})
        assert rows == [{"id": "doc1"}]

        results = await index.query(
            vectors[0], top_k=3, filters={"category": "greeting"}, include=["metadata"]
        )
        assert [r["id"] for r in results] == ["doc1"]
        assert results[0]["metadata"]["category"] == "greeting"

        got = await index.get(["doc1"])
        assert got[0]["contents"] == "hello world"
        assert got[0]["metadata"] == {"category": "greeting", "n": 1}


class TestDescriptors:
    async def test_descriptor_methods_report_index_configuration(self, client, index):
        assert index.index_name.startswith("test_async_")
        assert await index.dimension() == DIMENSION
        assert await index.metric() == "euclidean"
        assert await index.n_lists() == 1
        assert await index.metadata_schema() == {}
        assert await index.bm25() is None
        assert await index.is_trained() is False
        assert await index.is_training() is False

        loaded = await client.load_index(index.index_name, index._index_key)
        assert isinstance(loaded, AsyncEncryptedIndex)
        assert await loaded.dimension() == DIMENSION
        assert await loaded.metric() == "euclidean"

    async def test_metadata_schema_and_bm25_round_trip(self, client):
        name = f"test_async_schema_{uuid.uuid4().hex[:12]}"
        index = await client.create_index(
            name,
            AsyncClient.generate_key(),
            dimension=DIMENSION,
            metadata_schema={
                "title": {"filterable": True, "pattern": True},
                "body": {"full_text": True},
            },
            bm25_k1=1.5,
        )
        try:
            schema = await index.metadata_schema()
            assert schema["title"] == {
                "filterable": True,
                "pattern": True,
                "full_text": False,
            }
            assert schema["body"]["full_text"] is True
            bm25 = await index.bm25()
            assert bm25["k1"] == pytest.approx(1.5)
            assert bm25["b"] == pytest.approx(0.75)
        finally:
            await index.delete_index()


class TestConcurrency:
    async def test_concurrent_queries_all_succeed(self, index):
        n = 50
        ids = [f"vec_{i}" for i in range(n)]
        vectors = _vectors(n, seed=2)
        await index.upsert(ids, vectors)
        await wait_for_ids(index, ids)

        results = await asyncio.gather(
            *(index.query(vectors[i], top_k=1, include=["distance"]) for i in range(n))
        )

        assert len(results) == n
        for i, result in enumerate(results):
            assert len(result) == 1
            assert result[0]["id"] == ids[i]
            assert 0 <= result[0]["distance"] < SELF_DISTANCE_MAX

    async def test_concurrent_upserts_land(self, index):
        batches = {f"batch{b}": [f"b{b}_{i}" for i in range(20)] for b in range(5)}
        await asyncio.gather(
            *(
                index.upsert(ids, _vectors(len(ids), seed=10 + b))
                for b, ids in enumerate(batches.values())
            )
        )
        expected = [id_ for ids in batches.values() for id_ in ids]
        await wait_for_ids(index, expected)
        assert set(await index.list_ids()) == set(expected)


class TestCancellation:
    async def test_cancelling_in_flight_call_raises_cancelled_error(self, index):
        ids = [f"c_{i}" for i in range(20)]
        vectors = _vectors(20, seed=3)
        await index.upsert(ids, vectors)
        await wait_for_ids(index, ids)

        task = asyncio.create_task(index.query(vectors[0], top_k=5))
        # Let the task run up to its first await on the socket before cancelling.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        # The shared client is still usable afterwards.
        results = await index.query(vectors[0], top_k=1)
        assert results[0]["id"] == "c_0"


class TestAuthentication:
    @pytest.mark.skipif(
        not os.getenv("CYBORGDB_SERVICE_ROOT_KEY"),
        reason="auth disabled (no CYBORGDB_SERVICE_ROOT_KEY) — the service accepts any key",
    )
    async def test_wrong_api_key_raises_authentication_error(self):
        async with AsyncClient(BASE_URL, api_key="WRONG_KEY") as client:
            with pytest.raises(AuthenticationError) as ctx:
                await client.list_indexes()
        assert ctx.value.status_code in (401, 403)
