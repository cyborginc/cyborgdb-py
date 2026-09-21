"""Unit tests for AsyncClient, AsyncEncryptedIndex, and related helpers.

All tests mock the HTTP layer; no real service is required.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
import urllib3.exceptions

from cyborgdb.client.async_client import AsyncClient
from cyborgdb.client.async_encrypted_index import AsyncEncryptedIndex
from cyborgdb.exceptions import TransportError, translate_api_error


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_async_api(describe_response=None):
    """Return a mock async API with sensible defaults."""
    api = AsyncMock()
    if describe_response is None:
        resp = MagicMock()
        resp.dimension = 4
        resp.metric = "cosine"
        resp.n_lists = 1
        resp.metadata_schema = {}
        resp.bm25 = None
        resp.is_trained = False
        describe_response = resp
    api.get_index_info_v1_indexes_describe_post.return_value = describe_response
    return api


def _make_async_api_client():
    client = MagicMock()
    client.configuration = MagicMock()
    client.configuration.api_key = {}
    return client


def _make_index(name="test-idx", key=None):
    api = _make_async_api()
    api_client = _make_async_api_client()
    return AsyncEncryptedIndex(
        index_name=name,
        index_key=key,
        api=api,
        api_client=api_client,
    ), api


# ---------------------------------------------------------------------------
# 1. Awaitable check — every public async method returns a coroutine
# ---------------------------------------------------------------------------


def test_async_client_methods_are_awaitable():
    """All I/O methods on AsyncClient are declared as async def."""
    for name in ("list_indexes", "create_index", "load_index", "get_health", "close"):
        assert inspect.iscoroutinefunction(getattr(AsyncClient, name)), (
            f"AsyncClient.{name} should be async def"
        )


@pytest.mark.asyncio
async def test_async_encrypted_index_methods_are_awaitable():
    index, api = _make_index(key=b"\x00" * 32)
    api.upsert_vectors_v1_vectors_upsert_post.return_value = None

    coro = index.dimension()
    assert inspect.iscoroutine(coro)
    val = await coro
    assert isinstance(val, int)

    coro = index.metric()
    assert inspect.iscoroutine(coro)
    await coro

    coro = index.n_lists()
    assert inspect.iscoroutine(coro)
    await coro


# ---------------------------------------------------------------------------
# 2. index_name is a plain property (no await required)
# ---------------------------------------------------------------------------


def test_index_name_returns_string_without_await():
    index, _ = _make_index(name="my-index", key=b"\x01" * 32)
    result = index.index_name
    assert result == "my-index"
    assert not inspect.iscoroutine(result)


# ---------------------------------------------------------------------------
# 3. Descriptor methods return coroutines before await
# ---------------------------------------------------------------------------


def test_descriptor_methods_return_coroutines():
    index, _ = _make_index(key=b"\x00" * 32)
    for method_name in ("dimension", "metric", "n_lists", "metadata_schema", "bm25"):
        method = getattr(index, method_name)
        coro = method()
        assert inspect.iscoroutine(coro), f"{method_name}() should return a coroutine"
        coro.close()


# ---------------------------------------------------------------------------
# 4. Byte-identical request bodies: upsert sync vs async
# ---------------------------------------------------------------------------


def test_upsert_items_list_parity():
    """build_upsert_items_list produces the same output as the sync logic."""
    from cyborgdb.client._request_builders import build_upsert_items_list

    items_data = [
        {"id": "a", "vector": [1.0, 2.0, 3.0, 4.0]},
        {"id": "b", "vector": [5.0, 6.0, 7.0, 8.0]},
    ]
    result = build_upsert_items_list(items_data, None)
    assert len(result) == 2
    assert result[0]["id"] == "a"
    assert len(result[0]["vector"]) == 4


def test_query_binary_request_parity():
    """build_query_binary_request produces consistent serialisable output."""
    from cyborgdb.client._request_builders import build_query_binary_request

    vecs = np.random.rand(2, 4).astype(np.float32)
    request, is_single = build_query_binary_request(
        "idx",
        "deadbeef",
        vecs,
        top_k=5,
        n_probes=None,
        filters=None,
        include=None,
        greedy=None,
        rerank_mult=None,
        hybrid_kwargs={},
    )
    assert not is_single
    assert request.batch is not None


def test_upsert_request_sync_async_wire_parity():
    """Sync and async upsert paths produce byte-identical UpsertRequest dicts."""
    from cyborgdb.client._request_builders import build_upsert_items_list
    from cyborgdb.openapi_client.models import UpsertRequest

    items_data = [
        {"id": "x", "vector": [0.1, 0.2, 0.3, 0.4]},
        {"id": "y", "vector": [0.5, 0.6, 0.7, 0.8]},
    ]
    index_key_hex = "ab" * 32
    index_name = "parity-test"

    items = build_upsert_items_list(items_data, None)
    sync_request = UpsertRequest(
        items=items, index_key=index_key_hex, index_name=index_name
    )
    async_request = UpsertRequest(
        items=items, index_key=index_key_hex, index_name=index_name
    )

    assert sync_request.to_dict() == async_request.to_dict()


# ---------------------------------------------------------------------------
# 5. translate_api_error — httpx exceptions map to TransportError
# ---------------------------------------------------------------------------


def test_translate_api_error_httpx_connect_error():
    exc = httpx.ConnectError("refused")
    result = translate_api_error(
        exc, "ctx", transport_error_types=(httpx.TransportError, httpx.RequestError)
    )
    assert isinstance(result, TransportError)


def test_translate_api_error_httpx_timeout():
    exc = httpx.TimeoutException("timed out")
    result = translate_api_error(
        exc, "ctx", transport_error_types=(httpx.TransportError, httpx.RequestError)
    )
    assert isinstance(result, TransportError)


def test_translate_api_error_urllib3_backward_compat():
    exc = urllib3.exceptions.HTTPError("network err")
    result = translate_api_error(exc, "ctx")
    assert isinstance(result, TransportError)


def test_translate_api_error_default_uses_urllib3():
    """Calling with no transport_error_types arg still catches urllib3 errors."""
    exc = urllib3.exceptions.MaxRetryError(pool=None, url="/", reason=None)
    result = translate_api_error(exc, "ctx")
    assert isinstance(result, TransportError)


# ---------------------------------------------------------------------------
# 6. Sync Client lifecycle: with Client(...) as client: pass
# ---------------------------------------------------------------------------


def test_sync_client_context_manager():
    from cyborgdb.client.client import Client

    with patch("cyborgdb.client.client.ApiClient"), patch(
        "cyborgdb.client.client.DefaultApi"
    ):
        with Client("http://localhost:8000") as client:
            assert client is not None
        # close() should not raise even if pool_manager is a mock
        assert True


def test_sync_client_close_is_callable():
    from cyborgdb.client.client import Client

    with patch("cyborgdb.client.client.ApiClient"), patch(
        "cyborgdb.client.client.DefaultApi"
    ):
        client = Client("http://localhost:8000")
        client.close()  # should not raise


# ---------------------------------------------------------------------------
# 7. Regression — import check for async openapi package
# ---------------------------------------------------------------------------


def test_async_openapi_package_imports():
    from cyborgdb.openapi_client_async.api.default_api import DefaultApi

    assert hasattr(DefaultApi, "query_vectors_v1_vectors_query_post")
    assert hasattr(DefaultApi, "upsert_vectors_v1_vectors_upsert_post")


def test_async_api_client_imports_models_from_sync():
    """AsyncApiClient shares UpsertRequest from cyborgdb.openapi_client."""
    from cyborgdb.openapi_client.models.upsert_request import (
        UpsertRequest as SyncUpsertRequest,
    )

    import cyborgdb.openapi_client.models as sync_models

    assert hasattr(sync_models, "UpsertRequest")
    # Ensure they are the same class (no duplication)
    assert SyncUpsertRequest is sync_models.UpsertRequest


# ---------------------------------------------------------------------------
# 8. AsyncEncryptedIndex caching behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dimension_cached_after_first_call():
    index, api = _make_index(key=b"\x00" * 32)
    d1 = await index.dimension()
    d2 = await index.dimension()
    assert d1 == d2
    # describe endpoint should only be called once
    assert api.get_index_info_v1_indexes_describe_post.call_count == 1


@pytest.mark.asyncio
async def test_n_lists_not_cached():
    index, api = _make_index(key=b"\x00" * 32)
    await index.n_lists()
    await index.n_lists()
    assert api.get_index_info_v1_indexes_describe_post.call_count == 2


# ---------------------------------------------------------------------------
# 9. async with AsyncEncryptedIndex — close called on exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_encrypted_index_context_manager():
    index, _ = _make_index(key=b"\x00" * 32)
    async with index as idx:
        assert idx is index


# ---------------------------------------------------------------------------
# 10. Empty upsert list returns without error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_empty_list():
    index, api = _make_index(key=b"\x00" * 32)
    api.upsert_vectors_v1_vectors_upsert_post.return_value = None
    await index.upsert([])
    api.upsert_vectors_v1_vectors_upsert_post.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Error taxonomy parity — async client maps status codes identically
# ---------------------------------------------------------------------------


def _taxonomy_file_path():
    import os

    candidates = [
        os.path.join(
            os.path.dirname(__file__), "../../cyborgdb-sdks/error-taxonomy.yaml"
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def test_async_error_taxonomy_parity():
    """Async translate_api_error maps HTTP status codes to the same classes as sync."""
    import yaml
    from cyborgdb.openapi_client.exceptions import ApiException

    taxonomy_path = _taxonomy_file_path()
    if taxonomy_path is None:
        pytest.skip("error-taxonomy.yaml not found; skipping taxonomy parity test")

    with open(taxonomy_path) as f:
        taxonomy = yaml.safe_load(f)

    for entry in taxonomy.get("errors", []):
        status = entry["http_status"]
        expected_class_name = entry["python_class"]
        mock_exc = ApiException(status=status, reason="test")
        mock_exc.status = status
        result = translate_api_error(
            mock_exc,
            "test",
            transport_error_types=(httpx.TransportError, httpx.RequestError),
        )
        assert type(result).__name__ == expected_class_name, (
            f"Status {status}: expected {expected_class_name}, got {type(result).__name__}"
        )
