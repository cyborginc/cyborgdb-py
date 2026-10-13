"""Offline unit tests for AsyncClient / AsyncEncryptedIndex (no service).

The transport is stubbed at the HTTP-library boundary in both clients:
urllib3's pool manager for the sync client, an httpx MockTransport for the
async one. Everything above that line — the generated serializers, the shared
request builders, status handling and error translation — runs for real.
"""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest

from cyborgdb import AsyncClient, AsyncEncryptedIndex, Client, EncryptedIndex
from cyborgdb.exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ServiceError,
    TransportError,
    ValidationError,
    translate_api_error,
)
from cyborgdb.openapi_client.exceptions import ApiException as SyncApiException
from cyborgdb.openapi_client_async.exceptions import (
    ApiException as AsyncApiException,
)

KEY = b"\x01" * 32
BASE_URL = "http://api.example.com"
API_KEY = "test-key"

CLIENT_IO_METHODS = ["create_index", "load_index", "list_indexes", "get_health"]
INDEX_IO_METHODS = [
    "upsert",
    "upsert_binary",
    "query",
    "query_binary",
    "query_metadata",
    "get",
    "train",
    "delete",
    "delete_index",
    "list_ids",
    "is_trained",
    "is_training",
    "create_user",
    "list_users",
    "delete_user",
]
DESCRIPTOR_METHODS = ["dimension", "metric", "n_lists", "metadata_schema", "bm25"]

SUCCESS_BODY = {"status": "success", "message": "ok"}
EMPTY_RESULTS_BODY = {"results": []}


def _describe_response(dimension=8, metric="euclidean"):
    return SimpleNamespace(
        dimension=dimension,
        metric=metric,
        n_lists=1,
        metadata_schema=None,
        bm25=None,
        is_trained=False,
    )


def _async_index(api=None):
    api_client = MagicMock()
    api_client.configuration.api_key = {}
    api_client.close = AsyncMock()
    return AsyncEncryptedIndex("idx", KEY, api or MagicMock(), api_client)


def _sync_index(api=None):
    api_client = MagicMock()
    api_client.configuration.api_key = {}
    return EncryptedIndex("idx", KEY, api or MagicMock(), api_client)


class _Recorder:
    """Captures what each generated client hands to its HTTP library."""

    def __init__(self):
        self.calls = []

    def sync_pool_manager(self, body_for_path):
        recorder = self

        class _PoolManager:
            def request(self, method, url, body=None, headers=None, **kwargs):
                recorder.calls.append((method, url, body.encode("utf-8"), headers))
                return SimpleNamespace(
                    status=200,
                    reason="OK",
                    data=json.dumps(body_for_path(url)).encode("utf-8"),
                    headers={"content-type": "application/json"},
                )

        return _PoolManager()

    def httpx_client(self, body_for_path):
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(
                (
                    request.method,
                    str(request.url),
                    request.content,
                    dict(request.headers),
                )
            )
            return httpx.Response(200, json=body_for_path(str(request.url)))

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _body_for_path(url):
    if url.endswith("/v1/vectors/query") or url.endswith("/v1/vectors/query_binary"):
        return EMPTY_RESULTS_BODY
    return SUCCESS_BODY


def _sync_client(recorder, body_for_path=_body_for_path):
    client = Client(BASE_URL, api_key=API_KEY)
    client.api_client.rest_client.pool_manager = recorder.sync_pool_manager(
        body_for_path
    )
    return client


def _async_client(recorder, body_for_path=_body_for_path):
    client = AsyncClient(BASE_URL, api_key=API_KEY)
    client.api_client.rest_client.pool_manager = recorder.httpx_client(body_for_path)
    return client


def _canonical(raw: bytes) -> bytes:
    """Re-encode wire bytes with one encoder, keeping key order and values.

    The two generated REST layers encode the same payload with different
    whitespace (urllib3 path: ``json.dumps`` defaults; httpx: compact
    separators), which cannot be aligned without editing generated code, so
    the comparison normalizes whitespace only. Key order, numeric formatting,
    base64 payloads and the key set must still match exactly.
    """
    return json.dumps(json.loads(raw)).encode("utf-8")


class TestCoroutineSurface:
    @pytest.mark.parametrize("name", CLIENT_IO_METHODS)
    def test_client_io_methods_are_coroutine_functions(self, name):
        assert inspect.iscoroutinefunction(getattr(AsyncClient, name))

    @pytest.mark.parametrize("name", INDEX_IO_METHODS + DESCRIPTOR_METHODS)
    def test_index_io_methods_are_coroutine_functions(self, name):
        assert inspect.iscoroutinefunction(getattr(AsyncEncryptedIndex, name))

    @pytest.mark.parametrize("name", ["close", "__aenter__", "__aexit__"])
    def test_lifecycle_methods_are_coroutine_functions(self, name):
        assert inspect.iscoroutinefunction(getattr(AsyncClient, name))
        assert inspect.iscoroutinefunction(getattr(AsyncEncryptedIndex, name))

    def test_generate_key_is_a_static_non_coroutine(self):
        assert isinstance(
            inspect.getattr_static(AsyncClient, "generate_key"), staticmethod
        )
        assert not inspect.iscoroutinefunction(AsyncClient.generate_key)
        assert len(AsyncClient.generate_key()) == 32

    @pytest.mark.parametrize("name", INDEX_IO_METHODS + ["index_name"])
    def test_async_index_mirrors_sync_signatures(self, name):
        sync_attr = inspect.getattr_static(EncryptedIndex, name)
        async_attr = inspect.getattr_static(AsyncEncryptedIndex, name)
        if isinstance(sync_attr, property):
            assert isinstance(async_attr, property)
            return
        assert inspect.signature(sync_attr) == inspect.signature(async_attr)

    @pytest.mark.parametrize("name", CLIENT_IO_METHODS)
    def test_async_client_mirrors_sync_signatures(self, name):
        sync_sig = inspect.signature(getattr(Client, name))
        async_sig = inspect.signature(getattr(AsyncClient, name))
        assert list(sync_sig.parameters) == list(async_sig.parameters)
        for a, b in zip(sync_sig.parameters.values(), async_sig.parameters.values()):
            assert a.default == b.default
            assert a.annotation == b.annotation

    def test_constructor_signatures_match(self):
        assert inspect.signature(Client.__init__) == inspect.signature(
            AsyncClient.__init__
        )
        # The index constructors differ only in which generated DefaultApi /
        # ApiClient they annotate; names and defaults must match.
        sync_params = inspect.signature(EncryptedIndex.__init__).parameters
        async_params = inspect.signature(AsyncEncryptedIndex.__init__).parameters
        assert list(sync_params) == list(async_params)
        assert [p.default for p in sync_params.values()] == [
            p.default for p in async_params.values()
        ]


class TestDescriptors:
    @pytest.mark.parametrize("name", DESCRIPTOR_METHODS)
    def test_descriptor_methods_return_coroutines(self, name):
        index = _async_index()
        result = getattr(index, name)()
        assert inspect.iscoroutine(result)
        result.close()

    def test_index_name_is_a_plain_property(self):
        index = _async_index()
        assert index.index_name == "idx"
        assert isinstance(
            inspect.getattr_static(AsyncEncryptedIndex, "index_name"), property
        )

    @pytest.mark.asyncio
    async def test_dimension_caching_matches_sync(self):
        responses = [
            _describe_response(dimension=0),
            _describe_response(dimension=8),
            _describe_response(dimension=8),
        ]

        sync_api = MagicMock()
        sync_api.get_index_info_v1_indexes_describe_post.side_effect = list(responses)
        sync_index = _sync_index(sync_api)
        sync_seen = [sync_index.dimension, sync_index.dimension, sync_index.dimension]

        async_api = MagicMock()
        async_api.get_index_info_v1_indexes_describe_post = AsyncMock(
            side_effect=list(responses)
        )
        async_index = _async_index(async_api)
        async_seen = [
            await async_index.dimension(),
            await async_index.dimension(),
            await async_index.dimension(),
        ]

        assert async_seen == sync_seen == [0, 8, 8]
        assert (
            async_api.get_index_info_v1_indexes_describe_post.call_count
            == sync_api.get_index_info_v1_indexes_describe_post.call_count
            == 2
        )

    @pytest.mark.asyncio
    async def test_metric_caching_matches_sync(self):
        sync_api = MagicMock()
        sync_api.get_index_info_v1_indexes_describe_post.return_value = (
            _describe_response(metric="cosine")
        )
        sync_index = _sync_index(sync_api)
        sync_seen = [sync_index.metric, sync_index.metric]

        async_api = MagicMock()
        async_api.get_index_info_v1_indexes_describe_post = AsyncMock(
            return_value=_describe_response(metric="cosine")
        )
        async_index = _async_index(async_api)
        async_seen = [await async_index.metric(), await async_index.metric()]

        assert async_seen == sync_seen == ["cosine", "cosine"]
        assert (
            async_api.get_index_info_v1_indexes_describe_post.call_count
            == sync_api.get_index_info_v1_indexes_describe_post.call_count
            == 1
        )

    @pytest.mark.asyncio
    async def test_n_lists_is_fetched_every_call(self):
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post = AsyncMock(
            return_value=_describe_response()
        )
        index = _async_index(api)
        await index.n_lists()
        await index.n_lists()
        assert api.get_index_info_v1_indexes_describe_post.call_count == 2

    @pytest.mark.asyncio
    async def test_describe_errors_are_translated(self):
        exc = AsyncApiException(status=404, reason="Not Found")
        exc.headers = {}
        exc.body = json.dumps({"detail": "index not found"})
        api = MagicMock()
        api.get_index_info_v1_indexes_describe_post = AsyncMock(side_effect=exc)
        index = _async_index(api)
        for name in DESCRIPTOR_METHODS + ["is_trained"]:
            with pytest.raises(NotFoundError):
                await getattr(index, name)()


class TestWireParity:
    """The same inputs produce the same request through either transport."""

    @staticmethod
    def _dict_items():
        return [
            {
                "id": "a",
                "vector": [0.1, 0.2, 0.3],
                "contents": b"\x00\x01",
                "metadata": {"tag": "x", "n": 1},
            },
            {"id": "b", "vector": np.array([1.5, 2.5, 3.5]), "contents": "text"},
        ]

    @pytest.mark.asyncio
    async def test_upsert_bodies_match(self):
        sync_rec, async_rec = _Recorder(), _Recorder()
        sync_index = self._sync_index_on(sync_rec)
        async_index = self._async_index_on(async_rec)

        sync_index.upsert(self._dict_items())
        await async_index.upsert(self._dict_items())

        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/upsert")

    @pytest.mark.asyncio
    async def test_upsert_list_vectors_bodies_match(self):
        sync_rec, async_rec = _Recorder(), _Recorder()
        sync_index = self._sync_index_on(sync_rec)
        async_index = self._async_index_on(async_rec)

        sync_index.upsert(["a", "b"], [[0.1, 0.2], [0.3, 0.4]])
        await async_index.upsert(["a", "b"], [[0.1, 0.2], [0.3, 0.4]])

        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/upsert")

    @pytest.mark.asyncio
    async def test_upsert_binary_bodies_match(self):
        sync_rec, async_rec = _Recorder(), _Recorder()
        sync_index = self._sync_index_on(sync_rec)
        async_index = self._async_index_on(async_rec)
        vectors = np.arange(6, dtype=np.float64).reshape(3, 2)
        metadata = [{"k": 1}, None, {"k": 3}]
        contents = ["hello", None, b"\x00\x01"]

        sync_index.upsert_binary(["a", "b", "c"], vectors, metadata, contents)
        await async_index.upsert_binary(["a", "b", "c"], vectors, metadata, contents)

        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/upsert_binary")

    @pytest.mark.asyncio
    async def test_query_bodies_match(self):
        sync_rec, async_rec = _Recorder(), _Recorder()
        sync_index = self._sync_index_on(sync_rec)
        async_index = self._async_index_on(async_rec)
        kwargs = dict(
            top_k=3,
            n_probes=2,
            filters={"tag": {"$in": ["x", "y"]}},
            include=["metadata"],
            greedy=True,
            text="hello",
            alpha=0.3,
        )

        sync_index.query([0.1, 0.2, 0.3], **kwargs)
        await async_index.query([0.1, 0.2, 0.3], **kwargs)
        sync_index.query([[0.1, 0.2], [0.3, 0.4]], top_k=1)
        await async_index.query([[0.1, 0.2], [0.3, 0.4]], top_k=1)

        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/query", 0)
        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/query", 1)

    @pytest.mark.asyncio
    async def test_query_binary_bodies_match(self):
        sync_rec, async_rec = _Recorder(), _Recorder()
        sync_index = self._sync_index_on(sync_rec)
        async_index = self._async_index_on(async_rec)
        single = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        batch = np.arange(6, dtype=np.float64).reshape(2, 3)

        sync_index.query_binary(single, top_k=2, filters={"k": 1}, rerank_mult=4)
        await async_index.query_binary(single, top_k=2, filters={"k": 1}, rerank_mult=4)
        sync_index.query(batch, top_k=5, include=["distance"])
        await async_index.query(batch, top_k=5, include=["distance"])

        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/query_binary", 0)
        self._assert_same_request(sync_rec, async_rec, "/v1/vectors/query_binary", 1)

    @staticmethod
    def _sync_index_on(recorder):
        client = _sync_client(recorder)
        return EncryptedIndex("idx", KEY, client.api, client.api_client)

    @staticmethod
    def _async_index_on(recorder):
        client = _async_client(recorder)
        return AsyncEncryptedIndex("idx", KEY, client.api, client.api_client)

    @staticmethod
    def _assert_same_request(sync_rec, async_rec, path, call=0):
        sync_method, sync_url, sync_body, sync_headers = sync_rec.calls[call]
        async_method, async_url, async_body, async_headers = async_rec.calls[call]
        assert sync_method == async_method == "POST"
        assert sync_url == async_url == BASE_URL + path
        assert _canonical(sync_body) == _canonical(async_body)
        assert sync_headers["X-API-Key"] == async_headers["x-api-key"] == API_KEY
        assert sync_headers["Content-Type"] == async_headers["content-type"]


SYNC_TO_ASYNC_STATUSES = [401, 403, 404, 409, 422, 429, 500]
EXPECTED_BY_STATUS = {
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    429: RateLimitError,
    500: ServiceError,
}


def _api_exception(cls, status):
    exc = cls(status=status, reason="synthetic")
    exc.body = json.dumps({"detail": "synthetic failure"})
    exc.headers = {"X-Request-Id": "req-1", "Retry-After": "1.5"}
    return exc


class TestErrorTranslation:
    @pytest.mark.parametrize("status", SYNC_TO_ASYNC_STATUSES)
    def test_async_api_exception_maps_like_sync(self, status):
        from_sync = translate_api_error(_api_exception(SyncApiException, status), "op")
        from_async = translate_api_error(
            _api_exception(AsyncApiException, status), "op"
        )
        assert type(from_sync) is type(from_async) is EXPECTED_BY_STATUS[status]
        assert from_async.status_code == from_sync.status_code == status
        assert from_async.request_id == from_sync.request_id == "req-1"
        assert from_async.detail == from_sync.detail == "synthetic failure"
        assert from_async.retry_after == from_sync.retry_after == 1.5
        assert from_async.retryable == from_sync.retryable

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("timed out"),
            httpx.RemoteProtocolError("server disconnected"),
            httpx.ConnectTimeout("connect timed out"),
        ],
    )
    def test_httpx_transport_errors_map_to_transport_error(self, exc):
        result = translate_api_error(exc, "op")
        assert isinstance(result, TransportError)
        assert result.status_code is None
        assert result.retryable is True
        assert str(exc) in str(result)

    def test_unrelated_exceptions_pass_through(self):
        exc = RuntimeError("not ours")
        assert translate_api_error(exc, "op") is exc

    @pytest.mark.asyncio
    async def test_transport_failure_end_to_end(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        client = AsyncClient(BASE_URL, api_key=API_KEY)
        client.api_client.rest_client.pool_manager = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        with pytest.raises(TransportError):
            await client.get_health()
        index = AsyncEncryptedIndex("idx", KEY, client.api, client.api_client)
        with pytest.raises(TransportError):
            await index.query([0.1, 0.2])
        with pytest.raises(TransportError):
            await index.upsert_binary(["a"], np.zeros((1, 2), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_query_status_error_is_translated(self):
        # query uses the raw-response endpoint, so the status check is ours.
        def handler(request):
            return httpx.Response(403, json={"detail": "no read permission"})

        client = AsyncClient(BASE_URL, api_key=API_KEY)
        client.api_client.rest_client.pool_manager = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        index = AsyncEncryptedIndex("idx", KEY, client.api, client.api_client)
        with pytest.raises(AuthenticationError) as ctx:
            await index.query([0.1, 0.2])
        assert ctx.value.status_code == 403
        assert ctx.value.detail == "no read permission"


class TestValidation:
    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error_like_sync(self):
        with pytest.raises(ValueError) as sync_ctx:
            _sync_index().query([])
        with pytest.raises(ValueError) as async_ctx:
            await _async_index().query([])
        assert type(async_ctx.value) is type(sync_ctx.value) is ValidationError
        assert str(async_ctx.value) == str(sync_ctx.value)

    @pytest.mark.asyncio
    async def test_validation_errors_are_not_wrapped(self):
        index = _async_index()
        vectors = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValidationError) as ctx:
            await index.upsert_binary(["a"], [[0.0, 1.0]])
        assert isinstance(ctx.value, TypeError)
        with pytest.raises(ValidationError):
            await index.upsert_binary(["a", "b"], vectors, contents=["only one"])
        with pytest.raises(ValidationError):
            await index.upsert([{"vector": [0.1]}])
        with pytest.raises(ValidationError):
            await index.query(np.zeros((1, 1, 1), dtype=np.float32))
        with pytest.raises(ValidationError):
            await index.query_metadata(order_by={"a": 1, "b": -1})
        index._api.upsert_vectors_v1_vectors_upsert_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_bad_index_key_is_rejected_before_any_request(self):
        client = AsyncClient(BASE_URL, api_key=API_KEY)
        with pytest.raises(ValidationError):
            await client.create_index("idx", b"short", dimension=4)
        with pytest.raises(ValidationError):
            await client.load_index("idx", b"short")
        with pytest.raises(ValidationError):
            await client.create_index("idx")


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_async_with_closes_the_client(self):
        pool = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"status": "healthy"})
            )
        )
        async with AsyncClient(BASE_URL, api_key=API_KEY) as client:
            client.api_client.rest_client.pool_manager = pool
            await client.get_health()
            assert not pool.is_closed
        assert pool.is_closed

    @pytest.mark.asyncio
    async def test_async_with_on_index_closes_shared_pool(self):
        pool = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None))
        client = AsyncClient(BASE_URL, api_key=API_KEY)
        client.api_client.rest_client.pool_manager = pool
        async with AsyncEncryptedIndex("idx", KEY, client.api, client.api_client):
            assert not pool.is_closed
        assert pool.is_closed

    @pytest.mark.asyncio
    async def test_close_is_explicit_and_awaitable(self):
        client = AsyncClient(BASE_URL, api_key=API_KEY)
        client.api_client.close = AsyncMock()
        await client.close()
        client.api_client.close.assert_awaited_once()

    def test_api_key_is_sent_as_header(self):
        client = AsyncClient(BASE_URL, api_key=API_KEY)
        assert client.api_client.default_headers["X-API-Key"] == API_KEY
        assert client._request_headers()["X-API-Key"] == API_KEY
        bare = AsyncClient(BASE_URL)
        assert "X-API-Key" not in bare._request_headers()
