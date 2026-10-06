"""
CyborgDB async REST Client

Native asyncio counterpart of :class:`cyborgdb.Client`, backed by the httpx
transport in ``cyborgdb.openapi_client_async``.
"""

import logging
from typing import Dict, List, Literal, Optional
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError as PydanticValidationError

try:
    from cyborgdb.openapi_client_async.api_client import ApiClient, Configuration
    from cyborgdb.openapi_client_async.api.default_api import DefaultApi
    from cyborgdb.openapi_client_async.exceptions import ApiException
    from cyborgdb.openapi_client_async.models import CreateIndexRequest
except ImportError:
    raise ImportError(
        "Failed to import openapi_client_async. Make sure the OpenAPI client library is properly installed."
    )

from cyborgdb.client._request_builders import (
    apply_full_text_implication,
    request_headers,
    validate_index_key,
)
from cyborgdb.client.async_encrypted_index import AsyncEncryptedIndex
from cyborgdb.client.client import Client
from cyborgdb.exceptions import CyborgDBError, ValidationError, translate_api_error

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncClient",
    "AsyncEncryptedIndex",
]

_TRANSPORT_ERRORS = (ApiException, httpx.RequestError)


class AsyncClient:
    """
    Async client for interacting with CyborgDB via REST API.

    This class provides coroutine methods for creating, loading, and managing
    encrypted indexes; it is the asyncio counterpart of :class:`cyborgdb.Client`
    and accepts the same constructor arguments. Index handles it returns are
    :class:`AsyncEncryptedIndex` instances.

    The client owns an httpx connection pool. Close it with ``await
    client.close()`` or use the client as an async context manager::

        async with AsyncClient("http://localhost:8000", api_key=key) as client:
            index = await client.load_index("my-index", index_key)
            results = await index.query(query_vectors=vector, top_k=5)

    The ``api_key`` passed at construction is sent as the ``X-API-Key`` header
    on every request and may be any of three kinds, depending on how the
    service is deployed:

    - **Single service key** — the default; the one ``CYBORGDB_API_KEY`` the
      service was started with. Full access, no RBAC.
    - **Root key** — when the service runs with ``CYBORGDB_SERVICE_ROOT_KEY`` set,
      RBAC is on. A client using the root key has admin access and can mint
      per-user keys via :meth:`AsyncEncryptedIndex.create_user`.
    - **User key** (``cdbk_...``) — minted by ``create_user`` and scoped to one
      index with ``read`` / ``write`` permissions enforced cryptographically.
      A user client calls ``load_index(name)`` with **no** ``index_key`` (the
      service resolves it), then performs the data operations its permissions
      allow. User keys work only against KMS-backed indexes (the service must
      be able to resolve the index KEK server-side); SDK-supplied-KEK indexes
      have no server-side key to resolve for a user.
    """

    def __init__(self, base_url, api_key: Optional[str] = None, verify_ssl=None):
        self.config = Configuration()
        self.config.host = base_url

        # Configure SSL verification
        if base_url.startswith("http://"):
            if verify_ssl is True:
                logger.warning(
                    "verify_ssl=True has no effect on http:// URLs (no TLS to negotiate); ignored."
                )
            self.config.verify_ssl = False
        elif verify_ssl is None:
            parsed = urlparse(base_url)
            if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
                self.config.verify_ssl = False
                logger.warning(
                    "SSL verification auto-disabled for %r (loopback host detected; development mode). "
                    "Not recommended for production.",
                    parsed.hostname,
                )
            else:
                self.config.verify_ssl = True
        else:
            self.config.verify_ssl = verify_ssl
            if not verify_ssl:
                logger.warning(
                    "SSL verification is disabled. Not recommended for production."
                )

        if api_key:
            self.config.api_key = {"X-API-Key": api_key}

        # Construction opens no connections: the generated client creates its
        # httpx pool lazily on the first request, inside the running loop.
        try:
            self.api_client = ApiClient(self.config)
            self.api = DefaultApi(self.api_client)

            if api_key:
                self.api_client.default_headers["X-API-Key"] = api_key

        except Exception as e:
            error_msg = f"Failed to initialize client: {e}"
            logger.error(error_msg)
            raise CyborgDBError(error_msg) from e

    async def close(self) -> None:
        """Close the underlying connection pool.

        Indexes obtained from this client share the pool and stop working once
        it is closed.
        """
        await self.api_client.close()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    def _request_headers(self) -> Dict[str, str]:
        """Build the request headers for data-path calls. Only includes
        ``X-API-Key`` when one is configured; when the service has auth
        disabled (no ``CYBORGDB_SERVICE_ROOT_KEY`` set) the SDK can be
        constructed with no api_key and we must not send an empty
        header (and must not crash indexing into an empty config dict)."""
        return request_headers(self.config.api_key.get("X-API-Key"))

    generate_key = staticmethod(Client.generate_key)

    async def list_indexes(self) -> List[str]:
        """
        Get a list of all encrypted index names accessible via the client.

        Returns:
            A list of index names.

        Raises:
            CyborgDBError: If the list of indexes could not be retrieved.
        """
        try:
            response = await self.api.list_indexes_v1_indexes_list_get()
            return response.indexes
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to list indexes") from e

    async def create_index(
        self,
        index_name: str,
        index_key: Optional[bytes] = None,
        kms_name: Optional[str] = None,
        dimension: Optional[int] = None,
        embedding_model: Optional[str] = None,
        metric: Optional[str] = None,
        storage_precision: Optional[
            Literal["float32", "float16", "tq12", "tq8", "tq6", "tq4"]
        ] = None,
        metadata_schema: Optional[Dict[str, Dict[str, bool]]] = None,
        text_fields: Optional[List[str]] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
    ) -> AsyncEncryptedIndex:
        """
        Create and return a new encrypted DiskIVF index.

        At least one of ``index_key`` or ``kms_name`` must be provided, and
        the service accepts exactly one of them:

        - ``index_key`` only — the SDK supplies the 32-byte wrapping key; the
          service records the index as ``provider: none`` and does no KMS
          round-trips. The same key must be re-supplied to ``load_index``.
        - ``kms_name`` only — the service generates the key and wraps it under
          the named ``kms.registry`` entry (``aws-kms`` / ``aws``); the SDK
          never sees the plaintext key, and ``load_index`` needs no key.

        Supplying both is forwarded as-is and rejected by the service with a
        400, for every provider: the named slot already determines the key
        source, so an SDK-supplied key is contradictory. Note that ``none`` is
        not a registry slot type — the no-KMS path is reached by omitting
        ``kms_name``, not by naming a ``provider: none`` slot.

        ``storage_precision`` selects the on-disk rerank-vector format, chosen
        at create time and immutable. ``float32`` (the default) keeps full
        precision; ``float16`` halves storage at a small precision cost. The
        four TurboQuant tiers ``tq12`` / ``tq8`` / ``tq6`` / ``tq4`` pack
        12 / 8 / 6 / 4 bits per dimension, trading a little recall and latency
        for a large storage saving — ``tq4`` is the most aggressive (~8x
        smaller, ~94% recall@100). Every tier works with every metric.

        ``metadata_schema`` is the per-field metadata indexing policy, fixed at
        create time and immutable::

            {"title": {"filterable": True, "pattern": True},
             "blob":  {"filterable": False}}

        Fields left out are filterable (opt-out posture). ``filterable`` builds
        inverted-index postings; ``pattern`` additionally builds a regex
        dictionary and requires ``filterable=True``. On :meth:`AsyncEncryptedIndex.query`
        this only decides how a filter is resolved — index vs. post-filter, same
        rows either way. On :meth:`AsyncEncryptedIndex.query_metadata` it is enforced:
        only ``pattern`` fields accept ``$regex``/``$contains``, and
        non-filterable fields cannot be filtered on at all.

        A third policy, ``full_text``, routes the field's string value through
        the BM25 analyzer instead of exact-match indexing, making it searchable
        by :meth:`AsyncEncryptedIndex.query_metadata` (``text=...``) and hybrid
        :meth:`AsyncEncryptedIndex.query` (``text=...``). ``full_text=True`` implies
        ``filterable=False`` and is incompatible with ``pattern=True``::

            {"body": {"full_text": True}}

        ``text_fields`` is shorthand for marking fields ``full_text=True`` in
        ``metadata_schema``. ``bm25_k1`` (term-frequency saturation, default
        1.2) and ``bm25_b`` (length-normalization strength, default 0.75) tune
        the scorer; both require at least one full-text field. BM25 search is
        opt-in and derived — an index with no full-text field writes no BM25
        config at all.
        """
        if index_key is None and kms_name is None:
            raise ValidationError("create_index requires index_key, kms_name, or both")

        if index_key is not None:
            validate_index_key(index_key)

        try:
            # Build the handle first (no network I/O); it owns the single
            # hex encoding of the key, which we reuse for the request below.
            index = AsyncEncryptedIndex(
                index_name=index_name,
                index_key=index_key,
                api=self.api,
                api_client=self.api_client,
            )

            request = CreateIndexRequest(
                index_name=index_name,
                index_key=index._key_to_hex(),
                kms_name=kms_name,
                dimension=dimension,
                embedding_model=embedding_model,
                metric=metric,
                storage_precision=storage_precision,
                metadata_schema=apply_full_text_implication(metadata_schema),
                text_fields=text_fields,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            )

            await self.api.create_index_v1_indexes_create_post(
                create_index_request=request,
                _headers=self._request_headers(),
            )

            return index

        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to create index") from e
        except PydanticValidationError as ve:
            error_msg = f"Validation error while creating index: {ve}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from ve

    async def load_index(
        self,
        index_name: str,
        index_key: Optional[bytes] = None,
    ) -> AsyncEncryptedIndex:
        """
        Load an existing encrypted index by name.

        ``index_key`` is required for ``provider: none`` indexes (the SDK owns
        the KEK). For KMS-backed indexes the service resolves the DEK via the
        stored ``KMSBlob``, so ``index_key`` can be omitted.
        """
        if index_key is not None:
            validate_index_key(index_key)

        try:
            index = AsyncEncryptedIndex(
                index_name=index_name,
                index_key=index_key,
                api=self.api,
                api_client=self.api_client,
            )

            # Probe the describe endpoint so a missing/inaccessible index
            # raises here instead of silently returning a phantom handle.
            # The probe also primes the handle's metadata cache.
            await index._describe()

            return index

        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, f"Failed to load index '{index_name}'") from e
        except PydanticValidationError as ve:
            error_msg = f"Validation error while loading index '{index_name}': {ve}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from ve

    async def get_health(self) -> Dict[str, str]:
        """
        Get the health status of the CyborgDB instance.

        Returns:
            A dictionary containing health status information.

        Raises:
            CyborgDBError: If the health status could not be retrieved.
        """
        try:
            return await self.api.health_check_v1_health_get()
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to get health status") from e
