"""AsyncClient — async counterpart to Client."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Dict, List, Literal, Optional

import httpx

from cyborgdb.client.async_encrypted_index import AsyncEncryptedIndex
from cyborgdb.exceptions import translate_api_error

logger = logging.getLogger(__name__)

_ASYNC_TRANSPORT = (httpx.TransportError, httpx.RequestError)


class AsyncClient:
    """Async client for interacting with CyborgDB via REST API.

    Mirrors :class:`cyborgdb.client.client.Client` with all I/O methods
    as ``async def``.  Uses the httpx-backed transport from
    ``cyborgdb.openapi_client_async``; no ``asyncio.to_thread`` anywhere.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify_ssl=None,
    ) -> None:
        from cyborgdb.openapi_client.configuration import Configuration
        from cyborgdb.openapi_client_async.api.default_api import DefaultApi
        from cyborgdb.openapi_client_async.api_client import AsyncApiClient

        config = Configuration()
        config.host = base_url

        if verify_ssl is None:
            if base_url.startswith("http://"):
                verify_ssl = False
            elif "localhost" in base_url or "127.0.0.1" in base_url:
                verify_ssl = False
                logger.info(
                    "SSL verification disabled for localhost (development mode)"
                )
            else:
                verify_ssl = True
        elif not verify_ssl:
            logger.warning(
                "SSL verification is disabled. Not recommended for production."
            )
        config.verify_ssl = verify_ssl

        if api_key:
            config.api_key = {"X-API-Key": api_key}

        try:
            self._async_api_client = AsyncApiClient(config)
            self._api = DefaultApi(self._async_api_client)
        except Exception as e:
            error_msg = f"Failed to initialize async client: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.config = config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._async_api_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Static helpers (no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key(save: bool = False) -> bytes:
        """Generate a secure 32-byte index key (identical to Client.generate_key)."""
        if not save:
            return secrets.token_bytes(32)

        key_path = Path.home() / ".cyborgdb" / "index_key"
        key_path.parent.mkdir(parents=True, exist_ok=True)

        if key_path.exists() and key_path.stat().st_size == 32:
            logger.warning(
                f"Loading existing index key from '{key_path}'.\n"
                "Saving keys is not recommended for production use."
            )
            return key_path.read_bytes()

        key = secrets.token_bytes(32)
        key_path.write_bytes(key)
        logger.warning(
            f"Generated new index key and saved to '{key_path}'.\n"
            "Saving keys is not recommended for production use."
        )
        return key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = self.config.api_key.get("X-API-Key")
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    def _validate_index_key(self, index_key: bytes) -> None:
        if not isinstance(index_key, bytes) or len(index_key) != 32:
            raise ValueError("index_key must be a 32-byte bytes object")

    def _make_index(
        self, index_name: str, index_key: Optional[bytes]
    ) -> AsyncEncryptedIndex:
        return AsyncEncryptedIndex(
            index_name=index_name,
            index_key=index_key,
            api=self._api,
            api_client=self._async_api_client,
        )

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    async def list_indexes(self) -> List[str]:
        try:
            response = await self._api.list_indexes_v1_indexes_list_get()
            return response.indexes
        except Exception as e:
            raise translate_api_error(
                e, "Failed to list indexes", transport_error_types=_ASYNC_TRANSPORT
            ) from e

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
        if index_key is None and kms_name is None:
            raise ValueError("create_index requires index_key, kms_name, or both")

        if index_key is not None:
            self._validate_index_key(index_key)

        try:
            from cyborgdb.openapi_client.models import (
                CreateIndexRequest as _OpenAPICreateIndexRequest,
            )

            index = self._make_index(index_name, index_key)

            request = _OpenAPICreateIndexRequest(
                index_name=index_name,
                index_key=index._key_to_hex(),
                kms_name=kms_name,
                dimension=dimension,
                embedding_model=embedding_model,
                metric=metric,
                storage_precision=storage_precision,
                metadata_schema=metadata_schema,
                text_fields=text_fields,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            )

            await self._api.create_index_v1_indexes_create_post(
                create_index_request=request,
                _headers=self._request_headers(),
            )
            return index

        except ValueError:
            raise
        except Exception as e:
            raise translate_api_error(
                e, "Failed to create index", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def load_index(
        self,
        index_name: str,
        index_key: Optional[bytes] = None,
    ) -> AsyncEncryptedIndex:
        if index_key is not None:
            self._validate_index_key(index_key)

        try:
            index = self._make_index(index_name, index_key)
            # Probe the describe endpoint to verify existence
            _ = await index.dimension()
            return index

        except Exception as e:
            raise translate_api_error(
                e,
                f"Failed to load index '{index_name}'",
                transport_error_types=_ASYNC_TRANSPORT,
            ) from e

    async def get_health(self) -> Dict[str, str]:
        try:
            return await self._api.health_check_v1_health_get()
        except Exception as e:
            raise translate_api_error(
                e, "Failed to get health status", transport_error_types=_ASYNC_TRANSPORT
            ) from e
