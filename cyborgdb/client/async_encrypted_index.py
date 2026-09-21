"""AsyncEncryptedIndex — async counterpart to EncryptedIndex."""

from __future__ import annotations

import binascii
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np

from cyborgdb.client._request_builders import (
    build_query_binary_request,
    build_upsert_binary_request,
    build_upsert_items_list,
    parse_query_binary_response,
    parse_raw_query_response_bytes,
    prepare_query_request,
)
from cyborgdb.exceptions import translate_api_error

logger = logging.getLogger(__name__)

_ASYNC_TRANSPORT = (httpx.TransportError, httpx.RequestError)


class AsyncEncryptedIndex:
    """Async access to an encrypted vector index via the REST API.

    Mirrors EncryptedIndex but every I/O method is ``async def``.
    The five descriptor properties (dimension, metric, n_lists,
    metadata_schema, bm25) become awaitable methods because Python
    properties cannot be async.

    Caching behaviour matches the sync counterpart:
    - dimension / metric: cached after first call.
    - n_lists / metadata_schema / bm25: fetched fresh every time.
    """

    def __init__(
        self,
        index_name: str,
        index_key: Optional[bytes],
        api,
        api_client,
    ) -> None:
        self._index_name = index_name
        self._index_key_hex: Optional[str] = (
            binascii.hexlify(index_key).decode("ascii")
            if index_key is not None
            else None
        )
        self._api = api
        self._api_client = api_client
        self._dimension: Optional[int] = None
        self._metric: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Synchronous property (no I/O)
    # ------------------------------------------------------------------

    @property
    def index_name(self) -> str:
        return self._index_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_to_hex(self) -> Optional[str]:
        return self._index_key_hex

    def _index_op_request(self):
        from cyborgdb.openapi_client.models.index_operation_request import (
            IndexOperationRequest,
        )

        return IndexOperationRequest(
            index_key=self._key_to_hex(), index_name=self._index_name
        )

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = self._api_client.configuration.api_key.get("X-API-Key")
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    async def _describe(self):
        return await self._api.get_index_info_v1_indexes_describe_post(
            index_operation_request=self._index_op_request()
        )

    # ------------------------------------------------------------------
    # Awaitable descriptor methods
    # ------------------------------------------------------------------

    async def dimension(self) -> int:
        """Vector dimensionality. Cached after first call."""
        if self._dimension is None:
            response = await self._describe()
            self._dimension = response.dimension
            self._metric = response.metric
        return self._dimension

    async def metric(self) -> str:
        """Distance metric. Cached after first call."""
        if self._metric is None:
            response = await self._describe()
            self._dimension = response.dimension
            self._metric = response.metric
        return self._metric

    async def n_lists(self) -> int:
        """Number of inverted lists. Fetched fresh (mutated by train)."""
        return (await self._describe()).n_lists

    async def metadata_schema(self) -> Dict[str, Dict[str, bool]]:
        """Per-field metadata indexing policy. Fetched fresh."""
        return {
            field: {
                "filterable": policy.filterable,
                "pattern": policy.pattern,
                "full_text": policy.full_text,
            }
            for field, policy in (
                (await self._describe()).metadata_schema or {}
            ).items()
        }

    async def bm25(self) -> Optional[Dict[str, Any]]:
        """BM25 scorer config, or None. Fetched fresh."""
        config = (await self._describe()).bm25
        if config is None:
            return None
        return {
            "k1": config.k1,
            "b": config.b,
            "analyzer_version": config.analyzer_version,
        }

    # ------------------------------------------------------------------
    # Data methods
    # ------------------------------------------------------------------

    async def upsert(
        self,
        arg1: Union[List[Dict[str, Any]], List[str], "np.ndarray"],
        arg2: Optional["np.ndarray"] = None,
    ) -> None:
        if arg2 is not None and isinstance(arg2, np.ndarray):
            if not isinstance(arg1, list):
                raise TypeError("arg1 must be a list of IDs")
            ids = [str(id_val) for id_val in arg1]
            await self.upsert_binary(ids, arg2)
            return

        try:
            from cyborgdb.openapi_client.models import UpsertRequest

            items = build_upsert_items_list(arg1, arg2)
            request = UpsertRequest(
                items=items,
                index_key=self._key_to_hex(),
                index_name=self._index_name,
            )
            await self._api.upsert_vectors_v1_vectors_upsert_post(
                upsert_request=request,
                _headers=self._request_headers(),
            )
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise
        except Exception as e:
            raise translate_api_error(
                e, "Failed to upsert items", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def upsert_binary(
        self,
        ids: List[str],
        vectors: "np.ndarray",
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        contents: Optional[List[Optional[Union[str, bytes]]]] = None,
    ) -> None:
        try:
            request = build_upsert_binary_request(
                self._index_name,
                self._key_to_hex(),
                ids,
                vectors,
                metadata=metadata,
                contents=contents,
            )
            await self._api.upsert_vectors_binary_v1_vectors_upsert_binary_post(
                binary_upsert_request=request,
                _headers=self._request_headers(),
            )
        except (TypeError, ValueError):
            raise
        except Exception as e:
            raise translate_api_error(
                e,
                "Failed to upsert items (binary)",
                transport_error_types=_ASYNC_TRANSPORT,
            ) from e

    async def query(
        self,
        query_vectors=None,
        query_contents: Optional[str] = None,
        top_k: Optional[int] = None,
        n_probes: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        greedy: Optional[bool] = None,
        rerank_mult: Optional[int] = None,
        text: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        text_field_weights: Optional[List[float]] = None,
        require_all_terms: Optional[bool] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[float] = None,
        window_mult: Optional[int] = None,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        hybrid_kwargs = {
            k: v
            for k, v in {
                "text": text,
                "text_fields": text_fields,
                "text_field_weights": text_field_weights,
                "require_all_terms": require_all_terms,
                "alpha": alpha,
                "rrf_k": rrf_k,
                "window_mult": window_mult,
            }.items()
            if v is not None
        }

        try:
            if query_vectors is not None and isinstance(query_vectors, np.ndarray):
                if query_vectors.ndim not in (1, 2):
                    raise ValueError(
                        "Expected 1D or 2D NumPy array for `query_vectors`."
                    )
                return await self.query_binary(
                    query_vectors=query_vectors,
                    top_k=top_k,
                    n_probes=n_probes,
                    filters=filters,
                    include=include,
                    greedy=greedy,
                    rerank_mult=rerank_mult,
                    **hybrid_kwargs,
                )

            dispatch, request, is_single_query = prepare_query_request(
                self._index_name,
                self._key_to_hex(),
                query_vectors,
                query_contents,
                top_k,
                n_probes,
                filters,
                include,
                greedy,
                rerank_mult,
                hybrid_kwargs,
            )
            if dispatch:
                raise RuntimeError(
                    "prepare_query_request returned dispatch=True after numpy guard; "
                    "this is a bug — please report it"
                )

            raw_response = await self._api.query_vectors_v1_vectors_query_post(
                request=request,
                _headers=self._request_headers(),
            )

            if not 200 <= raw_response.status <= 299:
                from cyborgdb.openapi_client.exceptions import ApiException

                raise ApiException.from_response(
                    http_resp=raw_response,
                    body=raw_response.data.decode("utf-8"),
                    data=None,
                )

            return parse_raw_query_response_bytes(raw_response.data, include)

        except (TypeError, ValueError):
            raise
        except Exception as e:
            raise translate_api_error(
                e, "Query failed", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def query_binary(
        self,
        query_vectors: "np.ndarray",
        top_k: Optional[int] = None,
        n_probes: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        greedy: Optional[bool] = None,
        rerank_mult: Optional[int] = None,
        text: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        text_field_weights: Optional[List[float]] = None,
        require_all_terms: Optional[bool] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[float] = None,
        window_mult: Optional[int] = None,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        hybrid_kwargs = {
            k: v
            for k, v in {
                "text": text,
                "text_fields": text_fields,
                "text_field_weights": text_field_weights,
                "require_all_terms": require_all_terms,
                "alpha": alpha,
                "rrf_k": rrf_k,
                "window_mult": window_mult,
            }.items()
            if v is not None
        }

        try:
            request, is_single_query = build_query_binary_request(
                self._index_name,
                self._key_to_hex(),
                query_vectors,
                top_k,
                n_probes,
                filters,
                include,
                greedy,
                rerank_mult,
                hybrid_kwargs,
            )
            response = (
                await self._api.query_vectors_binary_v1_vectors_query_binary_post(
                    binary_query_request=request,
                    _headers=self._request_headers(),
                )
            )
            return parse_query_binary_response(response, is_single_query)

        except (TypeError, ValueError):
            raise
        except Exception as e:
            raise translate_api_error(
                e, "Failed to query (binary)", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def query_metadata(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        order_by: Optional[Union[str, Dict[str, int]]] = None,
        ascending: bool = True,
        text: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        text_field_weights: Optional[List[float]] = None,
        require_all_terms: Optional[bool] = None,
    ):
        from cyborgdb.openapi_client.models.order_by import OrderBy
        from cyborgdb.openapi_client.models.query_metadata_request import (
            QueryMetadataRequest,
        )

        if isinstance(order_by, dict):
            if len(order_by) != 1:
                raise ValueError(
                    f"order_by dict must specify exactly one field, got {len(order_by)}"
                )
            ((order_by, direction),) = order_by.items()
            ascending = int(direction) >= 0

        request_kwargs: Dict[str, Any] = {
            "index_key": self._key_to_hex(),
            "index_name": self._index_name,
            "filters": filters or {},
            "top_k": top_k,
            "order_by": OrderBy(order_by) if order_by is not None else None,
            "ascending": ascending,
        }
        for key, value in {
            "text": text,
            "text_fields": text_fields,
            "text_field_weights": text_field_weights,
            "require_all_terms": require_all_terms,
        }.items():
            if value is not None:
                request_kwargs[key] = value

        try:
            request = QueryMetadataRequest(**request_kwargs)
            response = await self._api.query_metadata_v1_vectors_query_metadata_post(
                query_metadata_request=request
            )
            rows = response.results or []
            if text:
                return [{"id": item.id, "score": item.score} for item in rows]
            return [{"id": item.id} for item in rows]
        except Exception as e:
            raise translate_api_error(
                e, "Failed to query metadata", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def get(
        self,
        ids: List[str],
        include: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if include is None:
            include = ["vector", "contents", "metadata"]
        from cyborgdb.openapi_client.models import GetRequest

        try:
            get_request = GetRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                ids=ids,
                include=include,
            )
            response = await self._api.get_vectors_v1_vectors_get_post(
                get_request=get_request,
                _headers=self._request_headers(),
            )

            items = []
            if hasattr(response, "results"):
                for item in response.results:
                    item_dict: Dict[str, Any] = {"id": item.id}
                    if "vector" in include and hasattr(item, "vector"):
                        item_dict["vector"] = item.vector
                    if "contents" in include and hasattr(item, "contents"):
                        item_dict["contents"] = item.contents
                    if "metadata" in include and hasattr(item, "metadata"):
                        if isinstance(item.metadata, str):
                            try:
                                item_dict["metadata"] = json.loads(item.metadata)
                            except json.JSONDecodeError:
                                item_dict["metadata"] = {}
                        else:
                            item_dict["metadata"] = item.metadata
                    items.append(item_dict)
            return items
        except Exception as e:
            raise translate_api_error(
                e, "Failed to retrieve items", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def train(
        self,
        n_lists: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_iters: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        from cyborgdb.openapi_client.models.train_request import TrainRequest

        try:
            request = TrainRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                n_lists=n_lists,
                batch_size=batch_size,
                max_iters=max_iters,
                tolerance=tolerance,
            )
            await self._api.train_index_v1_indexes_train_post(train_request=request)
        except Exception as e:
            raise translate_api_error(
                e, "Failed to train index", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def delete(self, ids: List[str]) -> None:
        from cyborgdb.openapi_client.models.delete_request import DeleteRequest

        try:
            delete_request = DeleteRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                ids=ids,
            )
            await self._api.delete_vectors_v1_vectors_delete_post(
                delete_request=delete_request
            )
        except Exception as e:
            raise translate_api_error(
                e, "Failed to delete items", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def delete_index(self) -> None:
        try:
            await self._api.delete_index_v1_indexes_delete_post(
                index_operation_request=self._index_op_request()
            )
        except Exception as e:
            raise translate_api_error(
                e, "Failed to delete index", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def list_ids(self) -> List[str]:
        from cyborgdb.openapi_client.models.list_ids_request import ListIDsRequest

        try:
            list_ids_request = ListIDsRequest(
                index_key=self._key_to_hex(), index_name=self._index_name
            )
            response = await self._api.list_ids_v1_vectors_list_ids_post(
                list_ids_request=list_ids_request
            )
            return response.ids
        except Exception as e:
            raise translate_api_error(
                e, "Failed to list document IDs", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def is_trained(self) -> bool:
        from cyborgdb.openapi_client.exceptions import ApiException

        try:
            response = await self._describe()
            return response.is_trained
        except ApiException as e:
            if e.status == 404:
                return False
            raise translate_api_error(
                e,
                "Failed to get training status",
                transport_error_types=_ASYNC_TRANSPORT,
            ) from e
        except Exception as e:
            raise translate_api_error(
                e,
                "Failed to get training status",
                transport_error_types=_ASYNC_TRANSPORT,
            ) from e

    async def is_training(self) -> bool:
        try:
            response = (
                await self._api.get_training_status_v1_indexes_training_status_get()
            )
            return self._index_name in response.training_indexes
        except Exception as e:
            raise translate_api_error(
                e,
                "Failed to get index training status",
                transport_error_types=_ASYNC_TRANSPORT,
            ) from e

    async def create_user(self, permissions: List[str]) -> Dict[str, str]:
        from cyborgdb.openapi_client.models.create_user_request import CreateUserRequest

        request = CreateUserRequest(
            permissions=permissions, index_key=self._index_key_hex
        )
        try:
            response = await self._api.create_user_v1_indexes_index_name_users_post(
                index_name=self._index_name, create_user_request=request
            )
            return {"user_id": response.user_id, "api_key": response.api_key}
        except Exception as e:
            raise translate_api_error(
                e, "Failed to create user", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def list_users(self) -> List[Dict[str, Any]]:
        try:
            response = await self._api.list_users_v1_indexes_index_name_users_get(
                index_name=self._index_name, x_index_key=self._index_key_hex
            )
            return [
                {"user_id": u.user_id, "permissions": u.permissions}
                for u in response.users
            ]
        except Exception as e:
            raise translate_api_error(
                e, "Failed to list users", transport_error_types=_ASYNC_TRANSPORT
            ) from e

    async def delete_user(self, user_id: str) -> None:
        try:
            await self._api.delete_user_v1_indexes_index_name_users_user_id_delete(
                index_name=self._index_name,
                user_id=user_id,
                x_index_key=self._index_key_hex,
            )
        except Exception as e:
            raise translate_api_error(
                e, "Failed to delete user", transport_error_types=_ASYNC_TRANSPORT
            ) from e
