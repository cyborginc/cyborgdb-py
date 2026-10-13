"""
AsyncEncryptedIndex class for CyborgDB

Native asyncio counterpart of :class:`cyborgdb.EncryptedIndex`, backed by the
httpx transport in ``cyborgdb.openapi_client_async``.
"""

import binascii
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np

from cyborgdb.client._request_builders import (
    build_binary_query_request,
    build_binary_upsert_request,
    build_query_metadata_request,
    build_query_request,
    build_upsert_request,
    hybrid_query_kwargs,
    parse_binary_query_response,
    parse_get_response,
    parse_query_metadata_response,
    parse_query_response,
    request_headers,
)
from cyborgdb.client.encrypted_index import MetadataResult
from cyborgdb.exceptions import (
    ValidationError,
    _ArgumentTypeError,
    translate_api_error,
)

try:
    import cyborgdb.openapi_client_async.models as _models
    from cyborgdb.openapi_client_async.api_client import ApiClient
    from cyborgdb.openapi_client_async.api.default_api import DefaultApi
    from cyborgdb.openapi_client_async.exceptions import ApiException
    from cyborgdb.openapi_client_async.models.create_user_request import (
        CreateUserRequest,
    )
    from cyborgdb.openapi_client_async.models.delete_request import DeleteRequest
    from cyborgdb.openapi_client_async.models.get_request import GetRequest
    from cyborgdb.openapi_client_async.models.index_operation_request import (
        IndexOperationRequest,
    )
    from cyborgdb.openapi_client_async.models.list_ids_request import ListIDsRequest
    from cyborgdb.openapi_client_async.models.train_request import TrainRequest
    from cyborgdb.openapi_client_async.rest import RESTResponse
except ImportError:
    raise ImportError(
        "Failed to import openapi_client_async. Make sure the OpenAPI client library is properly installed."
    )

logger = logging.getLogger(__name__)

_TRANSPORT_ERRORS = (ApiException, httpx.RequestError)


class AsyncEncryptedIndex:
    """
    Provides async access to an encrypted vector index via the REST API.

    This class handles operations on an encrypted vector index, including
    adding/updating vectors, searching, and managing index metadata. Every
    method that talks to the service is a coroutine and must be awaited.

    Differences from :class:`cyborgdb.EncryptedIndex`:

    - The describe-backed attributes ``dimension``, ``metric``, ``n_lists``,
      ``metadata_schema`` and ``bm25`` are properties on the sync class but
      **async methods** here, because a property cannot be awaited::

          dim = index.dimension        # sync
          dim = await index.dimension()  # async

      They cache exactly as the sync properties do: ``dimension`` is cached
      once non-zero and ``metric`` on first read, while ``n_lists``,
      ``metadata_schema`` and ``bm25`` are fetched on every call.
      ``index_name`` needs no request and stays a plain property.
    - The index shares its connection pool with the :class:`AsyncClient` that
      created it. ``close()`` (or leaving an ``async with`` block) closes that
      shared pool, so afterwards neither the index nor the client can make
      requests. Close the client once when you are done with all of its
      indexes; close an index directly only when it is the last user of the
      client.
    """

    def __init__(
        self,
        index_name: str,
        index_key: Optional[bytes],
        api: DefaultApi,
        api_client: ApiClient,
    ):
        """
        Initialize with API access to an index.

        Args:
            index_name: Name of the index
            index_key: Encryption key for the index. ``None`` for KMS-backed
                indexes where the service resolves the KEK from the stored
                ``KMSBlob``.
            api: API client instance
            api_client: The lower-level API client
        """
        self._index_name = index_name
        self._index_key = index_key
        self._index_key_hex = (
            binascii.hexlify(index_key).decode("ascii")
            if index_key is not None
            else None
        )
        self._api = api
        self._api_client = api_client
        # Lazy-cached describe-derived metadata. `metric` is immutable
        # post-creation. `dimension` is too once set, but an auto-dimension
        # index reports 0 until its first upsert, so 0 is never cached.
        # `n_lists` is fetched fresh on every read because training
        # mutates it (default 1 → trained cluster count).
        self._dimension: Optional[int] = None
        self._metric: Optional[str] = None

    async def close(self) -> None:
        """Close the connection pool shared with the owning ``AsyncClient``."""
        await self._api_client.close()

    async def __aenter__(self) -> "AsyncEncryptedIndex":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def _describe(self):
        """Fire the describe endpoint with this index's key (None for
        KMS-backed indexes) and refresh the cached immutable fields.
        Raises the raw ``ApiException`` so `AsyncClient.load_index` can
        translate it with its own context."""
        response = await self._api.get_index_info_v1_indexes_describe_post(
            index_operation_request=self._ior()
        )
        if response.dimension:
            self._dimension = response.dimension
        self._metric = response.metric
        return response

    async def _describe_translated(self):
        try:
            return await self._describe()
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to describe index") from e

    def _request_headers(self) -> Dict[str, str]:
        """Build the request headers for data-path calls. Only includes
        ``X-API-Key`` when one is configured; when the service has auth
        disabled (no ``CYBORGDB_SERVICE_ROOT_KEY`` set) the SDK can be
        constructed with no api_key and we must not send an empty
        header (and must not crash indexing into an empty config dict)."""
        return request_headers(self._api_client.configuration.api_key.get("X-API-Key"))

    @property
    def index_name(self) -> str:
        """Get the name of the index."""
        return self._index_name

    async def dimension(self) -> int:
        """Vector dimensionality. `0` if create_index was called
        without an explicit dimension and the first upsert hasn't
        happened yet; otherwise the real dimension. Cached once
        non-zero."""
        if self._dimension is None:
            return (await self._describe_translated()).dimension
        return self._dimension

    async def metric(self) -> str:
        """Distance metric (`euclidean`, `cosine`, or
        `squared_euclidean`). Cached on first read."""
        if self._metric is None:
            await self._describe_translated()
        return self._metric

    async def n_lists(self) -> int:
        """Number of inverted lists. `1` for untrained indexes; set to
        the trained cluster count after `train()`. Fetched fresh on
        every read so post-training callers see the new value."""
        return (await self._describe_translated()).n_lists

    async def metadata_schema(self) -> Dict[str, Dict[str, bool]]:
        """Per-field metadata indexing policy recorded at create time, as
        `{field: {"filterable": bool, "pattern": bool, "full_text": bool}}`.
        Empty dict when the index uses the default index-everything posture.
        Immutable, but not cached: an older service omits the field entirely,
        and normalizing that `None` to `{}` here keeps callers from having to.

        `full_text` marks a field routed through the BM25 analyzer (searchable
        by `query`/`query_metadata` with `text=...`); see `bm25` for the
        scorer config.

        Returned as plain dicts — the same shape `create_index` accepts — so
        generated openapi_client models never leak out of the wrapper."""
        return {
            field: {
                "filterable": policy.filterable,
                "pattern": policy.pattern,
                "full_text": policy.full_text,
            }
            for field, policy in (
                (await self._describe_translated()).metadata_schema or {}
            ).items()
        }

    async def bm25(self) -> Optional[Dict[str, Any]]:
        """BM25 scorer config the index reports back, as
        `{"k1": float, "b": float, "analyzer_version": str | None}`, or `None`
        when the index has no `full_text` field (BM25 is opt-in and derived,
        never flagged). `k1`/`b` are the tuning parameters supplied at create
        time or their defaults; `analyzer_version` identifies the tokenizer /
        stemmer pipeline the corpus was indexed with.

        Returned as a plain dict so generated openapi_client models never leak
        out of the wrapper."""
        config = (await self._describe_translated()).bm25
        if config is None:
            return None
        return {
            "k1": config.k1,
            "b": config.b,
            "analyzer_version": config.analyzer_version,
        }

    async def is_trained(self) -> bool:
        """
        Check if the index has been trained.

        Returns:
            bool: True if the index is trained, otherwise False.

        Raises:
            CyborgDBError: If the training status could not be retrieved.
        """
        return (await self._describe_translated()).is_trained

    async def delete_index(self) -> None:
        """
        Delete the current index and all its associated data.

        Warning:
            This action is irreversible.

        Raises:
            CyborgDBError: If the index could not be deleted.
        """
        try:
            await self._api.delete_index_v1_indexes_delete_post(
                index_operation_request=self._ior()
            )
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to delete index") from e

    async def get(
        self, ids: List[str], include: List[str] = ["vector", "contents", "metadata"]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and decrypt items associated with the specified IDs.

        Args:
            ids: IDs to retrieve.
            include: Item fields to return. Can include 'vector', 'contents', and 'metadata'.
                Default is ['vector', 'contents', 'metadata'].

        Returns:
            A list of dictionaries representing the items with the requested fields.
            IDs will always be included in the returned items.

        Raises:
            CyborgDBError: If the items could not be retrieved or decrypted.
        """
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
            return parse_get_response(response, include)
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to retrieve items") from e
        except Exception as e:
            error_msg = f"Get operation failed: {str(e)}"
            logger.error(error_msg)
            raise

    async def train(
        self,
        n_lists: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_iters: Optional[int] = None,
        tolerance: Optional[float] = None,
        max_memory: Optional[int] = None,
    ) -> None:
        """
        Build the index using the specified training configuration.

        Prior to calling this, all queries will be conducted using encrypted exhaustive search.
        After, they will be conducted using encrypted ANN search.

        Args:
            n_lists: Number of inverted lists for the index. Default is auto.
            batch_size: Size of each batch for training. Default is 2048.
            max_iters: Maximum iterations for training. Default is 100.
            tolerance: Convergence tolerance for training. Default is 1e-6.
            max_memory: Maximum memory (MB) used during training. Default is 0
                (no limit).

        Note:
            There must be at least 2 * n_lists vector embeddings in the index prior to calling
            this function.

        Raises:
            CyborgDBError: If the service rejects the training request.
        """
        try:
            request = TrainRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                n_lists=n_lists,
                batch_size=batch_size,
                max_iters=max_iters,
                tolerance=tolerance,
                max_memory=max_memory,
            )

            await self._api.train_index_v1_indexes_train_post(train_request=request)
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to train index") from e

    async def upsert(
        self,
        arg1: Union[List[Dict[str, Any]], List[str], np.ndarray],
        arg2: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add or update vector embeddings in the index.

        If an item already exists at the specified ID, it will be overwritten.

        This method can be called in one of two ways:
        1. With a list of dictionaries, each containing 'id', 'vector', and optional 'contents'
        and 'metadata'.
        - If the index was created with an embedding model and 'vector' is not provided,
            'contents' will be automatically embedded.
        2. With separate IDs and vectors arrays (automatically uses efficient binary format).

        Args:
            arg1: Either a list of dictionaries or a list/array of IDs.
            arg2: If arg1 is a list of IDs, this should be an array of vector embeddings.

        Raises:
            ValidationError: If the arguments are malformed (wrong types, a missing
                ``id``, or mismatched ID and vector counts). A wrong-type error is
                also a ``TypeError``.
            CyborgDBError: If the service rejects the upsert.
        """
        # Case 2: arg1 is a list of IDs, arg2 is a numpy array -> use binary format
        if arg2 is not None and isinstance(arg2, np.ndarray):
            if not isinstance(arg1, list):
                raise _ArgumentTypeError("arg1 must be a list of IDs")
            ids = [str(id_val) for id_val in arg1]
            await self.upsert_binary(ids, arg2)
            return

        try:
            request = build_upsert_request(
                _models, self._index_name, self._key_to_hex(), arg1, arg2
            )
            await self._api.upsert_vectors_v1_vectors_upsert_post(
                upsert_request=request,
                _headers=self._request_headers(),
            )

        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to upsert items") from e
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise

    async def upsert_binary(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        contents: Optional[List[Optional[Union[str, bytes]]]] = None,
    ) -> None:
        """
        Add or update vector embeddings using binary format for efficiency.

        This method is optimized for large batches. Vectors are sent as base64-encoded
        binary data instead of JSON arrays, which can be significantly faster for large datasets.

        Args:
            ids: List of unique identifiers for each vector.
            vectors: NumPy array of shape (n_vectors, dimension) with dtype float32.
            metadata: Optional list of metadata dicts for each vector.
            contents: Optional list of contents for each vector.

        Raises:
            ValidationError: If ``vectors`` is not a 2D numpy array (a wrong type
                is also a ``TypeError``), or if ``vectors``, ``metadata`` or
                ``contents`` don't match ``ids`` in length.
            CyborgDBError: If the service rejects the upsert.
        """
        request = build_binary_upsert_request(
            _models,
            self._index_name,
            self._key_to_hex(),
            ids,
            vectors,
            metadata,
            contents,
        )

        try:
            await self._api.upsert_vectors_binary_v1_vectors_upsert_binary_post(
                binary_upsert_request=request,
                _headers=self._request_headers(),
            )
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to upsert items (binary)") from e

    async def delete(self, ids: List[str]) -> None:
        """
        Delete the specified encrypted items stored in the index.

        Removes all associated fields (vector, contents, metadata) for the given IDs.

        Warning:
            This action is irreversible.

        Args:
            ids: IDs to delete.

        Raises:
            CyborgDBError: If the items could not be deleted.
        """
        try:
            delete_request = DeleteRequest(
                index_key=self._key_to_hex(), index_name=self._index_name, ids=ids
            )
            await self._api.delete_vectors_v1_vectors_delete_post(
                delete_request=delete_request
            )
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to delete items") from e

    async def query(
        self,
        query_vectors: Optional[
            Union[np.ndarray, List[List[float]], List[float]]
        ] = None,
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
        """
        Retrieve the nearest neighbors for given query vectors.
        Supports both single vector (1D) and batched vectors (2D).

        For batch queries with 2D numpy arrays, automatically uses efficient
        binary format for faster transfer.

        Passing ``text`` turns this into a hybrid (BM25 + vector) query against
        an index with at least one ``full_text`` field; a query vector is still
        required. Hybrid results carry a fused ``score`` (larger = more
        relevant) instead of ``distance``. The remaining knobs tune the text
        leg and its fusion with the vector leg:

        Args:
            text: Query text for the BM25 leg. Omitted/empty leaves the query
                text-free (pure vector search).
            text_fields: ``full_text`` fields the text leg searches; omitted
                means all of them. Naming a non-full-text field raises.
            text_field_weights: Per-field weights on the summed per-field BM25
                scores, parallel to the searched fields. Omitted means 1.0 each.
            require_all_terms: Require every query term to match (AND) instead
                of any (OR, the default).
            alpha: Leg blend in ``[0, 1]``: 0 = pure BM25, 1 = pure vector;
                omitted means 0.5.
            rrf_k: RRF rank-smoothing constant (> 0; omitted means 60).
            window_mult: Per-leg candidate depth as a multiple of ``top_k``
                (>= 1; omitted means 3).
        """
        hybrid_kwargs = hybrid_query_kwargs(
            text,
            text_fields,
            text_field_weights,
            require_all_terms,
            alpha,
            rrf_k,
            window_mult,
        )
        try:
            if isinstance(query_vectors, np.ndarray):
                if query_vectors.ndim == 1 or query_vectors.ndim == 2:
                    # NumPy arrays (1D or 2D) -> use binary format for efficiency
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
                raise ValidationError(
                    "Expected 1D or 2D NumPy array for `query_vectors`."
                )

            request = build_query_request(
                _models,
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

            try:
                raw_response = await self._api.query_vectors_v1_vectors_query_post_without_preload_content(
                    request=request,
                    _headers=self._request_headers(),
                )
                body = await raw_response.aread()

                # _without_preload_content skips status validation, so surface
                # 4xx/5xx (e.g. an RBAC 403) instead of parsing the error body.
                if not 200 <= raw_response.status_code <= 299:
                    raise ApiException.from_response(
                        http_resp=RESTResponse(raw_response),
                        body=body.decode("utf-8"),
                        data=None,
                    )

                response_json = json.loads(body.decode("utf-8"))
                return parse_query_response(response_json, include)
            except Exception as e:
                error_msg = f"Unexpected error in query: {str(e)}"
                logger.error(error_msg)
                import traceback

                logger.error(traceback.format_exc())
                raise
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Query failed") from e
        except Exception as e:
            error_msg = f"Unexpected error in query: {str(e)}"
            logger.error(error_msg)
            import traceback

            logger.error(traceback.format_exc())
            raise

    async def query_binary(
        self,
        query_vectors: np.ndarray,
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
        """
        Retrieve the nearest neighbors for given query vectors using binary format.

        This method is optimized for large batch queries. Query vectors are sent as
        base64-encoded binary data instead of JSON arrays, which is more efficient.

        Args:
            query_vectors: NumPy array of shape (dimension,) for single query or
                (n_queries, dimension) for batch queries, with dtype float32.
            top_k: Number of nearest neighbors to return for each query.
            n_probes: Number of lists to probe during the query.
            filters: Dictionary specifying metadata filters.
            include: List of fields to include in the response.
            greedy: Whether to use greedy search.
            rerank_mult: Multiplier for stage 1 retrieval in reranking indexes.
            text: Query text for a BM25 leg (hybrid search). See
                :meth:`query` for the text-leg / fusion knobs below.
            text_fields: ``full_text`` fields the text leg searches.
            text_field_weights: Per-field weights, parallel to the searched fields.
            require_all_terms: Require every query term to match (AND vs OR).
            alpha: Leg blend in ``[0, 1]`` (0 = BM25, 1 = vector; default 0.5).
            rrf_k: RRF rank-smoothing constant (> 0; default 60).
            window_mult: Per-leg candidate depth as a multiple of ``top_k``.

        Returns:
            For single query (1D input): List of result dictionaries.
            For batch query (2D input): List of lists of result dictionaries, one list per query vector.

        Raises:
            ValidationError: If ``query_vectors`` is not a 1D or 2D numpy array
                (a wrong type is also a ``TypeError``).
            CyborgDBError: If the service rejects the query.
        """
        request, is_single_query = build_binary_query_request(
            _models,
            self._index_name,
            self._key_to_hex(),
            query_vectors,
            top_k,
            n_probes,
            filters,
            include,
            greedy,
            rerank_mult,
            hybrid_query_kwargs(
                text,
                text_fields,
                text_field_weights,
                require_all_terms,
                alpha,
                rrf_k,
                window_mult,
            ),
        )

        try:
            response = (
                await self._api.query_vectors_binary_v1_vectors_query_binary_post(
                    binary_query_request=request,
                    _headers=self._request_headers(),
                )
            )
            return parse_binary_query_response(response, is_single_query)

        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to query (binary)") from e

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
    ) -> List[MetadataResult]:
        """
        Find items by metadata alone — no query vector, no distances.

        Resolves ``filters`` entirely against the encrypted metadata index and
        returns the matching items as :class:`MetadataResult` rows (``{"id"}``,
        matching core's ``list[MetadataResult]``). Works on untrained indexes.

        Unlike :meth:`query`, there is no post-filter stage to fall back on, so
        the index's ``metadata_schema`` is enforced rather than advisory:
        ``$regex``/``$contains`` require a ``pattern`` field, and a field
        declared ``filterable=False`` cannot be filtered on at all. Both raise
        ``ValidationError``. Use :meth:`query` with a vector for those.

        Passing ``text`` adds a BM25 full-text leg (requires an index with at
        least one ``full_text`` field). Results are then ranked by relevance
        and each row also carries a ``score`` (``{"id", "score"}``, descending
        score); ``filters`` given alongside acts as a pre-filter and
        ``order_by`` is not supported with ``text``.

        Args:
            filters: Metadata filters; ``None``/empty matches everything.
            top_k: Cap on results returned, applied AFTER ``order_by``.
                ``None`` returns every match.
            order_by: Field to sort matches by, either a name or a MongoDB-style
                single-field dict (``{"views": -1}``, which also sets the
                direction). Unordered when omitted. Not supported with ``text``.
            ascending: Sort direction, when ``order_by`` is a plain field name.
            text: Query text for the BM25 leg. Omitted/empty keeps this a
                filter-only query returning ``{"id"}`` rows (no score).
            text_fields: ``full_text`` fields the text leg searches; omitted
                means all of them. Naming a non-full-text field raises.
            text_field_weights: Per-field weights on the summed per-field BM25
                scores, parallel to the searched fields. Omitted means 1.0 each.
            require_all_terms: Require every query term to match (AND) instead
                of any (OR, the default).

        Returns:
            A list of ``{"id"}`` dicts (``list[MetadataResult]``, matching
            core). Without ``text``: ordered when ``order_by`` was given, no
            ``score`` key. With ``text``: each row also carries ``score``,
            ranked by descending BM25 score.

        Raises:
            ValidationError: If the filter cannot be resolved from the metadata
                index, or ``order_by`` is malformed.
        """
        request = build_query_metadata_request(
            _models,
            self._index_name,
            self._key_to_hex(),
            filters,
            top_k,
            order_by,
            ascending,
            text,
            text_fields,
            text_field_weights,
            require_all_terms,
        )

        try:
            response = await self._api.query_metadata_v1_vectors_query_metadata_post(
                query_metadata_request=request
            )
            return parse_query_metadata_response(response, text)
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to query metadata") from e

    async def list_ids(self) -> List[str]:
        """
        List all document IDs in the index.

        Returns:
            List of document IDs.
        """
        try:
            list_ids_request = ListIDsRequest(
                index_key=self._key_to_hex(), index_name=self._index_name
            )
            response = await self._api.list_ids_v1_vectors_list_ids_post(
                list_ids_request=list_ids_request
            )

            return response.ids
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to list document IDs") from e

    async def is_training(self) -> bool:
        """
        Get the current training status of the index.

        Returns:
            A dictionary containing training status information.
        """
        try:
            response = (
                await self._api.get_training_status_v1_indexes_training_status_get()
            )

            if self._index_name in response.training_indexes:
                return True

            return False

        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to get index training status") from e

    # RBAC — user management (root API key required). See EncryptedIndex for
    # the model: a user is scoped to this index, and the wrapped keys that
    # exist for them *are* their permission set.

    async def create_user(self, permissions: List[str]) -> Dict[str, str]:
        """Mint a user API key scoped to this index.

        Args:
            permissions: Non-empty subset of ``{"read", "write"}``. The
                grant is enforced cryptographically by the service, not by
                a checked policy field.

        Returns:
            ``{"user_id": "<hex>", "api_key": "cdbk_..."}``. The ``api_key``
            is returned **exactly once** and is never stored by the
            service — capture it now, it cannot be recovered. Hand it to
            the user; they authenticate by passing it as ``api_key`` to
            ``AsyncClient`` and need no index key of their own.

        Raises:
            AuthenticationError: If the client is not using the root key.
            ValidationError: If ``permissions`` is invalid.
        """
        # SDK-supplied-KEK indexes: the service needs the index key to
        # unwrap the root DEK and re-wrap it under the new user's key.
        # KMS-backed indexes resolve it server-side, so index_key is None.
        request = CreateUserRequest(
            permissions=permissions, index_key=self._index_key_hex
        )
        try:
            response = await self._api.create_user_v1_indexes_index_name_users_post(
                index_name=self._index_name, create_user_request=request
            )
            return {"user_id": response.user_id, "api_key": response.api_key}
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to create user") from e

    async def list_users(self) -> List[Dict[str, Any]]:
        """List the users provisioned for this index.

        Returns:
            A list of ``{"user_id": "<hex>", "permissions": [...]}`` dicts.
            Permissions are derived from which wrapped keys exist for each
            user (the cryptographic source of truth), not a stored field.

        Raises:
            AuthenticationError: If the client is not using the root key.
            CyborgDBError: If the users could not be listed for another reason.
        """
        try:
            response = await self._api.list_users_v1_indexes_index_name_users_get(
                index_name=self._index_name, x_index_key=self._index_key_hex
            )
            return [
                {"user_id": u.user_id, "permissions": u.permissions}
                for u in response.users
            ]
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to list users") from e

    async def delete_user(self, user_id: str) -> None:
        """Revoke a user, erasing their wrapped keys for this index.

        After this returns, the user's API key is rejected on the next
        request — the service can no longer unwrap any key for them.

        Args:
            user_id: The hex ``user_id`` returned by ``create_user`` (also
                surfaced by ``list_users``).

        Raises:
            CyborgDBError: If the user could not be deleted.
        """
        try:
            await self._api.delete_user_v1_indexes_index_name_users_user_id_delete(
                index_name=self._index_name,
                user_id=user_id,
                x_index_key=self._index_key_hex,
            )
        except _TRANSPORT_ERRORS as e:
            raise translate_api_error(e, "Failed to delete user") from e

    def _key_to_hex(self) -> Optional[str]:
        """Hex-encoded key for API calls, or ``None`` for KMS-backed indexes.
        Computed once in ``__init__`` since the key never changes."""
        return self._index_key_hex

    def _ior(self) -> IndexOperationRequest:
        """Build the name+key request used by describe/delete-style endpoints."""
        return IndexOperationRequest(
            index_key=self._key_to_hex(), index_name=self._index_name
        )
