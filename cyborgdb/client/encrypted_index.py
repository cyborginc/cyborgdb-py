"""
EncryptedIndex class for CyborgDB

This module provides the EncryptedIndex class for interacting with encrypted vector indexes in CyborgDB.
"""

import base64
import binascii
import json
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np

# Import the OpenAPI generated client
try:
    from cyborgdb.openapi_client.api_client import ApiClient
    from cyborgdb.openapi_client.api.default_api import DefaultApi
    from cyborgdb.openapi_client.models.train_request import TrainRequest
    from cyborgdb.openapi_client.models.delete_request import DeleteRequest
    from cyborgdb.openapi_client.models.batch_query_request import BatchQueryRequest
    from cyborgdb.openapi_client.models.index_operation_request import (
        IndexOperationRequest,
    )
    from cyborgdb.openapi_client.exceptions import ApiException
    from cyborgdb.openapi_client.models.query_request import QueryRequest
    from cyborgdb.openapi_client.models.list_ids_request import ListIDsRequest
    from cyborgdb.openapi_client.models.query_metadata_request import (
        QueryMetadataRequest,
    )
    from cyborgdb.openapi_client.models.request import Request
    from cyborgdb.openapi_client.models import Contents
    from cyborgdb.openapi_client.models.binary_upsert_request import BinaryUpsertRequest
    from cyborgdb.openapi_client.models.binary_vector_batch import BinaryVectorBatch
    from cyborgdb.openapi_client.models.binary_query_request import BinaryQueryRequest
    from cyborgdb.openapi_client.models.binary_query_batch import BinaryQueryBatch
    from cyborgdb.openapi_client.models.create_user_request import CreateUserRequest
except ImportError:
    raise ImportError(
        "Failed to import openapi_client. Make sure the OpenAPI client library is properly installed."
    )

logger = logging.getLogger(__name__)


class EncryptedIndex:
    """
    Provides access to an encrypted vector index via the REST API.

    This class handles operations on an encrypted vector index, including
    adding/updating vectors, searching, and managing index metadata.
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
        # Lazy-cached describe-derived metadata. `dimension` and `metric`
        # are immutable post-creation, so the first describe populates
        # both and we reuse the values. `n_lists` is fetched fresh on
        # every read because training mutates it (default 1 → trained
        # cluster count).
        self._dimension: Optional[int] = None
        self._metric: Optional[str] = None

    def _describe(self):
        """Fire the describe endpoint with this index's key (None for
        KMS-backed indexes). Shared by the lazy property accessors and
        by `Client.load_index`'s existence probe."""
        return self._api.get_index_info_v1_indexes_describe_post(
            index_operation_request=self._ior()
        )

    def _request_headers(self) -> Dict[str, str]:
        """Build the request headers for data-path calls. Only includes
        ``X-API-Key`` when one is configured; when the service has auth
        disabled (no ``CYBORGDB_SERVICE_ROOT_KEY`` set) the SDK can be
        constructed with no api_key and we must not send an empty
        header (and must not crash indexing into an empty config dict)."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = self._api_client.configuration.api_key.get("X-API-Key")
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    @property
    def index_name(self) -> str:
        """Get the name of the index."""
        return self._index_name

    @property
    def dimension(self) -> int:
        """Vector dimensionality. `0` if create_index was called
        without an explicit dimension and the first upsert hasn't
        happened yet; otherwise the real dimension. Cached on first
        read."""
        if self._dimension is None:
            response = self._describe()
            self._dimension = response.dimension
            self._metric = response.metric
        return self._dimension

    @property
    def metric(self) -> str:
        """Distance metric (`euclidean`, `cosine`, or
        `squared_euclidean`). Cached on first read."""
        if self._metric is None:
            response = self._describe()
            self._dimension = response.dimension
            self._metric = response.metric
        return self._metric

    @property
    def n_lists(self) -> int:
        """Number of inverted lists. `1` for untrained indexes; set to
        the trained cluster count after `train()`. Fetched fresh on
        every read so post-training callers see the new value."""
        return self._describe().n_lists

    @property
    def metadata_schema(self) -> Dict[str, Dict[str, bool]]:
        """Per-field metadata indexing policy recorded at create time, as
        `{field: {"filterable": bool, "pattern": bool}}`. Empty dict when the
        index uses the default index-everything posture. Immutable, but not
        cached: an older service omits the field entirely, and normalizing
        that `None` to `{}` here keeps callers from having to.

        Returned as plain dicts — the same shape `create_index` accepts — so
        generated openapi_client models never leak out of the wrapper."""
        return {
            field: {"filterable": policy.filterable, "pattern": policy.pattern}
            for field, policy in (self._describe().metadata_schema or {}).items()
        }

    def is_trained(self) -> bool:
        """
        Check if the index has been trained.

        Returns:
            bool: True if the index is trained, otherwise False.
        """
        try:
            response = self._api.get_index_info_v1_indexes_describe_post(
                index_operation_request=self._ior()
            )
            return response.is_trained
        except ApiException as e:
            logger.error(f"Failed to get index training status: {e}")
            return False

    def delete_index(self) -> None:
        """
        Delete the current index and all its associated data.

        Warning:
            This action is irreversible.

        Raises:
            ValueError: If the index could not be deleted.
        """
        try:
            self._api.delete_index_v1_indexes_delete_post(
                index_operation_request=self._ior()
            )
        except ApiException as e:
            error_msg = f"Failed to delete index: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get(
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
            ValueError: If the items could not be retrieved or decrypted.
        """
        try:
            from cyborgdb.openapi_client.models import GetRequest

            # Create the proper request objects
            get_request = GetRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                ids=ids,
                include=include,
            )
            response = self._api.get_vectors_v1_vectors_get_post(
                get_request=get_request,
                _headers=self._request_headers(),
            )

            # Convert API response to our format
            items = []
            if hasattr(response, "results"):
                for item in response.results:
                    item_dict = {"id": item.id}

                    if "vector" in include and hasattr(item, "vector"):
                        item_dict["vector"] = item.vector

                    if "contents" in include and hasattr(item, "contents"):
                        item_dict["contents"] = item.contents

                    if "metadata" in include and hasattr(item, "metadata"):
                        # Convert metadata string to dict if needed
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
            error_msg = f"Get operation failed: {str(e)}"
            logger.error(error_msg)
            raise
        except ApiException as e:
            error_msg = f"Failed to retrieve items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def train(
        self,
        n_lists: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_iters: Optional[int] = None,
        tolerance: Optional[float] = None,
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

        Note:
            There must be at least 2 * n_lists vector embeddings in the index prior to calling
            this function.

        Raises:
            ValueError: If there are not enough vector embeddings in the index for training,
                or if the index could not be trained.
        """
        try:
            request = TrainRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                n_lists=n_lists,
                batch_size=batch_size,
                max_iters=max_iters,
                tolerance=tolerance,
            )

            self._api.train_index_v1_indexes_train_post(train_request=request)
        except ApiException as e:
            error_msg = f"Failed to train index: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def upsert(
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
            ValueError: If vector dimensions are incompatible with the index configuration,
                if index was not created or loaded yet, if there is a mismatch between
                the number of vectors and IDs, or if the vectors could not be upserted.
            TypeError: If the arguments do not match expected types.
        """
        # Case 2: arg1 is a list of IDs, arg2 is a numpy array -> use binary format
        if arg2 is not None and isinstance(arg2, np.ndarray):
            if not isinstance(arg1, list):
                raise TypeError("arg1 must be a list of IDs")
            # Convert IDs to strings if needed
            ids = [str(id_val) for id_val in arg1]
            # Use binary upsert for efficiency
            self.upsert_binary(ids, arg2)
            return

        try:
            items = []

            # Case 1: arg1 is a list of dictionaries
            if arg2 is None:
                if not isinstance(arg1, list) or not all(
                    isinstance(item, dict) for item in arg1
                ):
                    raise TypeError(
                        "When arg2 is None, arg1 must be a list of dictionaries"
                    )

                # Convert each dict to an Item
                for item_dict in arg1:
                    if "id" not in item_dict:
                        raise ValueError(
                            "Each item dictionary must contain an 'id' field"
                        )

                    item = {"id": item_dict["id"]}

                    if "vector" in item_dict:
                        vec = item_dict["vector"]
                        # Normalize to float32 so JSON serialization matches binary path
                        if isinstance(vec, np.ndarray):
                            item["vector"] = vec.astype(np.float32).tolist()
                        elif isinstance(vec, list):
                            item["vector"] = np.array(vec, dtype=np.float32).tolist()
                        else:
                            item["vector"] = vec

                    if "contents" in item_dict:
                        contents_value = item_dict["contents"]

                        # Convert bytes to base64 string for JSON serialization
                        if isinstance(contents_value, bytes):
                            # Convert bytes to base64 string
                            contents_value = base64.b64encode(contents_value).decode(
                                "utf-8"
                            )
                        elif isinstance(contents_value, bytearray):
                            # Convert bytearray to base64 string
                            contents_value = base64.b64encode(
                                bytes(contents_value)
                            ).decode("utf-8")
                        # If it's already a string, use as-is

                        # Contents model accepts string or bytearray
                        item["contents"] = Contents(contents_value)

                    if "metadata" in item_dict:
                        # Convert dict metadata to JSON string if needed
                        if isinstance(item_dict["metadata"], dict):
                            item["metadata"] = item_dict[
                                "metadata"
                            ]  # json.dumps(item_dict["metadata"])
                        else:
                            item["metadata"] = item_dict["metadata"]

                    items.append(item)

            # Case 2: arg1 is a list of IDs, arg2 is a list of vectors (non-numpy)
            else:
                if not isinstance(arg1, list):
                    raise TypeError("arg1 must be a list of IDs")

                vectors = arg2
                if len(arg1) != len(vectors):
                    raise ValueError("Number of IDs must match number of vectors")

                # Create items from IDs and vectors
                for id_val, vector in zip(arg1, vectors):
                    # Normalize to float32 so JSON serialization matches binary path
                    if isinstance(vector, np.ndarray):
                        vector = vector.astype(np.float32).tolist()
                    elif isinstance(vector, list):
                        vector = np.array(vector, dtype=np.float32).tolist()
                    items.append({"id": str(id_val), "vector": vector})

            # Import the UpsertRequest model from the OpenAPI-generated code
            from cyborgdb.openapi_client.models import UpsertRequest

            # Create the upsert request with all required fields
            request = UpsertRequest(
                items=items, index_key=self._key_to_hex(), index_name=self._index_name
            )

            # Make the API call with the correct parameter
            self._api.upsert_vectors_v1_vectors_upsert_post(
                upsert_request=request,  # This is the only required parameter
                _headers=self._request_headers(),
            )

        except ApiException as e:
            error_msg = f"Failed to upsert items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise

    def upsert_binary(
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
            ValueError: If vectors shape doesn't match ids length, or if upsert fails.
            TypeError: If vectors is not a numpy array.
        """
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a numpy array")

        if vectors.ndim != 2:
            raise ValueError(
                "vectors must be a 2D array of shape (n_vectors, dimension)"
            )

        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of ids ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
            )

        # Ensure little-endian float32 dtype for cross-platform binary compatibility
        if vectors.dtype != np.dtype("<f4"):
            vectors = vectors.astype("<f4")

        # Encode vectors as base64
        vectors_b64 = base64.b64encode(vectors.tobytes()).decode("ascii")

        # Build the request using generated models
        batch = BinaryVectorBatch(
            ids=ids,
            vectors_b64=vectors_b64,
            dimension=vectors.shape[1],
            metadata=metadata,
            contents=contents,
        )

        request = BinaryUpsertRequest(
            index_name=self._index_name,
            index_key=self._key_to_hex(),
            batch=batch,
        )

        try:
            self._api.upsert_vectors_binary_v1_vectors_upsert_binary_post(
                binary_upsert_request=request,
                _headers=self._request_headers(),
            )
        except ApiException as e:
            error_msg = f"Failed to upsert items (binary): {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def delete(self, ids: List[str]) -> None:
        """
        Delete the specified encrypted items stored in the index.

        Removes all associated fields (vector, contents, metadata) for the given IDs.

        Warning:
            This action is irreversible.

        Args:
            ids: IDs to delete.

        Raises:
            ValueError: If the items could not be deleted.
        """
        try:
            delete_request = DeleteRequest(
                index_key=self._key_to_hex(), index_name=self._index_name, ids=ids
            )
            self._api.delete_vectors_v1_vectors_delete_post(
                delete_request=delete_request
            )
        except ApiException as e:
            error_msg = f"Failed to delete items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def query(
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
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Retrieve the nearest neighbors for given query vectors.
        Supports both single vector (1D) and batched vectors (2D).

        For batch queries with 2D numpy arrays, automatically uses efficient
        binary format for faster transfer.
        """
        try:
            # Determine the correct vector input
            vector_list = None
            is_single_query = False

            if query_vectors is not None:
                if isinstance(query_vectors, np.ndarray):
                    if query_vectors.ndim == 1 or query_vectors.ndim == 2:
                        # NumPy arrays (1D or 2D) -> use binary format for efficiency
                        return self.query_binary(
                            query_vectors=query_vectors,
                            top_k=top_k,
                            n_probes=n_probes,
                            filters=filters,
                            include=include,
                            greedy=greedy,
                            rerank_mult=rerank_mult,
                        )
                    else:
                        raise ValueError(
                            "Expected 1D or 2D NumPy array for `query_vectors`."
                        )
                elif isinstance(query_vectors, list):
                    if not query_vectors:
                        raise ValueError("Empty list provided for `query_vectors`.")
                    if isinstance(query_vectors[0], (list, np.ndarray)):
                        # Batch of vectors as list of lists
                        # Normalize to float32 so JSON serialization matches binary path
                        vector_list = [
                            np.array(v, dtype=np.float32).tolist()
                            for v in query_vectors
                        ]
                    else:
                        # Single vector as flat list
                        # Normalize to float32 so JSON serialization matches binary path
                        is_single_query = True
                        vector_list = np.array(query_vectors, dtype=np.float32).tolist()
                else:
                    raise ValueError("Invalid type for `query_vectors`")

            if is_single_query or query_contents is not None:
                # Use QueryRequest for single vector or content-based query
                # Build kwargs to avoid passing None values (which would be serialized)
                query_kwargs = {
                    "index_key": self._key_to_hex(),
                    "index_name": self._index_name,
                    "query_vectors": vector_list,
                }
                if query_contents is not None:
                    query_kwargs["query_contents"] = query_contents
                if top_k is not None:
                    query_kwargs["top_k"] = top_k
                if n_probes is not None:
                    query_kwargs["n_probes"] = n_probes
                if greedy is not None:
                    query_kwargs["greedy"] = greedy
                if rerank_mult is not None:
                    query_kwargs["rerank_mult"] = rerank_mult
                if filters is not None:
                    query_kwargs["filters"] = filters
                if include is not None:
                    query_kwargs["include"] = include
                query_request = QueryRequest(**query_kwargs)
            else:
                # Use BatchQueryRequest for multiple vectors
                # Build kwargs to avoid passing None values (which would be serialized)
                query_kwargs = {
                    "index_key": self._key_to_hex(),
                    "index_name": self._index_name,
                    "query_vectors": vector_list,
                }
                if top_k is not None:
                    query_kwargs["top_k"] = top_k
                if n_probes is not None:
                    query_kwargs["n_probes"] = n_probes
                if greedy is not None:
                    query_kwargs["greedy"] = greedy
                if rerank_mult is not None:
                    query_kwargs["rerank_mult"] = rerank_mult
                if filters is not None:
                    query_kwargs["filters"] = filters
                if include is not None:
                    query_kwargs["include"] = include
                query_request = BatchQueryRequest(**query_kwargs)

            request = Request(query_request)

            # Execute query via REST
            try:
                # Get raw response instead of deserialized object
                raw_response = self._api.query_vectors_v1_vectors_query_post_without_preload_content(
                    request=request,
                    _headers=self._request_headers(),
                )

                # _without_preload_content skips status validation, so surface
                # 4xx/5xx (e.g. an RBAC 403) instead of parsing the error body.
                if not 200 <= raw_response.status <= 299:
                    raise ApiException.from_response(
                        http_resp=raw_response,
                        body=raw_response.data.decode("utf-8"),
                        data=None,
                    )

                # Parse raw JSON response manually
                response_text = raw_response.data.decode("utf-8")
                response_json = json.loads(response_text)

                # Determine include filtering strategy
                include_all = (
                    include is None
                )  # None means include everything server returns
                include_set = set(include) if include else set()

                # Process the results as plain dictionaries
                results = []
                if "results" in response_json:
                    # Check if the results is a list of lists or just a list
                    if response_json["results"] and isinstance(
                        response_json["results"][0], list
                    ):
                        # It's a list of lists (batch query results)
                        for query_result in response_json["results"]:
                            query_items = []
                            for item in query_result:
                                result_item = {"id": item["id"]}

                                # Always include distance if present (core part of query results)
                                if "distance" in item:
                                    result_item["distance"] = item["distance"]

                                # Check metadata against include list
                                if "metadata" in item and (
                                    include_all or "metadata" in include_set
                                ):
                                    result_item["metadata"] = item["metadata"]

                                query_items.append(result_item)
                            results.append(query_items)
                    else:
                        # It's a flat list (single query results)
                        query_items = []
                        for item in response_json["results"]:
                            result_item = {"id": item["id"]}

                            # Always include distance if present (core part of query results)
                            if "distance" in item:
                                result_item["distance"] = item["distance"]

                            # Check metadata against include list
                            if "metadata" in item and (
                                include_all or "metadata" in include_set
                            ):
                                result_item["metadata"] = item["metadata"]

                            query_items.append(result_item)
                        results = query_items

                return results
            except Exception as e:
                error_msg = f"Unexpected error in query: {str(e)}"
                logger.error(error_msg)
                import traceback

                logger.error(traceback.format_exc())
                raise
        except ApiException as e:
            error_msg = f"Query failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error in query: {str(e)}"
            logger.error(error_msg)
            import traceback

            logger.error(traceback.format_exc())
            raise

    def query_binary(
        self,
        query_vectors: np.ndarray,
        top_k: Optional[int] = None,
        n_probes: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        greedy: Optional[bool] = None,
        rerank_mult: Optional[int] = None,
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

        Returns:
            For single query (1D input): List of result dictionaries.
            For batch query (2D input): List of lists of result dictionaries, one list per query vector.

        Raises:
            ValueError: If query fails or vectors have wrong shape.
            TypeError: If query_vectors is not a numpy array.
        """
        if not isinstance(query_vectors, np.ndarray):
            raise TypeError("query_vectors must be a numpy array")

        # Handle 1D array (single query vector)
        is_single_query = False
        if query_vectors.ndim == 1:
            is_single_query = True
            query_vectors = query_vectors.reshape(1, -1)
        elif query_vectors.ndim != 2:
            raise ValueError(
                "query_vectors must be a 1D array (single query) or 2D array (batch queries)"
            )

        # Ensure little-endian float32 dtype for cross-platform binary compatibility
        if query_vectors.dtype != np.dtype("<f4"):
            query_vectors = query_vectors.astype("<f4")

        # Encode vectors as base64
        vectors_b64 = base64.b64encode(query_vectors.tobytes()).decode("ascii")

        # Build the request using generated models
        batch = BinaryQueryBatch(
            vectors_b64=vectors_b64,
            dimension=query_vectors.shape[1],
        )

        # Build kwargs to avoid passing None values (which would be serialized)
        request_kwargs = {
            "index_name": self._index_name,
            "index_key": self._key_to_hex(),
            "batch": batch,
        }
        if top_k is not None:
            request_kwargs["top_k"] = top_k
        if n_probes is not None:
            request_kwargs["n_probes"] = n_probes
        if filters is not None:
            request_kwargs["filters"] = filters
        if include is not None:
            request_kwargs["include"] = include
        if greedy is not None:
            request_kwargs["greedy"] = greedy
        if rerank_mult is not None:
            request_kwargs["rerank_mult"] = rerank_mult
        request = BinaryQueryRequest(**request_kwargs)

        try:
            response = self._api.query_vectors_binary_v1_vectors_query_binary_post(
                binary_query_request=request,
                _headers=self._request_headers(),
            )

            # Results is an anyOf wrapper - extract actual_instance
            results = response.results.actual_instance
            # Convert QueryResultItem objects to dicts
            if results and isinstance(results[0], list):
                # Batch results: List[List[QueryResultItem]]
                batch_results = [
                    [item.to_dict() for item in result_list] for result_list in results
                ]
                # If input was 1D, return just the first result list
                if is_single_query:
                    return batch_results[0]
                return batch_results
            else:
                # Single query: List[QueryResultItem]
                return [item.to_dict() for item in results]

        except ApiException as e:
            error_msg = f"Failed to query (binary): {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def query_metadata(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        order_by: Optional[Union[str, Dict[str, int]]] = None,
        ascending: bool = True,
    ) -> List[str]:
        """
        Find items by metadata alone — no query vector, no distances.

        Resolves ``filters`` entirely against the encrypted metadata index and
        returns the matching item IDs. Works on untrained indexes.

        Unlike :meth:`query`, there is no post-filter stage to fall back on, so
        the index's ``metadata_schema`` is enforced rather than advisory:
        ``$regex``/``$contains`` require a ``pattern`` field, and a field
        declared ``filterable=False`` cannot be filtered on at all. Both raise
        ``ValueError``. Use :meth:`query` with a vector for those.

        Args:
            filters: Metadata filters; ``None``/empty matches everything.
            top_k: Cap on IDs returned, applied AFTER ``order_by``. ``None``
                returns every match.
            order_by: Field to sort matches by, either a name or a MongoDB-style
                single-field dict (``{"views": -1}``, which also sets the
                direction). Unordered when omitted.
            ascending: Sort direction, when ``order_by`` is a plain field name.

        Returns:
            Matching item IDs — ordered when ``order_by`` was given.

        Raises:
            ValueError: If the filter cannot be resolved from the metadata
                index, or ``order_by`` is malformed.
        """
        # Accept core's {field: 1|-1} form and normalize; the service takes a
        # field name plus a direction flag.
        if isinstance(order_by, dict):
            if len(order_by) != 1:
                raise ValueError(
                    f"order_by dict must specify exactly one field, got {len(order_by)}"
                )
            ((order_by, direction),) = order_by.items()
            ascending = int(direction) >= 0

        try:
            request = QueryMetadataRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                filters=filters or {},
                top_k=top_k,
                order_by=order_by,
                ascending=ascending,
            )
            response = self._api.query_metadata_v1_vectors_query_metadata_post(
                query_metadata_request=request
            )
            return response.ids
        except ApiException as e:
            error_msg = f"Failed to query metadata: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def list_ids(self) -> List[str]:
        """
        List all document IDs in the index.

        Returns:
            List of document IDs.
        """
        try:
            list_ids_request = ListIDsRequest(
                index_key=self._key_to_hex(), index_name=self._index_name
            )
            response = self._api.list_ids_v1_vectors_list_ids_post(
                list_ids_request=list_ids_request
            )

            return response.ids
        except ApiException as e:
            error_msg = f"Failed to list document IDs: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def is_training(self) -> bool:
        """
        Get the current training status of the index.

        Returns:
            A dictionary containing training status information.
        """
        try:
            response = self._api.get_training_status_v1_indexes_training_status_get()

            if self._index_name in response.training_indexes:
                return True

            return False

        except ApiException as e:
            error_msg = f"Failed to get index training status: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # ------------------------------------------------------------------
    # RBAC — user management (root API key required)
    #
    # A user is scoped to this one index with a permission set drawn from
    # {"read", "write"}, enforced cryptographically by the service: the
    # wrapped data-encryption keys that exist for a user *are* their
    # permission set, so there is no policy blob to keep in sync and
    # revoking a user erases their keys. These routes are only accepted
    # when the service runs with CYBORGDB_SERVICE_ROOT_KEY set and this client
    # was constructed with that root key.
    # ------------------------------------------------------------------

    def create_user(self, permissions: List[str]) -> Dict[str, str]:
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
            ``Client`` and need no index key of their own.

        Raises:
            ValueError: If the user could not be created (e.g. the client
                is not using the root key, or ``permissions`` is invalid).
        """
        # SDK-supplied-KEK indexes: the service needs the index key to
        # unwrap the root DEK and re-wrap it under the new user's key.
        # KMS-backed indexes resolve it server-side, so index_key is None.
        request = CreateUserRequest(
            permissions=permissions, index_key=self._index_key_hex
        )
        try:
            response = self._api.create_user_v1_indexes_index_name_users_post(
                index_name=self._index_name, create_user_request=request
            )
            return {"user_id": response.user_id, "api_key": response.api_key}
        except ApiException as e:
            error_msg = f"Failed to create user: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def list_users(self) -> List[Dict[str, Any]]:
        """List the users provisioned for this index.

        Returns:
            A list of ``{"user_id": "<hex>", "permissions": [...]}`` dicts.
            Permissions are derived from which wrapped keys exist for each
            user (the cryptographic source of truth), not a stored field.

        Raises:
            ValueError: If the users could not be listed (e.g. the client
                is not using the root key).
        """
        try:
            response = self._api.list_users_v1_indexes_index_name_users_get(
                index_name=self._index_name, x_index_key=self._index_key_hex
            )
            return [
                {"user_id": u.user_id, "permissions": u.permissions}
                for u in response.users
            ]
        except ApiException as e:
            error_msg = f"Failed to list users: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def delete_user(self, user_id: str) -> None:
        """Revoke a user, erasing their wrapped keys for this index.

        After this returns, the user's API key is rejected on the next
        request — the service can no longer unwrap any key for them.

        Args:
            user_id: The hex ``user_id`` returned by ``create_user`` (also
                surfaced by ``list_users``).

        Raises:
            ValueError: If the user could not be deleted.
        """
        try:
            self._api.delete_user_v1_indexes_index_name_users_user_id_delete(
                index_name=self._index_name,
                user_id=user_id,
                x_index_key=self._index_key_hex,
            )
        except ApiException as e:
            error_msg = f"Failed to delete user: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _key_to_hex(self) -> Optional[str]:
        """Hex-encoded key for API calls, or ``None`` for KMS-backed indexes.
        Computed once in ``__init__`` since the key never changes."""
        return self._index_key_hex

    def _ior(self) -> IndexOperationRequest:
        """Build the name+key request used by describe/delete-style endpoints."""
        return IndexOperationRequest(
            index_key=self._key_to_hex(), index_name=self._index_name
        )
