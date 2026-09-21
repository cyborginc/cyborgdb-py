"""Shared transport-free request-building helpers.

Both the sync EncryptedIndex and the async AsyncEncryptedIndex call these
functions to build OpenAPI model instances.  I/O is the only step that
differs between the two paths.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def build_upsert_items_list(arg1, arg2) -> list:
    """Build the items list for a JSON upsert request.

    Mirrors the item-construction logic in EncryptedIndex.upsert (the
    non-binary path).  Raises TypeError / ValueError for invalid inputs,
    same as the original method.
    """
    from cyborgdb.openapi_client.models import Contents

    items: list = []

    if arg2 is None:
        if not isinstance(arg1, list) or not all(
            isinstance(item, dict) for item in arg1
        ):
            raise TypeError("When arg2 is None, arg1 must be a list of dictionaries")

        for item_dict in arg1:
            if "id" not in item_dict:
                raise ValueError("Each item dictionary must contain an 'id' field")

            item: Dict[str, Any] = {"id": item_dict["id"]}

            if "vector" in item_dict:
                vec = item_dict["vector"]
                if isinstance(vec, np.ndarray):
                    item["vector"] = vec.astype(np.float32).tolist()
                elif isinstance(vec, list):
                    item["vector"] = np.array(vec, dtype=np.float32).tolist()
                else:
                    item["vector"] = vec

            if "contents" in item_dict:
                contents_value = item_dict["contents"]
                if isinstance(contents_value, bytes):
                    contents_value = base64.b64encode(contents_value).decode("utf-8")
                elif isinstance(contents_value, bytearray):
                    contents_value = base64.b64encode(bytes(contents_value)).decode(
                        "utf-8"
                    )
                item["contents"] = Contents(contents_value)

            if "metadata" in item_dict:
                item["metadata"] = item_dict["metadata"]

            items.append(item)

    else:
        # arg1 is list of IDs, arg2 is list of vectors (non-numpy)
        if not isinstance(arg1, list):
            raise TypeError("arg1 must be a list of IDs")

        vectors = arg2
        if len(arg1) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")

        for id_val, vector in zip(arg1, vectors):
            if isinstance(vector, np.ndarray):
                vector = vector.astype(np.float32).tolist()
            elif isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32).tolist()
            items.append({"id": str(id_val), "vector": vector})

    return items


def build_upsert_binary_request(
    index_name: str,
    key_hex: Optional[str],
    ids: List[str],
    vectors: np.ndarray,
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    contents: Optional[List[Optional[Union[str, bytes]]]] = None,
):
    """Validate and build a BinaryUpsertRequest.

    Returns a BinaryUpsertRequest ready for the API call.
    Raises TypeError / ValueError for invalid inputs.
    """
    from cyborgdb.openapi_client.models.binary_upsert_request import BinaryUpsertRequest
    from cyborgdb.openapi_client.models.binary_vector_batch import BinaryVectorBatch

    if not isinstance(vectors, np.ndarray):
        raise TypeError("vectors must be a numpy array")
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array of shape (n_vectors, dimension)")
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"Number of ids ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
        )

    if vectors.dtype != np.dtype("<f4"):
        vectors = vectors.astype("<f4")

    vectors_b64 = base64.b64encode(vectors.tobytes()).decode("ascii")

    batch = BinaryVectorBatch(
        ids=ids,
        vectors_b64=vectors_b64,
        dimension=vectors.shape[1],
        metadata=metadata,
        contents=contents,
    )
    return BinaryUpsertRequest(
        index_name=index_name,
        index_key=key_hex,
        batch=batch,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def prepare_query_request(
    index_name: str,
    key_hex: Optional[str],
    query_vectors,
    query_contents: Optional[str],
    top_k: Optional[int],
    n_probes: Optional[int],
    filters: Optional[Dict[str, Any]],
    include: Optional[List[str]],
    greedy: Optional[bool],
    rerank_mult: Optional[int],
    hybrid_kwargs: Dict[str, Any],
):
    """Prepare a JSON query request (non-binary path).

    Returns ``(dispatch_to_binary, request_obj, is_single_query)``:

    - ``dispatch_to_binary=True``: caller should dispatch to query_binary
      with the original ``query_vectors`` (which is a numpy array).
    - Otherwise: ``request_obj`` is a ``Request`` ready for the API, and
      ``is_single_query`` indicates whether the result should be a flat list.
    """
    from cyborgdb.openapi_client.models.batch_query_request import BatchQueryRequest
    from cyborgdb.openapi_client.models.query_request import QueryRequest
    from cyborgdb.openapi_client.models.request import Request

    if query_vectors is not None and isinstance(query_vectors, np.ndarray):
        return True, None, None

    vector_list = None
    is_single_query = False

    if query_vectors is not None:
        if isinstance(query_vectors, list):
            if not query_vectors:
                raise ValueError("Empty list provided for `query_vectors`.")
            if isinstance(query_vectors[0], (list, np.ndarray)):
                vector_list = [
                    np.array(v, dtype=np.float32).tolist() for v in query_vectors
                ]
            else:
                is_single_query = True
                vector_list = np.array(query_vectors, dtype=np.float32).tolist()
        else:
            raise ValueError("Invalid type for `query_vectors`")

    if is_single_query or query_contents is not None:
        query_kwargs: Dict[str, Any] = {
            "index_key": key_hex,
            "index_name": index_name,
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
        query_kwargs.update(hybrid_kwargs)
        inner_request = QueryRequest(**query_kwargs)
    else:
        query_kwargs = {
            "index_key": key_hex,
            "index_name": index_name,
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
        query_kwargs.update(hybrid_kwargs)
        inner_request = BatchQueryRequest(**query_kwargs)

    return False, Request(inner_request), is_single_query


def build_query_binary_request(
    index_name: str,
    key_hex: Optional[str],
    query_vectors: np.ndarray,
    top_k: Optional[int],
    n_probes: Optional[int],
    filters: Optional[Dict[str, Any]],
    include: Optional[List[str]],
    greedy: Optional[bool],
    rerank_mult: Optional[int],
    hybrid_kwargs: Dict[str, Any],
):
    """Validate and build a BinaryQueryRequest.

    Returns ``(request, is_single_query)``.  Raises TypeError / ValueError
    for invalid inputs.
    """
    from cyborgdb.openapi_client.models.binary_query_batch import BinaryQueryBatch
    from cyborgdb.openapi_client.models.binary_query_request import BinaryQueryRequest

    if not isinstance(query_vectors, np.ndarray):
        raise TypeError("query_vectors must be a numpy array")

    is_single_query = False
    if query_vectors.ndim == 1:
        is_single_query = True
        query_vectors = query_vectors.reshape(1, -1)
    elif query_vectors.ndim != 2:
        raise ValueError(
            "query_vectors must be a 1D array (single query) or 2D array (batch queries)"
        )

    if query_vectors.dtype != np.dtype("<f4"):
        query_vectors = query_vectors.astype("<f4")

    vectors_b64 = base64.b64encode(query_vectors.tobytes()).decode("ascii")

    batch = BinaryQueryBatch(
        vectors_b64=vectors_b64,
        dimension=query_vectors.shape[1],
    )

    request_kwargs: Dict[str, Any] = {
        "index_name": index_name,
        "index_key": key_hex,
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
    request_kwargs.update({k: v for k, v in hybrid_kwargs.items() if v is not None})

    return BinaryQueryRequest(**request_kwargs), is_single_query


# ---------------------------------------------------------------------------
# Response-parsing helpers (no I/O)
# ---------------------------------------------------------------------------


def _extract_result_item(
    item: Dict[str, Any],
    include_all: bool,
    include_set: set,
) -> Dict[str, Any]:
    result_item: Dict[str, Any] = {"id": item["id"]}
    if item.get("distance") is not None:
        result_item["distance"] = item["distance"]
    if item.get("score") is not None:
        result_item["score"] = item["score"]
    if "metadata" in item and (include_all or "metadata" in include_set):
        result_item["metadata"] = item["metadata"]
    return result_item


def parse_query_json_response(
    response_json: Dict[str, Any],
    include: Optional[List[str]],
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Parse the raw JSON body returned by the query endpoint."""
    include_all = include is None
    include_set = set(include) if include else set()

    results: list = []
    if "results" in response_json:
        if response_json["results"] and isinstance(response_json["results"][0], list):
            for query_result in response_json["results"]:
                query_items = [
                    _extract_result_item(item, include_all, include_set)
                    for item in query_result
                ]
                results.append(query_items)
        else:
            for item in response_json["results"]:
                results.append(_extract_result_item(item, include_all, include_set))
    return results


def parse_query_binary_response(response, is_single_query: bool):
    """Parse the deserialized QueryResponse from the binary query endpoint."""
    results_anyof = response.results
    results = results_anyof.actual_instance
    if results and isinstance(results[0], list):
        batch_results = [
            [item.to_dict() for item in result_list] for result_list in results
        ]
        if is_single_query:
            return batch_results[0]
        return batch_results
    return [item.to_dict() for item in results]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def parse_raw_query_response_bytes(
    raw_data: bytes,
    include: Optional[List[str]],
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Decode and parse a raw bytes query response body."""
    response_text = raw_data.decode("utf-8")
    response_json = json.loads(response_text)
    return parse_query_json_response(response_json, include)
