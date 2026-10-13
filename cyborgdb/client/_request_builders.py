"""Transport-free request assembly and response parsing shared by the sync
and async clients.

Each builder takes a ``models`` namespace (``cyborgdb.openapi_client.models``
or ``cyborgdb.openapi_client_async.models``) and instantiates the request
classes from it, so the sync client never receives async models and vice
versa. Response parsers duck-type the deserialized response objects, which
expose the same attributes in both generated packages.

Nothing in this module performs I/O.
"""

import base64
import datetime as _dt
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cyborgdb.exceptions import ValidationError, _ArgumentTypeError

_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
_MS = _dt.timedelta(milliseconds=1)


def coerce_datetimes(value):
    """Recursively replace datetime/date objects with integer epoch milliseconds.

    No single stdlib function covers every case here: ``datetime.timestamp()``
    interprets naive datetimes using the local system timezone, but the engine
    contract requires naive → UTC. We also need recursive traversal so nested
    filter dicts (``$and``/``$or``/``$in``) are coerced transparently.

    Naive datetimes are treated as UTC. Sub-millisecond precision is truncated.
    Strings, numbers, and other types pass through unchanged.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return (value - _EPOCH) // _MS
    if isinstance(value, _dt.date):
        midnight = _dt.datetime.combine(value, _dt.time(), tzinfo=_dt.timezone.utc)
        return (midnight - _EPOCH) // _MS
    if isinstance(value, dict):
        return {k: coerce_datetimes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [coerce_datetimes(item) for item in value]
    return value


def validate_index_key(index_key: bytes) -> None:
    """Raise ValidationError unless ``index_key`` is a 32-byte ``bytes`` object."""
    if not isinstance(index_key, bytes) or len(index_key) != 32:
        raise ValidationError("index_key must be a 32-byte bytes object")


def apply_full_text_implication(
    metadata_schema: Optional[Dict[str, Dict[str, bool]]],
) -> Optional[Dict[str, Dict[str, bool]]]:
    """Make ``full_text=True`` imply ``filterable=False`` unless set explicitly.

    The generated ``MetadataFieldPolicy`` fills ``filterable=True`` by default,
    which the service reads as an explicit conflict with ``full_text``.
    Stopgap until the service schema drops that default (cyborgdb-core#2393).
    """
    if not metadata_schema:
        return metadata_schema
    return {
        field: {"filterable": False, **policy}
        if isinstance(policy, dict) and policy.get("full_text")
        else policy
        for field, policy in metadata_schema.items()
    }


def request_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Headers for data-path calls. ``X-API-Key`` is only included when one is
    configured: with auth disabled the SDK may be constructed without a key and
    must not send an empty header."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _float32_list(vector):
    """Normalize a vector to a float32 list so JSON serialization matches the
    binary path. Non-array inputs pass through unchanged."""
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32).tolist()
    if isinstance(vector, list):
        return np.array(vector, dtype=np.float32).tolist()
    return vector


def _vectors_to_b64(vectors: np.ndarray) -> str:
    """Little-endian float32 bytes, base64-encoded, for cross-platform transfer."""
    if vectors.dtype != np.dtype("<f4"):
        vectors = vectors.astype("<f4")
    return base64.b64encode(vectors.tobytes()).decode("ascii")


def _encode_contents(value):
    """Base64-encode bytes-like contents for JSON; strings pass through."""
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("utf-8")
    return value


def hybrid_query_kwargs(
    text: Optional[str],
    text_fields: Optional[List[str]],
    text_field_weights: Optional[List[float]],
    require_all_terms: Optional[bool],
    alpha: Optional[float],
    rrf_k: Optional[float],
    window_mult: Optional[int],
) -> Dict[str, Any]:
    """The hybrid text-leg knobs that were actually set. Only non-None values
    are forwarded so an index without full_text fields keeps seeing text-free
    requests."""
    return {
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


def build_upsert_request(
    models,
    index_name: str,
    index_key_hex: Optional[str],
    arg1: Union[List[Dict[str, Any]], List[str]],
    arg2: Optional[Any],
):
    """Assemble the JSON ``UpsertRequest`` for ``upsert``.

    ``arg1`` is a list of item dicts (``arg2`` is None) or a list of ids with
    ``arg2`` a non-numpy sequence of vectors. The numpy ``arg2`` case is routed
    to the binary endpoint by the caller before reaching here.
    """
    items = []

    if arg2 is None:
        if not isinstance(arg1, list) or not all(
            isinstance(item, dict) for item in arg1
        ):
            raise _ArgumentTypeError(
                "When arg2 is None, arg1 must be a list of dictionaries"
            )

        for item_dict in arg1:
            if "id" not in item_dict:
                raise ValidationError("Each item dictionary must contain an 'id' field")

            item = {"id": item_dict["id"]}

            if "vector" in item_dict:
                item["vector"] = _float32_list(item_dict["vector"])

            if "contents" in item_dict:
                item["contents"] = models.Contents(
                    _encode_contents(item_dict["contents"])
                )

            if "metadata" in item_dict:
                item["metadata"] = coerce_datetimes(item_dict["metadata"])

            items.append(item)
    else:
        if not isinstance(arg1, list):
            raise _ArgumentTypeError("arg1 must be a list of IDs")

        if len(arg1) != len(arg2):
            raise ValidationError("Number of IDs must match number of vectors")

        for id_val, vector in zip(arg1, arg2):
            items.append({"id": str(id_val), "vector": _float32_list(vector)})

    return models.UpsertRequest(
        items=items, index_key=index_key_hex, index_name=index_name
    )


def _wrap_binary_contents(
    models,
    contents: Optional[List[Optional[Union[str, bytes, bytearray]]]],
):
    """Wrap per-item contents in the generated anyOf model, base64-encoding
    bytes the same way ``upsert()`` does on the JSON path.

    ``None`` becomes ``""`` rather than staying ``None``: the generated
    ``BinaryVectorBatch.to_dict()`` drops ``None`` entries, which would shift
    every later item's contents onto the wrong id. The service treats empty
    contents as absent, so the stored result is the same.
    """
    if contents is None:
        return None
    return [
        models.BinaryVectorBatchContentsInner(
            "" if value is None else _encode_contents(value)
        )
        for value in contents
    ]


def build_binary_upsert_request(
    models,
    index_name: str,
    index_key_hex: Optional[str],
    ids: List[str],
    vectors: np.ndarray,
    metadata: Optional[List[Optional[Dict[str, Any]]]],
    contents: Optional[List[Optional[Union[str, bytes]]]],
):
    """Validate the inputs and assemble the ``BinaryUpsertRequest``."""
    if not isinstance(vectors, np.ndarray):
        raise _ArgumentTypeError("vectors must be a numpy array")

    if vectors.ndim != 2:
        raise ValidationError(
            "vectors must be a 2D array of shape (n_vectors, dimension)"
        )

    if len(ids) != vectors.shape[0]:
        raise ValidationError(
            f"Number of ids ({len(ids)}) must match number of vectors ({vectors.shape[0]})"
        )
    for name, values in (("metadata", metadata), ("contents", contents)):
        if values is not None and len(values) != len(ids):
            raise ValidationError(
                f"Number of {name} entries ({len(values)}) must match number of ids ({len(ids)})"
            )

    batch = models.BinaryVectorBatch(
        ids=ids,
        vectors_b64=_vectors_to_b64(vectors),
        dimension=vectors.shape[1],
        metadata=(
            [coerce_datetimes(m) for m in metadata] if metadata is not None else None
        ),
        contents=_wrap_binary_contents(models, contents),
    )

    return models.BinaryUpsertRequest(
        index_name=index_name,
        index_key=index_key_hex,
        batch=batch,
    )


def _query_options(
    top_k: Optional[int],
    n_probes: Optional[int],
    filters: Optional[Dict[str, Any]],
    include: Optional[List[str]],
    greedy: Optional[bool],
    rerank_mult: Optional[int],
) -> Dict[str, Any]:
    """The optional query knobs that were set, so None is never serialized."""
    kwargs: Dict[str, Any] = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if n_probes is not None:
        kwargs["n_probes"] = n_probes
    if greedy is not None:
        kwargs["greedy"] = greedy
    if rerank_mult is not None:
        kwargs["rerank_mult"] = rerank_mult
    if filters is not None:
        kwargs["filters"] = coerce_datetimes(filters)
    if include is not None:
        kwargs["include"] = include
    return kwargs


def build_query_request(
    models,
    index_name: str,
    index_key_hex: Optional[str],
    query_vectors: Optional[Union[List[List[float]], List[float]]],
    query_contents: Optional[str],
    top_k: Optional[int],
    n_probes: Optional[int],
    filters: Optional[Dict[str, Any]],
    include: Optional[List[str]],
    greedy: Optional[bool],
    rerank_mult: Optional[int],
    hybrid_kwargs: Dict[str, Any],
):
    """Assemble the JSON ``Request`` for ``query``.

    ``query_vectors`` is a flat list (single query), a list of vectors (batch),
    or None (content-based query). Numpy input is routed to the binary endpoint
    by the caller before reaching here.
    """
    vector_list = None
    is_single_query = False

    if query_vectors is not None:
        if isinstance(query_vectors, list):
            if not query_vectors:
                raise ValidationError("Empty list provided for `query_vectors`.")
            if isinstance(query_vectors[0], (list, np.ndarray)):
                vector_list = [_float32_list(v) for v in query_vectors]
            else:
                is_single_query = True
                vector_list = _float32_list(query_vectors)
        else:
            raise ValidationError("Invalid type for `query_vectors`")

    query_kwargs = {
        "index_key": index_key_hex,
        "index_name": index_name,
        "query_vectors": vector_list,
    }
    if is_single_query or query_contents is not None:
        if query_contents is not None:
            query_kwargs["query_contents"] = query_contents
        query_kwargs.update(
            _query_options(top_k, n_probes, filters, include, greedy, rerank_mult)
        )
        query_kwargs.update(hybrid_kwargs)
        query_request = models.QueryRequest(**query_kwargs)
    else:
        query_kwargs.update(
            _query_options(top_k, n_probes, filters, include, greedy, rerank_mult)
        )
        query_kwargs.update(hybrid_kwargs)
        query_request = models.BatchQueryRequest(**query_kwargs)

    return models.Request(query_request)


def _parse_query_item(item: Dict[str, Any], include_metadata: bool) -> Dict[str, Any]:
    result_item = {"id": item["id"]}
    if item.get("distance") is not None:
        result_item["distance"] = item["distance"]
    # Hybrid (text=...) results carry a fused score instead of a distance.
    if item.get("score") is not None:
        result_item["score"] = item["score"]
    if "metadata" in item and include_metadata:
        result_item["metadata"] = item["metadata"]
    return result_item


def parse_query_response(
    response_json: Dict[str, Any], include: Optional[List[str]]
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Turn the raw JSON body of the query endpoint into plain result dicts.

    ``include=None`` keeps everything the server returned; otherwise metadata
    is only kept when it was asked for.
    """
    include_metadata = include is None or "metadata" in set(include)

    if "results" not in response_json:
        return []
    results = response_json["results"]
    if results and isinstance(results[0], list):
        return [
            [_parse_query_item(item, include_metadata) for item in query_result]
            for query_result in results
        ]
    return [_parse_query_item(item, include_metadata) for item in results]


def build_binary_query_request(
    models,
    index_name: str,
    index_key_hex: Optional[str],
    query_vectors: np.ndarray,
    top_k: Optional[int],
    n_probes: Optional[int],
    filters: Optional[Dict[str, Any]],
    include: Optional[List[str]],
    greedy: Optional[bool],
    rerank_mult: Optional[int],
    hybrid_kwargs: Dict[str, Any],
) -> Tuple[Any, bool]:
    """Validate ``query_vectors`` and assemble the ``BinaryQueryRequest``.

    Returns the request and whether the input was a single (1D) query, which
    decides the result shape.
    """
    if not isinstance(query_vectors, np.ndarray):
        raise _ArgumentTypeError("query_vectors must be a numpy array")

    is_single_query = False
    if query_vectors.ndim == 1:
        is_single_query = True
        query_vectors = query_vectors.reshape(1, -1)
    elif query_vectors.ndim != 2:
        raise ValidationError(
            "query_vectors must be a 1D array (single query) or 2D array (batch queries)"
        )

    batch = models.BinaryQueryBatch(
        vectors_b64=_vectors_to_b64(query_vectors),
        dimension=query_vectors.shape[1],
    )

    request_kwargs = {
        "index_name": index_name,
        "index_key": index_key_hex,
        "batch": batch,
    }
    request_kwargs.update(
        _query_options(top_k, n_probes, filters, include, greedy, rerank_mult)
    )
    request_kwargs.update(hybrid_kwargs)
    return models.BinaryQueryRequest(**request_kwargs), is_single_query


def parse_binary_query_response(
    response, is_single_query: bool
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Unwrap the binary query response into plain result dicts.

    A 1D input always yields a flat list, even though the service answers a
    batch of one.
    """
    results = response.results.actual_instance
    if results and isinstance(results[0], list):
        batch_results = [
            [item.to_dict() for item in result_list] for result_list in results
        ]
        if is_single_query:
            return batch_results[0]
        return batch_results
    return [item.to_dict() for item in results]


def build_query_metadata_request(
    models,
    index_name: str,
    index_key_hex: Optional[str],
    filters: Optional[Dict[str, Any]],
    top_k: Optional[int],
    order_by: Optional[Union[str, Dict[str, int]]],
    ascending: bool,
    text: Optional[str],
    text_fields: Optional[List[str]],
    text_field_weights: Optional[List[float]],
    require_all_terms: Optional[bool],
):
    """Normalize ``order_by`` and assemble the ``QueryMetadataRequest``."""
    # Accept core's {field: 1|-1} form; the service takes a field name plus a
    # direction flag.
    if isinstance(order_by, dict):
        if len(order_by) != 1:
            raise ValidationError(
                f"order_by dict must specify exactly one field, got {len(order_by)}"
            )
        ((order_by, direction),) = order_by.items()
        ascending = int(direction) >= 0

    request_kwargs = {
        "index_key": index_key_hex,
        "index_name": index_name,
        "filters": coerce_datetimes(filters or {}),
        "top_k": top_k,
        # order_by is an anyOf(str, {field: 1|-1}); after the normalization
        # above it is always a field name, so wrap the string.
        "order_by": models.OrderBy(order_by) if order_by is not None else None,
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

    return models.QueryMetadataRequest(**request_kwargs)


def parse_query_metadata_response(response, text: Optional[str]) -> List[Dict]:
    """Rows as ``{"id"}``, plus ``score`` on the text path, matching core's
    ``list[MetadataResult]``. A filter-only query has nothing to score, so the
    key is absent rather than None."""
    rows = response.results or []
    if text:
        return [{"id": item.id, "score": item.score} for item in rows]
    return [{"id": item.id} for item in rows]


def parse_get_response(response, include: List[str]) -> List[Dict[str, Any]]:
    """Convert the get-vectors response into plain item dicts, keeping only the
    requested fields. Metadata that arrives as a JSON string is decoded."""
    items = []
    if hasattr(response, "results"):
        for item in response.results:
            item_dict = {"id": item.id}

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
