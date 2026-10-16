"""@file adapter.py
@brief cyborgdb-tests adapter for the cyborgdb Python SDK.

Reads JSON-RPC requests on fd 3 and writes responses on fd 4 (see cyborgdb-tests
ADAPTERS.md and ADAPTER-PY-SDK.md). Maps each op to one call on the installed
cyborgdb package. No defaults are filled in and no results are reshaped beyond
the transformations ADAPTERS.md allows.

CYBORGDB_TESTS_VARIANT picks the SDK path: json (default), binary or async.

Run under the driver, from a cyborgdb-tests checkout:
    CYBORG_TEST_SDK_BASE_URL=http://localhost:8000 bin/cyborgdb-tests run \
        --adapter "python ../cyborgdb-py/tests/conformance/adapter.py" \
        --env sdk --variant json
"""

import asyncio
import base64
import datetime as dt
import inspect
import json
import math
import os
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version

import numpy as np

import cyborgdb
from cyborgdb.exceptions import TransportError, ValidationError

PROTOCOL = 1
IMPL = "cyborgdb-py"
CODE_OP_FAILED = -32000
CODE_UNSUPPORTED = -32601
CODE_INVALID_PARAMS = -32602
CODE_ADAPTER = -32603
CODE_PARSE = -32700

VARIANTS = ("json", "binary", "async")
VARIANT = os.environ.get("CYBORGDB_TESTS_VARIANT") or "json"

TYPED_TAGS = {
    "$f32",
    "$bytes",
    "$i64",
    "$datetime",
    "$float",
    "$tensor",
    "$range",
    "$literal",
}

TENSOR_DTYPES = {
    "f32": np.float32,
    "f16": np.float16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
}


class Unsupported(Exception):
    """@brief The SDK can't express the request; reported as -32601."""


class BadParams(Exception):
    """@brief The request itself is malformed; reported as -32602."""


class SdkError(Exception):
    """@brief An exception raised by a cyborgdb call.

    The only failure that gets an error kind.
    """

    def __init__(self, native_error):
        """@brief Wraps the SDK's exception.

        @param native_error The exception the SDK raised.
        """
        super().__init__(str(native_error))
        self.native_error = native_error


_event_loop = None


def sdk(sdk_function, *args, **kwargs):
    """@brief Calls into cyborgdb, awaiting the result on the async client.

    Anything raised here is the product's answer; everything else is the
    adapter's.

    @param sdk_function The SDK method or constructor to call.
    @param args Positional arguments forwarded unchanged.
    @param kwargs Keyword arguments forwarded unchanged.
    @return The SDK's result, awaited if it is awaitable.
    @throws SdkError Whatever the SDK raised, wrapped.
    """
    global _event_loop
    try:
        result = sdk_function(*args, **kwargs)
        if inspect.isawaitable(result):
            if _event_loop is None:
                # One loop for the whole run: the async client's HTTP session
                # is bound to the loop it was first used on.
                _event_loop = asyncio.new_event_loop()
            result = _event_loop.run_until_complete(result)
        return result
    except Exception as native_error:  # noqa: BLE001
        raise SdkError(native_error) from native_error


# ---------- typed values ----------


def decode(wire_value):
    """@brief Converts a wire value to native Python.

    Only a single-key object whose key is a known tag is typed; any other `$`
    key is data.

    @param wire_value A JSON-decoded param value.
    @return The native value.
    """
    if isinstance(wire_value, list):
        return [decode(element) for element in wire_value]
    if isinstance(wire_value, dict):
        if len(wire_value) == 1:
            ((tag, payload),) = wire_value.items()
            if tag in TYPED_TAGS:
                return decode_typed(tag, payload)
        return {key: decode(value) for key, value in wire_value.items()}
    return wire_value


def decode_typed(tag, payload):
    """@brief Decodes one typed value.

    @param tag The type tag, e.g. `$f32`.
    @param payload The value under the tag.
    @return The native value.
    @throws BadParams The tag is unknown.
    """
    if tag == "$literal":
        return payload
    if tag == "$bytes":
        return base64.b64decode(payload, validate=True)
    if tag == "$f32":
        # Plain lists, not numpy: the SDK sends a numpy array down the binary
        # path, so the ops pick the array type per variant.
        if len(payload) > 0 and isinstance(payload[0], list):
            return [[decode_number(number) for number in row] for row in payload]
        return [decode_number(number) for number in payload]
    if tag == "$float":
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[payload]
    if tag == "$i64":
        return int(payload)
    if tag == "$datetime":
        return dt.datetime.fromisoformat(payload.replace("Z", "+00:00"))
    if tag == "$range":
        start, stop = int(payload[0]), int(payload[1])
        return [str(i) for i in range(start, stop)]
    if tag == "$tensor":
        shape = tuple(int(size) for size in payload["shape"])
        dtype = TENSOR_DTYPES[payload["dtype"]]
        if 0 in shape:  # mmap can't map zero bytes
            return np.zeros(shape, dtype=dtype)
        return np.memmap(
            payload["path"],
            dtype=dtype,
            mode="r",
            offset=int(payload["offset"]),
            shape=shape,
        )
    raise BadParams(f"unknown typed value {tag}")


def decode_number(number):
    """@brief Decodes one vector element, which may be a `$float` tag.

    @param number A JSON number or a `{"$float": ...}` object.
    @return The number.
    """
    if isinstance(number, dict) and "$float" in number:
        return decode_typed("$float", number["$float"])
    return number


def encode(native_value):
    """@brief Converts a native result to a wire value.

    @param native_value A value returned by the SDK.
    @return The JSON-encodable wire value.
    """
    if isinstance(native_value, (bytes, bytearray, memoryview)):
        return {"$bytes": base64.b64encode(bytes(native_value)).decode()}
    if isinstance(native_value, np.ndarray):
        if native_value.ndim == 1:
            return encode_f32(native_value)
        return {"$f32": [[encode_float(float(f)) for f in row] for row in native_value]}
    if isinstance(native_value, dict):
        return {key: encode(value) for key, value in native_value.items()}
    if isinstance(native_value, (list, tuple)):
        return [encode(element) for element in native_value]
    if native_value is None or isinstance(native_value, (bool, str)):
        return native_value
    if isinstance(native_value, (int, np.integer)):
        integer = int(native_value)
        return integer if abs(integer) <= 2**53 else {"$i64": str(integer)}
    if isinstance(native_value, (float, np.floating)):
        return encode_float(float(native_value))
    return native_value


def encode_float(number):
    """@brief Encodes a float, tagging the values JSON can't carry.

    @param number A float.
    @return The float, or a `$float` object for nan and infinities.
    """
    if math.isnan(number):
        return {"$float": "nan"}
    if math.isinf(number):
        return {"$float": "inf" if number > 0 else "-inf"}
    return number


def encode_f32(vector):
    """@brief Encodes a vector as `$f32`.

    @param vector A sequence of floats.
    @return A `{"$f32": [...]}` object.
    """
    return {"$f32": [encode_float(float(element)) for element in vector]}


# ---------- handles ----------

client_handles = {}  # handle name -> Client or AsyncClient
index_handles = {}  # handle name -> EncryptedIndex or AsyncEncryptedIndex


def require_param(params, name):
    """@brief Returns a param the protocol requires.

    @param params The decoded request params.
    @param name The param name.
    @return The param's value.
    @throws BadParams The param is missing.
    """
    if name not in params:
        raise BadParams(f"missing {name}")
    return params[name]


def lookup_client(params):
    """@brief Returns the client named by the `client` param.

    @param params The decoded request params.
    @return The bound client.
    @throws BadParams No client is bound to that name.
    """
    handle_name = require_param(params, "client")
    if handle_name not in client_handles:
        raise BadParams(f"unknown client handle {handle_name!r}")
    return client_handles[handle_name]


def lookup_index(params):
    """@brief Returns the index named by the `index` param.

    @param params The decoded request params.
    @return The bound index.
    @throws BadParams No index is bound to that name.
    """
    handle_name = require_param(params, "index")
    if handle_name not in index_handles:
        raise BadParams(f"unknown index handle {handle_name!r}")
    return index_handles[handle_name]


def pick_params(params, *names, rename=None):
    """@brief Selects the params a spec set, leaving omitted ones out.

    @param params The decoded request params.
    @param names The param names to forward, if present.
    @param rename Optional map from spec name to SDK keyword name.
    @return Keyword arguments for the SDK call.
    """
    rename = rename or {}
    return {rename.get(name, name): params[name] for name in names if name in params}


# ---------- op helpers ----------


def reject_params(params, *names):
    """@brief Rejects params the SDK has no way to express.

    @param params The decoded request params.
    @param names Param names the SDK can't take for this op.
    @throws Unsupported One of the params is present.
    """
    present = [name for name in names if name in params]
    if present:
        raise Unsupported(f"the SDK can't express {', '.join(present)}")


def index_for_op(params, *overrides):
    """@brief Returns the op's index, rejecting per-call credential overrides.

    The SDK binds the index key at create_index / load_index, so a per-call
    `key` or `user_id` can't be passed.

    @param params The decoded request params.
    @param overrides The override params this op defines in ops.yaml.
    @return The bound index.
    @throws Unsupported An override is present.
    """
    reject_params(params, *overrides)
    return lookup_index(params)


# ---------- vectors per variant ----------


def json_vectors(vectors):
    """@brief Prepares vectors for the JSON path: plain lists, never numpy.

    A numpy array would send the SDK down the binary path.

    @param vectors A decoded `$f32` list or a `$tensor` array.
    @return The vectors as nested Python lists.
    """
    if isinstance(vectors, np.ndarray):
        return vectors.tolist()
    return vectors


def binary_vectors(vectors):
    """@brief Prepares vectors for the binary path: a float32 numpy array.

    @param vectors A decoded `$f32` list or a `$tensor` array.
    @return A float32 numpy array.
    @throws Unsupported The vectors aren't rectangular or numeric, which a
        numpy array can't carry.
    """
    if isinstance(vectors, np.ndarray):
        return np.asarray(vectors, dtype=np.float32)
    rows = [row for row in vectors if isinstance(row, list)]
    if rows and len({len(row) for row in rows}) > 1:
        raise Unsupported("the binary path can't carry vectors of different lengths")
    try:
        return np.array(vectors, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise Unsupported(
            f"the binary path can't carry these vectors: {error}"
        ) from error


def query_row_count(vectors, contents):
    """@brief Counts the query rows a request asked for.

    @param vectors The decoded `vectors` param, or None.
    @param contents The decoded `contents` param, or None.
    @return The number of result lists the contract expects.
    """
    if vectors is not None:
        if isinstance(vectors, np.ndarray):
            return vectors.shape[0] if vectors.ndim == 2 else 1
        if vectors and isinstance(vectors[0], list):
            return len(vectors)
        return 1
    if isinstance(contents, list):
        return len(contents)
    return 1


def one_list_per_row(results, row_count):
    """@brief Restores the per-row nesting the SDK drops for a single query.

    @param results The SDK's query results.
    @param row_count The number of query rows in the request.
    @return One list of results per query row.
    """
    if row_count == 1 and (not results or isinstance(results[0], dict)):
        return [results]
    return results


# ---------- result encoding ----------


def contents_bytes(contents):
    """@brief Converts returned contents to bytes, whatever the SDK's type.

    @param contents Contents as the SDK returned them.
    @return The contents as bytes; None stays None.
    """
    if isinstance(contents, str):
        return contents.encode("utf-8")
    return contents


def encode_item(item):
    """@brief Encodes one `get` row: vector as `$f32`, contents as `$bytes`.

    @param item A row returned by `get`.
    @return The wire row.
    """
    row = {}
    for field, value in item.items():
        if field == "vector" and value is not None:
            row[field] = encode_f32(value)
        elif field == "contents":
            row[field] = encode(contents_bytes(value))
        elif field == "metadata" and isinstance(value, str):
            row[field] = encode(json.loads(value))
        else:
            row[field] = encode(value)
    return row


# ---------- ops: client ----------


def op_client_new(params):
    """@brief client.new: opens a Client, or an AsyncClient in the async variant.

    @param params `as`, `base_url`, optional `api_key`.
    @return None.
    """
    handle_name = require_param(params, "as")
    reject_params(params, "storage", "cpu_threads", "gpu")
    client_class = cyborgdb.AsyncClient if VARIANT == "async" else cyborgdb.Client
    client_handles[handle_name] = sdk(
        client_class,
        require_param(params, "base_url"),
        **pick_params(params, "api_key"),
    )
    return None


def op_list_indexes(params):
    """@brief client.list_indexes.

    @param params `client`.
    @return The index names.
    """
    return encode(sdk(lookup_client(params).list_indexes))


def op_health(params):
    """@brief client.health.

    @param params `client`.
    @return The service's health response.
    """
    return encode(sdk(lookup_client(params).get_health))


def op_create_index(params):
    """@brief client.create_index.

    @param params `client`, `as`, `name` and the optional index settings.
    @return None.
    """
    client = lookup_client(params)
    handle_name = require_param(params, "as")
    settings = pick_params(
        params,
        "key",
        "kms_name",
        "dimension",
        "metric",
        "storage_precision",
        "metadata_schema",
        "text_fields",
        "bm25_k1",
        "bm25_b",
        "embedding_model",
        rename={"key": "index_key"},
    )
    index_handles[handle_name] = sdk(
        client.create_index, require_param(params, "name"), **settings
    )
    return None


def op_load_index(params):
    """@brief client.load_index.

    @param params `client`, `as`, `name`, optional `key`.
    @return None.
    """
    client = lookup_client(params)
    handle_name = require_param(params, "as")
    reject_params(params, "user_id")
    index_handles[handle_name] = sdk(
        client.load_index,
        require_param(params, "name"),
        **pick_params(params, "key", rename={"key": "index_key"}),
    )
    return None


# ---------- ops: index ----------


def op_delete_index(params):
    """@brief index.delete_index.

    @param params `index`.
    @return None.
    """
    index = index_for_op(params, "key")
    sdk(index.delete_index)
    return None


def op_upsert(params):
    """@brief index.upsert, through the variant's upsert method.

    @param params `index`, and either `items` or `ids` + `vectors` (+ `metadata`).
    @return None.
    @throws BadParams Neither form is present.
    """
    index = index_for_op(params, "key", "user_id")
    if "items" not in params and "ids" not in params:
        raise BadParams("upsert needs items or ids")
    if VARIANT == "binary":
        upsert_binary(index, params)
    else:
        upsert_json(index, params)
    return None


def upsert_json(index, params):
    """@brief Upserts through `upsert`, which sends JSON.

    @param index The bound index.
    @param params The decoded request params.
    @throws Unsupported Parallel metadata of a different length, which `upsert`
        has no form for.
    """
    if "items" in params:
        sdk(index.upsert, params["items"])
        return
    ids = params["ids"]
    if "metadata" not in params:
        if "vectors" in params:
            sdk(index.upsert, ids, json_vectors(params["vectors"]))
        else:
            sdk(index.upsert, ids)
        return
    # upsert has no parallel-metadata form; the items form carries the same data.
    vectors, metadata = json_vectors(params.get("vectors")), params["metadata"]
    if not isinstance(metadata, list):
        raise BadParams("metadata must be a list")
    if vectors is None or not (len(ids) == len(vectors) == len(metadata)):
        raise Unsupported("upsert can't express parallel metadata of another length")
    items = []
    for item_id, vector, item_metadata in zip(ids, vectors, metadata):
        item = {"id": item_id, "vector": vector}
        if item_metadata is not None:
            item["metadata"] = item_metadata
        items.append(item)
    sdk(index.upsert, items)


BINARY_ITEM_FIELDS = {"id", "vector", "contents", "metadata"}


def upsert_binary(index, params):
    """@brief Upserts through `upsert_binary`.

    @param index The bound index.
    @param params The decoded request params.
    @throws Unsupported An item the binary form can't carry: no id, no vector,
        or a field outside id/vector/contents/metadata.
    """
    if "ids" in params:
        if "vectors" not in params:
            raise Unsupported("upsert_binary needs vectors")
        sdk(
            index.upsert_binary,
            params["ids"],
            binary_vectors(params["vectors"]),
            **pick_params(params, "metadata"),
        )
        return
    items = params["items"]
    for item in items:
        if not isinstance(item, dict) or not {"id", "vector"} <= item.keys():
            raise Unsupported("upsert_binary needs an id and a vector for every item")
        if item.keys() - BINARY_ITEM_FIELDS:
            raise Unsupported("upsert_binary can't carry item fields beyond id/vector")
    columns = {}
    for field in ("metadata", "contents"):
        if any(field in item for item in items):
            columns[field] = [item.get(field) for item in items]
    sdk(
        index.upsert_binary,
        [item["id"] for item in items],
        binary_vectors([item["vector"] for item in items]),
        **columns,
    )


def op_get(params):
    """@brief index.get.

    @param params `index`, `ids`, optional `include`.
    @return The rows, with vectors as `$f32` and contents as `$bytes`.
    """
    index = index_for_op(params, "key", "user_id")
    rows = sdk(
        index.get, require_param(params, "ids"), **pick_params(params, "include")
    )
    return [encode_item(row) for row in rows]


def op_delete(params):
    """@brief index.delete.

    @param params `index`, `ids`.
    @return None.
    """
    index = index_for_op(params, "key", "user_id")
    sdk(index.delete, require_param(params, "ids"))
    return None


def op_list_ids(params):
    """@brief index.list_ids.

    @param params `index`.
    @return The ids.
    """
    index = index_for_op(params, "key", "user_id")
    return encode(sdk(index.list_ids))


def op_train(params):
    """@brief index.train.

    @param params `index` and the optional training settings.
    @return None.
    """
    index = index_for_op(params, "key", "user_id")
    settings = pick_params(
        params, "n_lists", "batch_size", "max_iters", "tolerance", "max_memory"
    )
    sdk(index.train, **settings)
    return None


QUERY_OPTIONS = (
    "top_k",
    "n_probes",
    "filters",
    "include",
    "greedy",
    "rerank_mult",
    "text",
    "text_fields",
    "text_field_weights",
    "require_all_terms",
    "alpha",
    "rrf_k",
    "window_mult",
)


def op_query(params):
    """@brief index.query, through the variant's query method.

    @param params `index`, `vectors` and/or `contents`, and the query options.
    @return One list of results per query row.
    @throws Unsupported In the binary variant, `contents` or no `vectors`.
    """
    index = index_for_op(params, "key", "user_id")
    options = pick_params(params, *QUERY_OPTIONS)
    if VARIANT == "binary":
        reject_params(params, "contents")
        if "vectors" not in params:
            raise Unsupported("query_binary needs vectors")
        results = sdk(
            index.query_binary,
            query_vectors=binary_vectors(params["vectors"]),
            **options,
        )
    else:
        if "vectors" in params:
            options["query_vectors"] = json_vectors(params["vectors"])
        if "contents" in params:
            options["query_contents"] = params["contents"]
        results = sdk(index.query, **options)
    row_count = query_row_count(params.get("vectors"), params.get("contents"))
    return encode(one_list_per_row(results, row_count))


def op_query_metadata(params):
    """@brief index.query_metadata.

    @param params `index` and the optional filter, ordering and text options.
    @return The matching rows.
    """
    index = index_for_op(params, "key", "user_id")
    options = pick_params(
        params,
        "filters",
        "top_k",
        "order_by",
        "ascending",
        "text",
        "text_fields",
        "text_field_weights",
        "require_all_terms",
    )
    return encode(sdk(index.query_metadata, **options))


def op_create_user(params):
    """@brief index.create_user.

    The driver keeps the returned API key and acts as the user through it.

    @param params `index`, `permissions`.
    @return `{user_id, api_key}`.
    """
    index = index_for_op(params, "key", "user_id", "user_kek")
    return encode(sdk(index.create_user, require_param(params, "permissions")))


def op_list_users(params):
    """@brief index.list_users.

    @param params `index`.
    @return `{user_id, permissions}` per user.
    """
    index = index_for_op(params, "key")
    return encode(sdk(index.list_users))


def op_delete_user(params):
    """@brief index.delete_user.

    @param params `index`, `user_id` as bytes or the SDK's hex string.
    @return None.
    """
    index = index_for_op(params, "key")
    user_id = require_param(params, "user_id")
    if isinstance(user_id, bytes):
        user_id = user_id.hex()
    sdk(index.delete_user, user_id)
    return None


# index.describe, index.num_vectors and client.training_status have no SDK
# method, and the client.kms_* blob ops are core only; they stay out of OPS so
# they reply -32601.
OPS = {
    "client.new": op_client_new,
    "client.list_indexes": op_list_indexes,
    "client.health": op_health,
    "client.create_index": op_create_index,
    "client.load_index": op_load_index,
    "index.delete_index": op_delete_index,
    "index.upsert": op_upsert,
    "index.get": op_get,
    "index.delete": op_delete,
    "index.list_ids": op_list_ids,
    "index.train": op_train,
    "index.query": op_query,
    "index.query_metadata": op_query_metadata,
    "index.create_user": op_create_user,
    "index.list_users": op_list_users,
    "index.delete_user": op_delete_user,
}

# The service's KMS slot names, as the driver reads them (internal/run/secrets.go).
KMS_SLOT_VARIABLES = (
    "CYBORGDB_TESTS_KMS_NAME_AWS_KMS",
    "CYBORGDB_KMS_NAME_REAL",
    "CYBORGDB_TESTS_KMS_NAME_AWS",
    "CYBORGDB_KMS_NAME_SM",
)


def capabilities():
    """@brief Lists the capabilities this adapter implements.

    @return Capability names from ops.yaml.
    """
    names = [
        "vector",
        "auto_dimension",
        "metadata_schema",
        "pattern",
        "bm25",
        "text_fields_sugar",
        "training",
        "storage_precision",
        "rbac",
        "embedding",
        "health",
    ]
    if VARIANT != "binary":
        names.append("ragged_batches")  # JSON items carry each vector on its own
    if any(os.environ.get(variable) for variable in KMS_SLOT_VARIABLES):
        names.append("kms_provider")
    return names


# ---------- errors ----------

# HTTP status -> kind, from errors.yaml. 401 and 403 each name two kinds; the SDK
# can't tell them apart, so each gets one fixed kind and the wrong guesses are
# declared in expected-failures/cyborgdb-py.yaml.
STATUS_KINDS = {
    400: "invalid_argument",
    401: "unauthenticated",
    403: "permission_denied",
    404: "not_found",
    409: "already_exists",
    422: "invalid_argument",
    429: "rate_limited",
    500: "internal",
    502: "unavailable",
    503: "unavailable",
    504: "unavailable",
}


def kind_of(native_error):
    """@brief Maps an SDK exception to an error kind from errors.yaml.

    Uses the SDK's own `kind` once it exposes one (decided 2026-10-13, SDK.md
    area 14); until then, the exception type and status code. Never the
    message. An exception that isn't a typed SDK error is `internal`, so it
    always fails the step.

    @param native_error The exception the SDK raised.
    @return The error kind.
    """
    native_kind = getattr(native_error, "kind", None)
    if isinstance(native_kind, str) and native_kind:
        return native_kind
    if isinstance(native_error, TransportError):
        return "transport"
    status_code = getattr(native_error, "status_code", None)
    if status_code is None and isinstance(native_error, ValidationError):
        return "invalid_argument"
    return STATUS_KINDS.get(status_code, "internal")


def native_description(native_error):
    """@brief Describes an SDK exception for `data.native`.

    @param native_error The exception the SDK raised.
    @return The class name, followed by the status code when there is one.
    """
    status_code = getattr(native_error, "status_code", None)
    class_name = type(native_error).__name__
    return class_name if status_code is None else f"{class_name} {status_code}"


# ---------- dispatch ----------


def sdk_version():
    """@brief Returns the installed cyborgdb version.

    @return The package version.
    """
    try:
        return version("cyborgdb")
    except PackageNotFoundError:
        return getattr(cyborgdb, "__version__", "unknown")


def handle_request(request):
    """@brief Runs one request and builds its response body.

    @param request The decoded JSON-RPC request.
    @return A dict with either `result` or `error`.
    """
    method = request.get("method")
    if method == "hello":
        return {
            "result": {
                "protocol": PROTOCOL,
                "impl": IMPL,
                "version": sdk_version(),
                "capabilities": capabilities(),
            }
        }
    op_function = OPS.get(method)
    if op_function is None:
        return {
            "error": {
                "code": CODE_UNSUPPORTED,
                "message": f"{IMPL} does not implement {method}",
            }
        }
    # A decoding failure never reached the SDK, so it is never a product error.
    try:
        params = decode(request.get("params") or {})
    except Unsupported as error:
        return {"error": {"code": CODE_UNSUPPORTED, "message": str(error)}}
    except Exception as error:  # noqa: BLE001
        return {
            "error": {
                "code": CODE_INVALID_PARAMS,
                "message": f"could not decode params: {error}",
            }
        }
    try:
        return {"result": op_function(params)}
    except Unsupported as error:
        return {"error": {"code": CODE_UNSUPPORTED, "message": str(error)}}
    except BadParams as error:
        return {"error": {"code": CODE_INVALID_PARAMS, "message": str(error)}}
    except SdkError as error:
        return {
            "error": {
                "code": CODE_OP_FAILED,
                "message": str(error.native_error),
                "data": {
                    "kind": kind_of(error.native_error),
                    "native": native_description(error.native_error),
                },
            }
        }
    except Exception:  # noqa: BLE001 - the adapter's own bug, never a product error
        return adapter_error()


def adapter_error():
    """@brief Builds the response for a bug in the adapter itself.

    Must be called from an `except` block.

    @return A -32603 error carrying the traceback.
    """
    return {
        "error": {
            "code": CODE_ADAPTER,
            "message": "adapter error: " + traceback.format_exc(),
        }
    }


def main():
    """@brief Serves requests from fd 3 until the driver closes it."""
    if VARIANT not in VARIANTS:
        sys.exit(f"adapter: CYBORGDB_TESTS_VARIANT={VARIANT!r}, want one of {VARIANTS}")
    try:
        requests_in = os.fdopen(3, "rb", buffering=0)
        responses_out = os.fdopen(4, "wb", buffering=0)
    except OSError:
        sys.exit("adapter: fd 3/4 not open; run under the cyborgdb-tests driver")
    for line in requests_in:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as error:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": CODE_PARSE, "message": str(error)},
            }
        else:
            response = {"jsonrpc": "2.0", "id": request.get("id")}
            try:
                response.update(handle_request(request))
            except Exception:  # noqa: BLE001 - adapter bug; report it, keep serving
                response.update(adapter_error())
        responses_out.write(json.dumps(response, allow_nan=False).encode() + b"\n")


if __name__ == "__main__":
    main()
