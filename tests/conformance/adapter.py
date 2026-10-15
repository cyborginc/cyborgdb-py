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


# ---------- ops ----------

OPS = {}


def capabilities():
    """@brief Lists the capabilities this adapter implements.

    @return Capability names from ops.yaml.
    """
    return []


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

    Uses only the exception type and status code, never the message.

    @param native_error The exception the SDK raised.
    @return The error kind.
    """
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
