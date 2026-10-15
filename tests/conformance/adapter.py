"""cyborgdb-tests adapter for the cyborgdb Python SDK.

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

TYPED = {
    "$f32",
    "$bytes",
    "$i64",
    "$datetime",
    "$float",
    "$tensor",
    "$range",
    "$literal",
}


class Unsupported(Exception):
    pass


class BadParams(Exception):
    pass


class SdkError(Exception):
    """An exception raised by a cyborgdb call: the only failure that gets a kind."""

    def __init__(self, native):
        super().__init__(str(native))
        self.native = native


_loop = None


def sdk(f, *args, **kwargs):
    """Calls into cyborgdb, awaiting the result on the async client.

    Anything raised here is the product's answer; everything else is the adapter's.
    """
    global _loop
    try:
        result = f(*args, **kwargs)
        if inspect.isawaitable(result):
            if _loop is None:
                # One loop for the whole run: the async client's HTTP session
                # is bound to the loop it was first used on.
                _loop = asyncio.new_event_loop()
            result = _loop.run_until_complete(result)
        return result
    except Exception as e:  # noqa: BLE001
        raise SdkError(e) from e


# ---------- typed values ----------


def decode(v):
    """Wire value -> native Python. Only known single-key tags are typed."""
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        if len(v) == 1:
            ((k, x),) = v.items()
            if k in TYPED:
                return decode_typed(k, x)
        return {k: decode(x) for k, x in v.items()}
    return v


def decode_typed(tag, x):
    if tag == "$literal":
        return x
    if tag == "$bytes":
        return base64.b64decode(x, validate=True)
    if tag == "$f32":
        # Plain lists, not numpy: the SDK sends a numpy array down the binary
        # path, so the ops pick the array type per variant.
        if len(x) > 0 and isinstance(x[0], list):
            return [[num(e) for e in row] for row in x]
        return [num(e) for e in x]
    if tag == "$float":
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[x]
    if tag == "$i64":
        return int(x)
    if tag == "$datetime":
        return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
    if tag == "$range":
        return [str(i) for i in range(int(x[0]), int(x[1]))]
    if tag == "$tensor":
        shape = tuple(int(s) for s in x["shape"])
        dtype = {
            "f32": np.float32,
            "f16": np.float16,
            "i32": np.int32,
            "i64": np.int64,
            "u8": np.uint8,
        }[x["dtype"]]
        if 0 in shape:  # mmap can't map zero bytes
            return np.zeros(shape, dtype=dtype)
        return np.memmap(
            x["path"], dtype=dtype, mode="r", offset=int(x["offset"]), shape=shape
        )
    raise BadParams(f"unknown typed value {tag}")


def num(e):
    if isinstance(e, dict) and "$float" in e:
        return decode_typed("$float", e["$float"])
    return e


def encode(v):
    """Native result -> wire value."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"$bytes": base64.b64encode(bytes(v)).decode()}
    if isinstance(v, np.ndarray):
        if v.ndim == 1:
            return {"$f32": [encode_float(float(f)) for f in v]}
        return {"$f32": [[encode_float(float(f)) for f in r] for r in v]}
    if isinstance(v, dict):
        return {k: encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        i = int(v)
        return i if abs(i) <= 2**53 else {"$i64": str(i)}
    if isinstance(v, (float, np.floating)):
        return encode_float(float(v))
    return v


def encode_float(f):
    if math.isnan(f):
        return {"$float": "nan"}
    if math.isinf(f):
        return {"$float": "inf" if f > 0 else "-inf"}
    return f


def f32(v):
    return {"$f32": [encode_float(float(x)) for x in v]}


# ---------- handles ----------

clients = {}  # name -> Client or AsyncClient
indexes = {}  # name -> EncryptedIndex or AsyncEncryptedIndex


def need(p, k):
    if k not in p:
        raise BadParams(f"missing {k}")
    return p[k]


def client_of(p):
    name = need(p, "client")
    if name not in clients:
        raise BadParams(f"unknown client handle {name!r}")
    return clients[name]


def index_of(p):
    name = need(p, "index")
    if name not in indexes:
        raise BadParams(f"unknown index handle {name!r}")
    return indexes[name]


def pick(p, *names, rename=None):
    rename = rename or {}
    return {rename.get(n, n): p[n] for n in names if n in p}


# ---------- ops ----------

OPS = {}


def capabilities():
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


def kind_of(e):
    if isinstance(e, TransportError):
        return "transport"
    status = getattr(e, "status_code", None)
    if status is None and isinstance(e, ValidationError):
        return "invalid_argument"
    return STATUS_KINDS.get(status, "internal")


def native_of(e):
    status = getattr(e, "status_code", None)
    name = type(e).__name__
    return name if status is None else f"{name} {status}"


# ---------- dispatch ----------


def sdk_version():
    try:
        return version("cyborgdb")
    except PackageNotFoundError:
        return getattr(cyborgdb, "__version__", "unknown")


def handle(req):
    method = req.get("method")
    if method == "hello":
        return {
            "result": {
                "protocol": PROTOCOL,
                "impl": IMPL,
                "version": sdk_version(),
                "capabilities": capabilities(),
            }
        }
    fn = OPS.get(method)
    if fn is None:
        return {
            "error": {
                "code": CODE_UNSUPPORTED,
                "message": f"{IMPL} does not implement {method}",
            }
        }
    # A decoding failure never reached the SDK, so it is never a product error.
    try:
        params = decode(req.get("params") or {})
    except Unsupported as e:
        return {"error": {"code": CODE_UNSUPPORTED, "message": str(e)}}
    except Exception as e:  # noqa: BLE001
        return {
            "error": {
                "code": CODE_INVALID_PARAMS,
                "message": f"could not decode params: {e}",
            }
        }
    try:
        return {"result": fn(params)}
    except Unsupported as e:
        return {"error": {"code": CODE_UNSUPPORTED, "message": str(e)}}
    except BadParams as e:
        return {"error": {"code": CODE_INVALID_PARAMS, "message": str(e)}}
    except SdkError as e:
        return {
            "error": {
                "code": CODE_OP_FAILED,
                "message": str(e.native),
                "data": {"kind": kind_of(e.native), "native": native_of(e.native)},
            }
        }
    except Exception:  # noqa: BLE001 - the adapter's own bug, never a product error
        return adapter_error()


def adapter_error():
    return {
        "error": {
            "code": CODE_ADAPTER,
            "message": "adapter error: " + traceback.format_exc(),
        }
    }


def main():
    if VARIANT not in VARIANTS:
        sys.exit(f"adapter: CYBORGDB_TESTS_VARIANT={VARIANT!r}, want one of {VARIANTS}")
    try:
        rin = os.fdopen(3, "rb", buffering=0)
        wout = os.fdopen(4, "wb", buffering=0)
    except OSError:
        sys.exit("adapter: fd 3/4 not open; run under the cyborgdb-tests driver")
    for line in rin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": CODE_PARSE, "message": str(e)},
            }
        else:
            resp = {"jsonrpc": "2.0", "id": req.get("id")}
            try:
                resp.update(handle(req))
            except Exception:  # noqa: BLE001 - adapter bug; report it, keep serving
                resp.update(adapter_error())
        wout.write(json.dumps(resp, allow_nan=False).encode() + b"\n")


if __name__ == "__main__":
    main()
