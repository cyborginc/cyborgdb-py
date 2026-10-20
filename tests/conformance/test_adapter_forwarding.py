"""@file test_adapter_forwarding.py
@brief Recording-fake checks for the conformance adapter (cyborgdb-tests ADAPTERS.md,
"Recording fake").

The adapter's real dispatch code runs against a fake `cyborgdb` module whose
clients and indexes record every call. The fake raises the SDK's real exception
classes, so the error mapping is checked against the types the SDK ships. No
service is needed.
"""

import asyncio
import base64
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import cyborgdb.exceptions as sdk_exceptions
from cyborgdb.exceptions import (
    AuthenticationError,
    ConflictError,
    CyborgDBError,
    NotFoundError,
    RateLimitError,
    ServiceError,
    TransportError,
    ValidationError,
)

ADAPTER_PATH = Path(__file__).resolve().parent / "adapter.py"
KEY = {"$bytes": base64.b64encode(bytes(range(32))).decode()}
CODE_OP_FAILED = -32000
CODE_UNSUPPORTED = -32601
CODE_INVALID_PARAMS = -32602
CODE_ADAPTER = -32603

INDEX_RETURNS = {
    "get": [],
    "list_ids": [],
    "describe": {"index_name": "r", "dimension": 0},
    "get_num_vectors": 0,
    "query": [[]],
    "query_binary": [[]],
    "query_metadata": [],
    "create_user": {"user_id": "00ff", "api_key": "cdbk_fake"},
    "list_users": [],
}


class Recorder:
    """@brief Records every method call into a shared log.

    `returns[name]` is what a call returns; `raise_on[name]` makes it raise.
    With `is_async`, each call returns a coroutine, as the async SDK does.
    """

    def __init__(self, log, prefix, returns, is_async):
        self._log = log
        self._prefix = prefix
        self._returns = returns
        self._is_async = is_async
        self.raise_on = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self._log.append((f"{self._prefix}.{name}", args, kwargs))

            def finish():
                if name in self.raise_on:
                    raise self.raise_on[name]
                return self._returns.get(name)

            if self._is_async:

                async def awaited():
                    return finish()

                return awaited()
            return finish()

        return call


def fake_cyborgdb(log, index):
    """@brief Builds a fake `cyborgdb` module whose clients record calls.

    @param log The shared call log.
    @param index The Recorder every create_index / load_index returns.
    @return The fake module.
    """
    module = types.ModuleType("cyborgdb")
    module.__version__ = "fake"

    def client_class(class_name, is_async):
        def construct(*args, **kwargs):
            log.append((class_name, args, kwargs))
            returns = {
                "create_index": index,
                "load_index": index,
                "list_indexes": [],
                "get_health": {"status": "healthy"},
            }
            return Recorder(log, "client", returns, is_async)

        return construct

    module.Client = client_class("Client", is_async=False)
    module.AsyncClient = client_class("AsyncClient", is_async=True)
    return module


class Harness:
    """@brief One adapter instance, loaded fresh against the fake module."""

    def __init__(self, variant, monkeypatch):
        if variant is None:
            monkeypatch.delenv("CYBORGDB_TESTS_VARIANT", raising=False)
        else:
            monkeypatch.setenv("CYBORGDB_TESTS_VARIANT", variant)
        self.log = []
        self.index = Recorder(
            self.log, "index", dict(INDEX_RETURNS), is_async=variant == "async"
        )
        monkeypatch.setitem(
            sys.modules, "cyborgdb", fake_cyborgdb(self.log, self.index)
        )
        monkeypatch.setitem(sys.modules, "cyborgdb.exceptions", sdk_exceptions)
        spec = importlib.util.spec_from_file_location(
            "adapter_under_test", ADAPTER_PATH
        )
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.call("client.new", {"as": "c0", "base_url": "http://fake"})
        self.call(
            "client.create_index", {"client": "c0", "as": "i0", "name": "r", "key": KEY}
        )
        self.log.clear()

    def call(self, method, params):
        """@brief Runs one request through the adapter's dispatch.

        @return The response body: `{"result": ...}` or `{"error": ...}`.
        """
        return self.adapter.handle_request({"method": method, "params": params})

    def calls(self, name):
        """@brief Returns the recorded `(args, kwargs)` of every call to `name`."""
        return [(args, kwargs) for called, args, kwargs in self.log if called == name]

    def only_call(self, name):
        """@brief Returns the single recorded call to `name`, failing otherwise."""
        recorded = self.calls(name)
        assert len(recorded) == 1, f"expected one {name} call, got {self.log}"
        return recorded[0]


@pytest.fixture(params=["json", "binary", "async"])
def harness(request, monkeypatch):
    return Harness(request.param, monkeypatch)


@pytest.fixture
def json_harness(monkeypatch):
    return Harness("json", monkeypatch)


@pytest.fixture
def binary_harness(monkeypatch):
    return Harness("binary", monkeypatch)


# ---------- omitted, null, empty, zero and false ----------

# (method, params, recorded call, args, kwargs). The `index` param is added for
# index ops; every case must reach the SDK with exactly these arguments.
FORWARDING_CASES = [
    ("client.new", {"as": "c1", "base_url": "http://x"}, "Client", ("http://x",), {}),
    (
        "client.new",
        {"as": "c1", "base_url": "http://x", "api_key": ""},
        "Client",
        ("http://x",),
        {"api_key": ""},
    ),
    ("client.list_indexes", {"client": "c0"}, "client.list_indexes", (), {}),
    ("client.health", {"client": "c0"}, "client.get_health", (), {}),
    (
        "client.create_index",
        {"client": "c0", "as": "i1", "name": "n"},
        "client.create_index",
        ("n",),
        {},
    ),
    (
        "client.create_index",
        {
            "client": "c0",
            "as": "i1",
            "name": "",
            "key": KEY,
            "dimension": 0,
            "metric": None,
            "metadata_schema": {},
            "text_fields": [],
            "bm25_k1": 0.0,
        },
        "client.create_index",
        ("",),
        {
            "index_key": bytes(range(32)),
            "dimension": 0,
            "metric": None,
            "metadata_schema": {},
            "text_fields": [],
            "bm25_k1": 0.0,
        },
    ),
    (
        "client.create_index",
        {"client": "c0", "as": "i1", "name": "n", "kms_name": "slot"},
        "client.create_index",
        ("n",),
        {"kms_name": "slot"},
    ),
    (
        "client.load_index",
        {"client": "c0", "as": "i1", "name": "n"},
        "client.load_index",
        ("n",),
        {},
    ),
    (
        "client.load_index",
        {"client": "c0", "as": "i1", "name": "n", "key": KEY},
        "client.load_index",
        ("n",),
        {"index_key": bytes(range(32))},
    ),
    ("index.delete_index", {}, "index.delete_index", (), {}),
    ("index.get", {"ids": ["a"]}, "index.get", (["a"],), {}),
    ("index.get", {"ids": [], "include": []}, "index.get", ([],), {"include": []}),
    ("index.delete", {"ids": []}, "index.delete", ([],), {}),
    ("index.list_ids", {}, "index.list_ids", (), {}),
    ("index.describe", {}, "index.describe", (), {}),
    ("index.num_vectors", {}, "index.get_num_vectors", (), {}),
    ("index.train", {}, "index.train", (), {}),
    (
        "index.train",
        {
            "n_lists": 0,
            "batch_size": 0,
            "max_iters": 0,
            "tolerance": 0.0,
            "max_memory": 0,
        },
        "index.train",
        (),
        {
            "n_lists": 0,
            "batch_size": 0,
            "max_iters": 0,
            "tolerance": 0.0,
            "max_memory": 0,
        },
    ),
    ("index.query_metadata", {}, "index.query_metadata", (), {}),
    (
        "index.query_metadata",
        {"filters": {}, "top_k": 0, "ascending": False, "text": ""},
        "index.query_metadata",
        (),
        {"filters": {}, "top_k": 0, "ascending": False, "text": ""},
    ),
    (
        "index.create_user",
        {"permissions": []},
        "index.create_user",
        ([],),
        {},
    ),
    ("index.list_users", {}, "index.list_users", (), {}),
    ("index.delete_user", {"user_id": "00ff"}, "index.delete_user", ("00ff",), {}),
    (
        "index.delete_user",
        {"user_id": {"$bytes": base64.b64encode(b"\x00\xff").decode()}},
        "index.delete_user",
        ("00ff",),
        {},
    ),
]


@pytest.mark.parametrize(
    "method, params, recorded, args, kwargs",
    FORWARDING_CASES,
    ids=[f"{case[0]}-{index}" for index, case in enumerate(FORWARDING_CASES)],
)
def test_params_forwarded_exactly(harness, method, params, recorded, args, kwargs):
    if method.startswith("index."):
        params = {"index": "i0", **params}
    if recorded == "Client" and harness.adapter.VARIANT == "async":
        recorded = "AsyncClient"
    response = harness.call(method, params)
    assert "error" not in response, response
    assert harness.only_call(recorded) == (args, kwargs)


@pytest.mark.parametrize("variant", ["json", "async"])
def test_json_query_forwards_options_as_given(variant, monkeypatch):
    harness = Harness(variant, monkeypatch)
    options = {
        "top_k": 0,
        "n_probes": 0,
        "filters": {},
        "include": [],
        "greedy": False,
        "rerank_mult": 0,
        "text": "",
        "text_fields": [],
        "text_field_weights": [],
        "require_all_terms": False,
        "alpha": 0.0,
        "rrf_k": 0.0,
        "window_mult": 0,
    }
    harness.call(
        "index.query", {"index": "i0", "vectors": {"$f32": [[1, 2]]}, **options}
    )
    assert harness.only_call("index.query") == (
        (),
        {**options, "query_vectors": [[1, 2]]},
    )


def test_json_query_omitted_options_stay_omitted(json_harness):
    json_harness.call("index.query", {"index": "i0", "contents": "hello"})
    assert json_harness.only_call("index.query") == ((), {"query_contents": "hello"})


def test_binary_query_forwards_options_as_given(binary_harness):
    binary_harness.call(
        "index.query",
        {"index": "i0", "vectors": {"$f32": [[1, 2]]}, "top_k": 0, "include": []},
    )
    args, kwargs = binary_harness.only_call("index.query_binary")
    vectors = kwargs.pop("query_vectors")
    assert (args, kwargs) == ((), {"top_k": 0, "include": []})
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, [[1, 2]])


# ---------- the variant picks the SDK path ----------


@pytest.mark.parametrize(
    "variant, client_class, upsert_method, query_method",
    [
        (None, "Client", "index.upsert", "index.query"),
        ("json", "Client", "index.upsert", "index.query"),
        ("binary", "Client", "index.upsert_binary", "index.query_binary"),
        ("async", "AsyncClient", "index.upsert", "index.query"),
    ],
)
def test_variant_picks_the_sdk_path(
    variant, client_class, upsert_method, query_method, monkeypatch
):
    harness = Harness(variant, monkeypatch)
    harness.call("client.new", {"as": "c1", "base_url": "http://x"})
    harness.call(
        "index.upsert", {"index": "i0", "ids": ["a"], "vectors": {"$f32": [[1, 2]]}}
    )
    harness.call("index.query", {"index": "i0", "vectors": {"$f32": [[1, 2]]}})
    assert [name for name, _, _ in harness.log] == [
        client_class,
        upsert_method,
        query_method,
    ]


def test_async_calls_are_awaited(monkeypatch):
    harness = Harness("async", monkeypatch)
    harness.index.raise_on["list_ids"] = NotFoundError("gone", status_code=404)
    response = harness.call("index.list_ids", {"index": "i0"})
    assert response["error"]["data"]["kind"] == "not_found"
    assert isinstance(harness.adapter._event_loop, asyncio.AbstractEventLoop)


def test_unknown_variant_exits(monkeypatch):
    monkeypatch.setenv("CYBORGDB_TESTS_VARIANT", "xml")
    result = subprocess.run(
        [sys.executable, str(ADAPTER_PATH)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "CYBORGDB_TESTS_VARIANT" in result.stderr


# ---------- batch shapes ----------


def test_json_upsert_forwards_items_unchanged(json_harness):
    items = [
        {"id": "a", "vector": {"$f32": [1, 2]}, "contents": {"$bytes": "aGk="}},
        {"id": "b", "vector": {"$f32": [1]}, "metadata": {"m": None}},
        {"vector": {"$f32": []}},
    ]
    json_harness.call("index.upsert", {"index": "i0", "items": items})
    assert json_harness.only_call("index.upsert") == (
        (
            [
                {"id": "a", "vector": [1, 2], "contents": b"hi"},
                {"id": "b", "vector": [1], "metadata": {"m": None}},
                {"vector": []},
            ],
        ),
        {},
    )


def test_json_upsert_forwards_ids_and_vectors_as_lists(json_harness):
    json_harness.call(
        "index.upsert",
        {"index": "i0", "ids": {"$range": [0, 2]}, "vectors": {"$f32": [[1, 2], [3]]}},
    )
    assert json_harness.only_call("index.upsert") == ((["0", "1"], [[1, 2], [3]]), {})


def test_json_upsert_ids_without_vectors_reaches_the_sdk(json_harness):
    json_harness.call("index.upsert", {"index": "i0", "ids": ["a"]})
    assert json_harness.only_call("index.upsert") == ((["a"],), {})


def test_json_upsert_parallel_metadata_builds_one_item_per_id(json_harness):
    json_harness.call(
        "index.upsert",
        {
            "index": "i0",
            "ids": ["a", "b"],
            "vectors": {"$f32": [[1, 2], [3, 4]]},
            "metadata": [{"x": 1}, None],
        },
    )
    assert json_harness.only_call("index.upsert") == (
        (
            [
                {"id": "a", "vector": [1, 2], "metadata": {"x": 1}},
                {"id": "b", "vector": [3, 4]},
            ],
        ),
        {},
    )


@pytest.mark.parametrize(
    "ids, vectors, metadata",
    [
        (["a", "b"], [[1, 2], [3, 4]], [{"x": 1}]),
        (["a"], [[1, 2], [3, 4]], [{"x": 1}]),
    ],
)
def test_json_upsert_parallel_metadata_mismatch_is_unsupported(
    json_harness, ids, vectors, metadata
):
    response = json_harness.call(
        "index.upsert",
        {"index": "i0", "ids": ids, "vectors": {"$f32": vectors}, "metadata": metadata},
    )
    assert response["error"]["code"] == CODE_UNSUPPORTED
    assert json_harness.log == []


def test_tensor_vectors_go_to_json_as_lists(json_harness, tmp_path):
    path = tmp_path / "vectors.f32"
    np.array([[1, 2], [3, 4]], dtype=np.float32).tofile(path)
    tensor = {
        "$tensor": {"path": str(path), "dtype": "f32", "shape": [2, 2], "offset": 0}
    }
    json_harness.call(
        "index.upsert", {"index": "i0", "ids": ["a", "b"], "vectors": tensor}
    )
    args, _ = json_harness.only_call("index.upsert")
    assert args == (["a", "b"], [[1.0, 2.0], [3.0, 4.0]])
    assert type(args[1]) is list


def test_binary_upsert_ids_vectors_metadata_forwarded_as_given(binary_harness):
    # The SDK checks the counts; the adapter neither checks nor pads them.
    binary_harness.call(
        "index.upsert",
        {
            "index": "i0",
            "ids": ["a", "b"],
            "vectors": {"$f32": [[1, 2], [3, 4]]},
            "metadata": [None],
        },
    )
    args, kwargs = binary_harness.only_call("index.upsert_binary")
    assert args[0] == ["a", "b"]
    assert args[1].shape == (2, 2) and args[1].dtype == np.float32
    assert kwargs == {"metadata": [None]}


def test_binary_upsert_items_become_columns(binary_harness):
    binary_harness.call(
        "index.upsert",
        {
            "index": "i0",
            "items": [
                {"id": "a", "vector": {"$f32": [1, 2]}, "contents": "text"},
                {"id": "b", "vector": {"$f32": [3, 4]}},
            ],
        },
    )
    args, kwargs = binary_harness.only_call("index.upsert_binary")
    assert args[0] == ["a", "b"]
    np.testing.assert_array_equal(args[1], [[1, 2], [3, 4]])
    assert kwargs == {"contents": ["text", None]}


@pytest.mark.parametrize(
    "params",
    [
        {"ids": ["a", "b"], "vectors": {"$f32": [[1, 2], [3]]}},
        {
            "items": [
                {"id": "a", "vector": {"$f32": [1]}},
                {"id": "b", "vector": {"$f32": [1, 2]}},
            ]
        },
        {"items": [{"id": "a", "contents": "text"}]},
        {"items": [{"vector": {"$f32": [1, 2]}}]},
        {"items": [{"id": "a", "vector": {"$f32": [1, 2]}, "extra": 1}]},
        {"ids": ["a"], "vectors": [["x", "y"]]},
        {"ids": ["a"]},
    ],
)
def test_binary_upsert_inexpressible_input_is_unsupported(binary_harness, params):
    response = binary_harness.call("index.upsert", {"index": "i0", **params})
    assert response["error"]["code"] == CODE_UNSUPPORTED
    assert binary_harness.log == []


@pytest.mark.parametrize(
    "params", [{"contents": "hello"}, {"vectors": {"$f32": [[1]]}, "contents": "hello"}]
)
def test_binary_query_by_text_is_unsupported(binary_harness, params):
    response = binary_harness.call("index.query", {"index": "i0", **params})
    assert response["error"]["code"] == CODE_UNSUPPORTED
    assert binary_harness.log == []


def test_upsert_without_items_or_ids_is_bad_params(harness):
    response = harness.call("index.upsert", {"index": "i0", "vectors": {"$f32": [[1]]}})
    assert response["error"]["code"] == CODE_INVALID_PARAMS
    assert harness.log == []


# ---------- params the SDK can't express ----------

UNSUPPORTED_CASES = [
    ("client.new", {"as": "c1", "base_url": "http://x", "storage": {"kind": "disk"}}),
    ("client.new", {"as": "c1", "base_url": "http://x", "cpu_threads": 0}),
    ("client.load_index", {"client": "c0", "as": "i1", "name": "n", "user_id": "u"}),
    ("index.delete_index", {"index": "i0", "key": KEY}),
    ("index.upsert", {"index": "i0", "items": [], "key": KEY}),
    ("index.get", {"index": "i0", "ids": ["a"], "user_id": "u"}),
    ("index.delete", {"index": "i0", "ids": ["a"], "key": KEY}),
    ("index.list_ids", {"index": "i0", "user_id": "u"}),
    ("index.train", {"index": "i0", "key": KEY}),
    ("index.query", {"index": "i0", "vectors": {"$f32": [[1]]}, "key": KEY}),
    ("index.query_metadata", {"index": "i0", "user_id": "u"}),
    ("index.create_user", {"index": "i0", "permissions": ["read"], "user_id": "u"}),
    ("index.create_user", {"index": "i0", "permissions": ["read"], "user_kek": KEY}),
    ("index.list_users", {"index": "i0", "key": KEY}),
    ("index.delete_user", {"index": "i0", "user_id": "u", "key": KEY}),
    ("index.num_vectors", {"index": "i0", "key": KEY}),
    ("client.training_status", {"client": "c0"}),
    ("client.kms_get", {"client": "c0", "name": "n"}),
]


@pytest.mark.parametrize(
    "method, params",
    UNSUPPORTED_CASES,
    ids=[f"{case[0]}-{index}" for index, case in enumerate(UNSUPPORTED_CASES)],
)
def test_inexpressible_params_are_unsupported(harness, method, params):
    response = harness.call(method, params)
    assert response["error"]["code"] == CODE_UNSUPPORTED
    assert "data" not in response["error"]
    assert harness.log == []


# ---------- results ----------


@pytest.mark.parametrize(
    "params, sdk_result, expected",
    [
        ({"vectors": {"$f32": [[1, 2]]}}, [{"id": "a"}], [[{"id": "a"}]]),
        ({"vectors": {"$f32": [[1, 2]]}}, [], [[]]),
        ({"vectors": {"$f32": [[1, 2]]}}, [[{"id": "a"}]], [[{"id": "a"}]]),
        ({"vectors": {"$f32": [[1, 2], [3, 4]]}}, [[], []], [[], []]),
        ({"contents": "hello"}, [{"id": "a"}], [[{"id": "a"}]]),
    ],
)
def test_query_returns_one_list_per_row(json_harness, params, sdk_result, expected):
    json_harness.index._returns["query"] = sdk_result
    response = json_harness.call("index.query", {"index": "i0", **params})
    assert response == {"result": expected}


def test_query_encodes_float_specials(json_harness):
    json_harness.index._returns["query"] = [[{"id": "a", "distance": float("inf")}]]
    response = json_harness.call(
        "index.query", {"index": "i0", "vectors": {"$f32": [[1]]}}
    )
    assert response == {"result": [[{"id": "a", "distance": {"$float": "inf"}}]]}


def test_get_encodes_vector_contents_and_metadata(json_harness):
    json_harness.index._returns["get"] = [
        {"id": "a", "vector": [1.0, 2.0], "contents": "hi", "metadata": '{"k": 1}'},
        {"id": "b", "contents": b"\x00", "metadata": {"k": 2}},
        {"id": "c", "vector": None, "contents": None, "metadata": None},
    ]
    response = json_harness.call("index.get", {"index": "i0", "ids": ["a", "b", "c"]})
    assert response == {
        "result": [
            {
                "id": "a",
                "vector": {"$f32": [1.0, 2.0]},
                "contents": {"$bytes": "aGk="},
                "metadata": {"k": 1},
            },
            {"id": "b", "contents": {"$bytes": "AA=="}, "metadata": {"k": 2}},
            {"id": "c", "vector": None, "contents": None, "metadata": None},
        ]
    }


def test_describe_renames_index_name_and_keeps_every_other_field(harness):
    harness.index._returns["describe"] = {
        "index_name": "r",
        "index_type": "disk_ivf",
        "training_state": "untrained",
        "is_trained": False,
        "bm25": None,
    }
    response = harness.call("index.describe", {"index": "i0"})
    assert response == {
        "result": {
            "name": "r",
            "index_type": "disk_ivf",
            "training_state": "untrained",
            "is_trained": False,
            "bm25": None,
        }
    }


def test_num_vectors_returns_the_count(harness):
    harness.index._returns["get_num_vectors"] = 1000002
    response = harness.call("index.num_vectors", {"index": "i0"})
    assert response == {"result": 1000002}


def test_create_user_returns_the_sdk_result(harness):
    response = harness.call(
        "index.create_user", {"index": "i0", "permissions": ["read"]}
    )
    assert response == {"result": {"user_id": "00ff", "api_key": "cdbk_fake"}}


# ---------- errors ----------


@pytest.mark.parametrize(
    "error, kind, native",
    [
        (ValidationError("v"), "invalid_argument", "ValidationError"),
        (
            ValidationError("v", status_code=400),
            "invalid_argument",
            "ValidationError 400",
        ),
        (
            ValidationError("v", status_code=422),
            "invalid_argument",
            "ValidationError 422",
        ),
        (
            AuthenticationError("a", status_code=401),
            "unauthenticated",
            "AuthenticationError 401",
        ),
        (
            AuthenticationError("a", status_code=403),
            "permission_denied",
            "AuthenticationError 403",
        ),
        (NotFoundError("n", status_code=404), "not_found", "NotFoundError 404"),
        (ConflictError("c", status_code=409), "already_exists", "ConflictError 409"),
        (RateLimitError("r", status_code=429), "rate_limited", "RateLimitError 429"),
        (ServiceError("s", status_code=500), "internal", "ServiceError 500"),
        (ServiceError("s", status_code=501), "internal", "ServiceError 501"),
        (ServiceError("s", status_code=502), "unavailable", "ServiceError 502"),
        (ServiceError("s", status_code=503), "unavailable", "ServiceError 503"),
        (ServiceError("s", status_code=504), "unavailable", "ServiceError 504"),
        (CyborgDBError("x", status_code=405), "internal", "CyborgDBError 405"),
        (CyborgDBError("x"), "internal", "CyborgDBError"),
        (TransportError("t"), "transport", "TransportError"),
        (KeyError("k"), "internal", "KeyError"),
    ],
)
def test_sdk_errors_come_back_as_their_kind(harness, error, kind, native):
    harness.index.raise_on["delete"] = error
    response = harness.call("index.delete", {"index": "i0", "ids": ["a"]})
    assert response["error"]["code"] == CODE_OP_FAILED
    assert response["error"]["data"] == {"kind": kind, "native": native}
    assert len(harness.calls("index.delete")) == 1


def test_sdk_kind_wins_over_the_status_table(json_harness):
    error = AuthenticationError("a", status_code=401)
    error.kind = "wrong_key"
    json_harness.index.raise_on["delete"] = error
    response = json_harness.call("index.delete", {"index": "i0", "ids": ["a"]})
    assert response["error"]["data"]["kind"] == "wrong_key"


def test_client_constructor_errors_get_a_kind(json_harness, monkeypatch):
    def refuse(*args, **kwargs):
        raise CyborgDBError("bad url")

    monkeypatch.setattr(sys.modules["cyborgdb"], "Client", refuse)
    response = json_harness.call("client.new", {"as": "c1", "base_url": "ftp://x"})
    assert response["error"]["code"] == CODE_OP_FAILED
    assert response["error"]["data"]["kind"] == "internal"


def test_adapter_errors_get_no_kind(harness):
    # A row that isn't a dict breaks the adapter's own encoding, after the SDK call.
    harness.index._returns["get"] = [5]
    response = harness.call("index.get", {"index": "i0", "ids": ["a"]})
    assert response["error"]["code"] == CODE_ADAPTER
    assert "data" not in response["error"]


@pytest.mark.parametrize(
    "method, params",
    [
        ("index.get", {"index": "nope", "ids": ["a"]}),
        ("index.get", {"index": "i0"}),
        ("client.create_index", {"client": "c0", "name": "n"}),
        ("index.get", {"index": "i0", "ids": {"$bytes": "!!"}}),
        ("index.get", {"index": "i0", "ids": {"$float": "huge"}}),
    ],
)
def test_bad_params_never_reach_the_sdk(harness, method, params):
    response = harness.call(method, params)
    assert response["error"]["code"] == CODE_INVALID_PARAMS
    assert harness.log == []


def test_unknown_dollar_keys_are_data(json_harness):
    json_harness.call("index.get", {"index": "i0", "ids": {"$nope": 1}})
    assert json_harness.only_call("index.get") == (({"$nope": 1},), {})


# ---------- startup and framing ----------


def test_hello_reports_variant_capabilities(monkeypatch):
    json_names = Harness("json", monkeypatch).call("hello", {"protocol": 1})["result"]
    binary_names = Harness("binary", monkeypatch).call("hello", {"protocol": 1})[
        "result"
    ]
    assert json_names["impl"] == "cyborgdb-py"
    assert "ragged_batches" in json_names["capabilities"]
    assert "ragged_batches" not in binary_names["capabilities"]
    for absent in ("kms_blob", "gpu", "training_status", "error_fields", "free_tier"):
        assert absent not in json_names["capabilities"]
    for present in ("service_auth", "durable_storage"):
        assert present in json_names["capabilities"]
        assert present in binary_names["capabilities"]


@pytest.mark.parametrize(
    "variable", ["CYBORGDB_KMS_NAME_REAL", "CYBORGDB_TESTS_KMS_NAME_AWS"]
)
def test_kms_provider_only_with_kms_slots(monkeypatch, variable):
    for name in (
        "CYBORGDB_TESTS_KMS_NAME_AWS_KMS",
        "CYBORGDB_KMS_NAME_REAL",
        "CYBORGDB_TESTS_KMS_NAME_AWS",
        "CYBORGDB_KMS_NAME_SM",
    ):
        monkeypatch.delenv(name, raising=False)
    harness = Harness("json", monkeypatch)
    assert "kms_provider" not in harness.adapter.capabilities()
    monkeypatch.setenv(variable, "slot")
    assert "kms_provider" in harness.adapter.capabilities()


def test_framing_over_fds_3_and_4():
    """Runs the real adapter process with the installed SDK; no service needed."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol": 1}},
        {"jsonrpc": "2.0", "id": 2, "method": "client.training_status", "params": {}},
    ]
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()

    def place_on_fds_3_and_4():
        # Copy first: a pipe end may already be fd 3 or 4.
        read_copy, write_copy = os.dup(request_read), os.dup(response_write)
        os.dup2(read_copy, 3)
        os.dup2(write_copy, 4)

    process = subprocess.Popen(
        [sys.executable, str(ADAPTER_PATH)],
        preexec_fn=place_on_fds_3_and_4,
        close_fds=False,
        env={**os.environ, "CYBORGDB_TESTS_VARIANT": "json"},
    )
    os.close(request_read)
    os.close(response_write)
    lines = [json.dumps(request) for request in requests] + ["not json", ""]
    with os.fdopen(request_write, "w") as to_adapter:
        to_adapter.write("\n".join(lines) + "\n")
    with os.fdopen(response_read) as from_adapter:
        responses = [json.loads(line) for line in from_adapter]
    assert process.wait(timeout=30) == 0
    assert [response["id"] for response in responses] == [1, 2, None]
    assert responses[0]["result"]["impl"] == "cyborgdb-py"
    assert responses[1]["error"]["code"] == CODE_UNSUPPORTED
    assert responses[2]["error"]["code"] == -32700
