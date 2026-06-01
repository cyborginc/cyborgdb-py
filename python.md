# KMS + Multi-Tenancy Re-Implementation Guide — Python SDK (cyborgdb-py)

A step-by-step guide for re-applying the KMS / multi-tenancy work on a
fresh branch off updated `origin/main` in **cyborgdb-py**. Mirrors the
structure of the Go SDK's [`kms.md`](./kms.md).

The reference implementation lives on the local `multi-tenancy` branch
of cyborgdb-py (commits `d2b71fc → 4ff7e64`, currently 5 commits ahead
of `origin/main`). Merge-base with `origin/main` is
`0e0acd3 Diskivf (#75)`. All diffs in this guide are
`origin/main..multi-tenancy` of cyborgdb-py.

> **Working assumption:** `origin/main` of cyborgdb-py has already
> landed `Diskivf (#75)`, which deleted the IVF model files,
> flattened `CreateIndexRequest`, and dropped the `IndexConfig` union.
> That work does **not** need to be re-done here. This guide covers
> only the KMS-specific delta on top.

---

## 1. What this PR does, in one paragraph

Track cyborgdb-service's `multi-tenancy` branch — specifically the
Phase 1 per-index KMS routing slice. Regenerate the OpenAPI client
against the new `CreateIndexRequest` shape, make `index_key`
optional, add `kms_name` to `Client.create_index`, and add offline +
live BYOK tests. Service-side behavior: `CreateIndexRequest` now
includes an optional `kms_name` (and `index_key` is optional);
`IndexOperationRequest.index_key` is also optional so KMS-backed
indexes can be loaded without an SDK-held key.

---

## 2. Source-of-truth spec

- Copy `openapi.json` verbatim from the reference branch (currently
  `cyborgdb-py/multi-tenancy`, `info.version 0.16.1`). It is identical
  to the spec already in `cyborgdb-go` after the Go port — the two
  repos share the exact same bytes.
- The spec on `origin/main` is **`0.16.0`** and lacks `kms_name`;
  this PR bumps it to **`0.16.1`**.

Sanity-check the spec:

```bash
python3 -c "
import json
d = json.load(open('openapi.json'))
print('version:', d['info']['version'])  # expect 0.16.1
print('CIR keys:', list(d['components']['schemas']['CreateIndexRequest']['properties'].keys()))
# expect: ['index_name', 'kms_name', 'index_key', 'dimension', 'embedding_model', 'metric', 'storage_precision']
print('CIR required:', d['components']['schemas']['CreateIndexRequest'].get('required'))
# expect: ['index_name']
print('IOR required:', d['components']['schemas']['IndexOperationRequest'].get('required'))
# expect: ['index_name']
"
```

---

## 3. Regenerate `cyborgdb/openapi_client/`

### 3.1 Add `openapitools.json`

Add at the repo root:

```json
{
  "$schema": "./node_modules/@openapitools/openapi-generator-cli/config.schema.json",
  "spaces": 2,
  "generator-cli": {
    "version": "7.22.0"
  }
}
```

### 3.2 Update `update-openapi-client.sh`

Prefer the npm-distributed `openapi-generator-cli` (pins via
`openapitools.json`); fall back to brew's `openapi-generator` if npm
isn't installed. Replace the existence check + invocation block:

```sh
# Pick a generator binary. Prefer the npm wrapper because it pins via
# openapitools.json — reproducible across machines. Fall back to the
# brew Java binary if that's what the environment has.
if command -v openapi-generator-cli &> /dev/null; then
    GENERATOR=openapi-generator-cli
elif command -v openapi-generator &> /dev/null; then
    GENERATOR=openapi-generator
else
    echo "Error: no OpenAPI generator found."
    echo "Install one of:"
    echo "    npm install -g @openapitools/openapi-generator-cli   (recommended)"
    echo "    brew install openapi-generator"
    exit 1
fi

echo "Generating client with $GENERATOR..."
"$GENERATOR" generate \
    -i openapi.json \
    -g python \
    -o . \
    ...
```

### 3.3 Run it

```bash
./update-openapi-client.sh
```

Expected diffs in `cyborgdb/openapi_client/models/`:

- `create_index_request.py` — gains `kms_name: Optional[StrictStr]`;
  `index_key` flips from `StrictStr` to `Optional[StrictStr]`.
- `index_operation_request.py` — `index_key` flips to
  `Optional[StrictStr]`.
- `query_request.py`, `upsert_request.py`, `get_request.py`,
  `delete_request.py`, `train_request.py`, `list_ids_request.py` —
  `index_key` flips to optional.
- No new files; no files deleted (`Diskivf #75` already removed the
  IVF model files).

Any pre-existing custom fixups in `update-openapi-client.sh` (e.g.,
`Contents` anyOf patches, response-model patches) stay as-is.

---

## 4. Hand-written code changes

All paths are inside `cyborgdb/` unless noted.

### 4.1 `cyborgdb/client/client.py`

#### Add a module-level validation helper

The `multi-tenancy` branch factored validation out of `create_index` /
`load_index` into a reusable helper. Put it near the top of the
module (after the imports, before the `Client` class):

```python
def _validate_index_key(index_key: bytes) -> None:
    """Raise ValueError unless ``index_key`` is a 32-byte ``bytes`` object."""
    if not isinstance(index_key, bytes) or len(index_key) != 32:
        raise ValueError("index_key must be a 32-byte bytes object")
```

#### Rewrite `Client.create_index`

Two-mode contract (`index_key` only / `kms_name` only — exactly one;
supplying both is a service-side 400):

```python
def create_index(
    self,
    index_name: str,
    index_key: Optional[bytes] = None,
    kms_name: Optional[str] = None,
    dimension: Optional[int] = None,
    embedding_model: Optional[str] = None,
    metric: Optional[str] = None,
    storage_precision: Optional[str] = None,
) -> EncryptedIndex:
    """
    Create and return a new encrypted DiskIVF index.

    At least one of ``index_key`` or ``kms_name`` must be provided, and
    the service accepts exactly one of them:

    - ``index_key`` only — the SDK supplies the 32-byte wrapping key; the
      service records the index as ``provider: none`` and does no KMS
      round-trips. The same key must be re-supplied to ``load_index``.
    - ``kms_name`` only — the service generates the key and wraps it under
      the named ``kms.registry`` entry (``aws-kms`` / ``aws``); the SDK
      never sees the plaintext key, and ``load_index`` needs no key.

    Supplying both is forwarded as-is and rejected by the service with a
    400, for every provider: the named slot already determines the key
    source. Note that ``none`` is not a registry slot type — the no-KMS
    path is reached by omitting ``kms_name``.
    """
    if index_key is None and kms_name is None:
        raise ValueError(
            "create_index requires index_key, kms_name, or both"
        )

    if index_key is not None:
        _validate_index_key(index_key)

    try:
        key_hex = (
            binascii.hexlify(index_key).decode("ascii")
            if index_key is not None
            else None
        )

        request = CreateIndexRequest(
            index_name=index_name,
            index_key=key_hex,
            kms_name=kms_name,
            dimension=dimension,
            embedding_model=embedding_model,
            metric=metric,
            storage_precision=storage_precision,
        )

        self.api.create_index_v1_indexes_create_post(
            create_index_request=request,
            _headers={
                "X-API-Key": self.config.api_key["X-API-Key"],
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        return EncryptedIndex(
            index_name=index_name,
            index_key=index_key,
            api=self.api,
            api_client=self.api_client,
        )
    except ApiException as e:
        error_msg = f"Failed to create index: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except ValidationError as ve:
        error_msg = f"Validation error while creating index: {ve}"
        logger.error(error_msg)
        raise ValueError(error_msg)
```

#### Rewrite `Client.load_index`

Make `index_key` optional. For KMS-backed indexes the service resolves
the DEK from the stored `KMSBlob`; the SDK doesn't need to supply it.

```python
def load_index(
    self,
    index_name: str,
    index_key: Optional[bytes] = None,
) -> EncryptedIndex:
    """
    Load an existing encrypted index by name.

    ``index_key`` is required for ``provider: none`` indexes (the SDK owns
    the KEK). For KMS-backed indexes the service resolves the DEK via the
    stored ``KMSBlob``, so ``index_key`` can be omitted.
    """
    if index_key is not None:
        _validate_index_key(index_key)

    try:
        return EncryptedIndex(
            index_name=index_name,
            index_key=index_key,
            api=self.api,
            api_client=self.api_client,
        )
    except ApiException as e:
        error_msg = f"Failed to load index: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
```

### 4.2 `cyborgdb/client/encrypted_index.py`

#### `__init__` accepts `Optional[bytes]`; pre-compute the hex

The `multi-tenancy` branch caches the hex encoding once in `__init__`
so `_key_to_hex` is a cheap accessor:

```python
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
    self._index_config = None
```

#### Rewrite `_key_to_hex`

Return `Optional[str]` and read from the cached field:

```python
def _key_to_hex(self) -> Optional[str]:
    """Hex-encoded key for API calls, or ``None`` for KMS-backed indexes.
    Computed once in ``__init__`` since the key never changes."""
    return self._index_key_hex
```

#### Data-plane callsites

There are ~13 occurrences of `index_key=self._key_to_hex()` across
`encrypted_index.py` (in `query`, `upsert`, `get`, `delete`,
`is_trained`, `is_training`, `list_ids`, `train`, `delete_index`,
etc.). **They don't need to change** — `_key_to_hex` now returns
`Optional[str]`, and the regenerated request models accept
`Optional[StrictStr]` for `index_key`. The plumbing is transparent.

Grep to confirm none of them need rewriting:

```bash
grep -n "index_key=self._key_to_hex" cyborgdb/client/encrypted_index.py
```

Every line should already be in the form `index_key=self._key_to_hex(), index_name=…` and remain unchanged.

### 4.3 Type imports

If `Optional` isn't already imported at the top of `client.py` /
`encrypted_index.py`, add it:

```python
from typing import Optional
```

(Both files already import `Optional` on `origin/main` for other
parameters — confirm before adding.)

---

## 5. Test changes

### 5.1 `tests/test_api_contract.py` — add `TestSDKConstructionOffline`

Mirror of the Go SDK's offline test, slightly expanded:

```python
from cyborgdb.openapi_client.models import (
    CreateIndexRequest,
    IndexOperationRequest,
    QueryRequest,
    UpsertRequest,
    GetRequest,
    DeleteRequest,
    TrainRequest,
    ListIDsRequest,
)


class TestSDKConstructionOffline(unittest.TestCase):
    """SDK-side construction and validation tests that do not require a live
    cyborgdb-service. Exercises the optional-key / KMS paths added when the
    service moved to per-index KMS routing."""

    def setUp(self):
        # Client.__init__ does not make any network calls.
        self.client = cyborgdb.Client(
            base_url="http://localhost:8000", api_key="offline-test-key"
        )

    def test_create_index_requires_key_or_kms_name(self):
        with self.assertRaises(ValueError) as ctx:
            self.client.create_index(index_name="x")
        self.assertIn("index_key", str(ctx.exception))
        self.assertIn("kms_name", str(ctx.exception))

    def test_create_index_request_serializes_kms_name(self):
        req = CreateIndexRequest(index_name="x", kms_name="vendor-slot")
        payload = req.to_dict()
        self.assertEqual(payload["index_name"], "x")
        self.assertEqual(payload["kms_name"], "vendor-slot")
        self.assertNotIn("index_key", payload)

    def test_load_index_without_key_builds_keyless_request(self):
        req = IndexOperationRequest(index_name="x")
        payload = req.to_dict()
        self.assertEqual(payload["index_name"], "x")
        self.assertNotIn("index_key", payload)

    def test_load_index_explicit_none_matches_omitted(self):
        # Client.load_index normally pings the service via a describe call
        # to validate existence. Without a live server we just need to confirm
        # that explicit None doesn't trip key-length validation.
        try:
            self.client.load_index("x", None)
        except ValueError as e:
            self.assertNotIn("32-byte", str(e))
        except Exception:
            pass  # network failure (no server) — fine

    def test_encrypted_index_handles_none_key(self):
        idx = cyborgdb.EncryptedIndex(
            index_name="x",
            index_key=None,
            api=self.client.api,
            api_client=self.client.api_client,
        )
        self.assertIsNone(idx._key_to_hex())

        req = IndexOperationRequest(
            index_name=idx.index_name, index_key=idx._key_to_hex()
        )
        payload = req.to_dict()
        self.assertEqual(payload["index_name"], "x")
        self.assertIsNone(payload.get("index_key"))

    def test_all_data_plane_requests_accept_none_key(self):
        models_and_kwargs = [
            (QueryRequest, {"index_name": "x", "query_vectors": [0.0]}),
            (UpsertRequest, {"index_name": "x", "items": []}),
            (GetRequest, {"index_name": "x", "ids": ["a"]}),
            (DeleteRequest, {"index_name": "x", "ids": ["a"]}),
            (TrainRequest, {"index_name": "x"}),
            (ListIDsRequest, {"index_name": "x"}),
        ]
        for model_cls, kwargs in models_and_kwargs:
            with self.subTest(model=model_cls.__name__):
                req = model_cls(index_key=None, **kwargs)
                payload = req.to_dict()
                self.assertIsNone(payload.get("index_key"))
```

### 5.2 `tests/test_kms_byok.py` (new file)

Env-gated live BYOK tests covering all three KMS posture variants.
Gating envs:

- `CYBORGDB_KMS_NAME_REAL` — `provider: aws-kms` slot (HSM-resident KEK)
- `CYBORGDB_KMS_NAME_SM` — `provider: aws` slot (Secrets Manager KEK)
- `CYBORGDB_KMS_NAME_NONE` — `provider: none` slot (SDK supplies KEK)

Structure: three `unittest.TestCase` classes (one per posture)
sharing a `_KMSRoundTripBase` mixin with the data-plane assertions
(`test_03_upsert_and_query`, `test_04_other_data_plane_methods`).
Each class defines `test_01_create_*` and `test_02_load_*` for the
combinations that posture supports:

| Class | `kms_name` | SDK key on create | `load_index` form |
|---|---|---|---|
| `TestKMSReal` | `KMS_NAME_REAL` | none | `client.load_index(name)` (no key) |
| `TestKMSSecretsManager` | `KMS_NAME_SM` | none | `client.load_index(name)` (no key) |
| `TestProviderNone` | *(omitted)* | `cyborgdb.Client.generate_key()` | `client.load_index(name, key)` |

Skip cleanly via `@unittest.skipUnless(KMS_NAME_X, "...")` when the
env var isn't set. Pull `.env.local` via
`from dotenv import load_dotenv; load_dotenv(".env.local")` so a
fresh shell isn't required.

The data-plane mixin should exercise every method whose request model
gained the optional-key path (`upsert`, `query`, `get`, `list_ids`,
`delete`, `is_trained`, `is_training`). For real-KMS variants, these
all run without an SDK-held key — that's the unique regression risk
of the new keyless path.

Full reference at the equivalent file in the Go SDK
(`test/kms_byok_test.go`) — port the structure 1:1.

### 5.3 No deletions needed

`Diskivf #75` already removed the mixed-type tests and the IVF tests
from `comprehensive_test.py` etc. Nothing further to delete.

---

## 6. README

Add a "Bring Your Own Key (BYOK) via KMS" section showing three code
blocks (drop straight under "Advanced Usage" or your equivalent
section header):

```python
# KMS-backed create — no index_key from the SDK side.
# "vendor-kms-slot" must match a kms.registry entry in cyborgdb.yaml.
index = client.create_index(
    index_name="kms-backed-index",
    kms_name="vendor-kms-slot",
    dimension=128,
    metric="euclidean",
)

# Reopening the index later doesn't require a key either; the service
# resolves the data key from the index's stored KMS envelope.
loaded = client.load_index("kms-backed-index")
```

```python
# No-KMS path — SDK supplies the key. Pass index_key only, omit kms_name
# (the service records this index as provider: none). Supplying both is a 400.
index = client.create_index(
    index_name="sdk-keyed-index",
    index_key=my_key_bytes,
    dimension=128,
)
```

Plus an operator-config callout pointing at `BYOK.md` in
cyborgdb-service for the AWS IAM + `kms.registry` setup.

---

## 7. Files touched, at a glance

```
Hand-written:
  cyborgdb/client/client.py            — _validate_index_key helper, create_index
                                          + load_index two-mode contract
  cyborgdb/client/encrypted_index.py   — Optional[bytes] __init__, cached _index_key_hex,
                                          _key_to_hex returns Optional[str]
  README.md                            — BYOK section
  update-openapi-client.sh             — npm-first generator pick with brew fallback
  openapitools.json                    — new, pins generator version

Spec / generated (run ./update-openapi-client.sh; do not hand-edit):
  openapi.json                                    — replaced from cyborgdb-py
                                                     multi-tenancy (v0.16.1)
  cyborgdb/openapi_client/models/*.py             — kms_name added to
                                                     CreateIndexRequest; index_key
                                                     flipped to Optional[StrictStr]
                                                     across all request models

Tests:
  tests/test_api_contract.py           — +TestSDKConstructionOffline (~100 lines)
  tests/test_kms_byok.py               — new, env-gated BYOK round-trips

No deletions:
  IVF model files (index_config.py, index_ivf_*.py) were already removed
  by Diskivf #75 on origin/main — no work to do.
```

---

## 8. Verification checklist (post-port)

```bash
# Install package in editable mode (if not already)
pip install -e .

# Offline tests — should pass with no running service
python -m pytest tests/test_api_contract.py::TestSDKConstructionOffline -v

# Full collection — confirm everything still imports against the
# regenerated client
python -m pytest tests/ --collect-only -q | tail

# Live BYOK (only after kms.registry is wired in cyborgdb.yaml AND the
# CYBORGDB_KMS_NAME_* env vars are exported)
python -m pytest tests/test_kms_byok.py -v
```

Expected wire-shape behavior for the offline test (matches the Go SDK):

- `kms_name` only → `kms_name` present, `index_key` absent (`.to_dict()`
  with `exclude_none=True` semantics).
- `index_key` only → `index_key` present as 64-char hex, `kms_name`
  absent.
- Both set → both present on the wire (the SDK forwards them unchanged); the
  service then rejects the pair with a 400 — no provider accepts both.
- Neither set → `client.create_index` raises `ValueError` mentioning
  both `index_key` and `kms_name` before any wire traffic.

---

## 9. Known follow-ups (out of scope for the port)

- **Live BYOK CI**: `tests/test_kms_byok.py` needs a CI slot wiring
  the three `CYBORGDB_KMS_NAME_*` env vars against a configured
  `cyborgdb.yaml`. Same gap exists on the Go side.
- **Phase 2+ scope**: Vendor / admin / user routes and RBAC are still
  pending in cyborgdb-service; no SDK surface for them here.
- **Typed error for both-against-real-KMS 400**: the service rejects
  this misconfiguration with a 400 (see
  `cyborgdb_service/api/routes/indexes.py::_resolve_kek`). On the
  Python side this surfaces as `ApiException` re-wrapped into
  `ValueError` by the `try/except` in `create_index`. If this
  becomes a common caller mistake, a typed subclass would be
  friendlier.

---

## 10. Source artifacts on the local cyborgdb-py branch

These exist on the local `multi-tenancy` branch and can be copied
verbatim into the fresh branch — they're not affected by `origin/main`
churn:

```bash
cd /Users/cyborg-jim/Documents/repos/cyborgdb-py

# Spec + generator pin
git show multi-tenancy:openapi.json        > openapi.json
git show multi-tenancy:openapitools.json   > openapitools.json

# Tests
git show multi-tenancy:tests/test_kms_byok.py > tests/test_kms_byok.py

# Generator script (already updated for openapi-generator-cli + brew fallback)
git show multi-tenancy:update-openapi-client.sh > update-openapi-client.sh
chmod +x update-openapi-client.sh
```

Then run `./update-openapi-client.sh` to regenerate
`cyborgdb/openapi_client/`, and apply the hand-edits from §4 to
`client.py` and `encrypted_index.py`.

---

## 11. Cross-reference

This guide is the Python sibling of:

- `kms.md` — Go SDK re-implementation guide (cyborgdb-go).
- `pr.md` — Go SDK PR description (the completed PR #27 in cyborgdb-go).
- `review.md` — Go SDK PR review (cyborgdb-go PR #27).

The service-side contract (which `kms.registry` entry types exist and
what their semantics are) is defined in:

- `cyborgdb-service/BYOK.md` — operator + customer setup walkthrough.
- `cyborgdb-service/cyborgdb.example.yaml` — `kms.registry` shape.
- `cyborgdb-service/cyborgdb_service/api/routes/indexes.py::_resolve_kek`
  — the authoritative server-side validator: exactly one of `index_key` /
  `kms_name`; supplying both is a 400 for every provider.
