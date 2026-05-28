"""
KMS BYOK integration tests for the CyborgDB Python SDK.

The service supports two wire encodings for index encryption keys, and
``kms_name`` + ``index_key`` are strictly mutually exclusive on the
create request (the server returns 400 regardless of which slot
``kms_name`` resolves to):

  * **SDK-supplied KEK** — ``index_key`` alone, no ``kms_name``. Service
    records the envelope as ``provider="none"`` and the SDK re-supplies
    the same key on every subsequent request. No KMS registry slot is
    referenced.
  * **KMS-managed KEK** — ``kms_name`` alone, no ``index_key``. Service
    generates a random KEK, wraps it via the named registry slot, and
    resolves it server-side on every subsequent request.

The KMS-managed suites are gated on the registry slot envs because they
require a configured kms.registry entry:

  - CYBORGDB_KMS_NAME_REAL — real-provider entry with ``provider: aws-kms``
    (HSM-resident KEK; service asks the HSM to wrap the per-index KEK).
  - CYBORGDB_KMS_NAME_SM   — real-provider entry with ``provider: aws``
    (Secrets Manager-resident KEK; service AES-GCM-wraps locally under
    the SM-fetched key).

The SDK-supplied path needs no registry slot and is exercised live
whenever ``CYBORGDB_API_KEY`` is set; it used to be gated on a
``provider: none`` slot that has since been removed from the registry —
strict mutex made that slot unreachable from the SDK anyway.
"""

import json
import os
import urllib.request
import uuid
import unittest

import numpy as np
from dotenv import load_dotenv

import cyborgdb


load_dotenv(".env.local")


BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
KMS_NAME_REAL = os.getenv("CYBORGDB_KMS_NAME_REAL")
KMS_NAME_SM = os.getenv("CYBORGDB_KMS_NAME_SM")

DIMENSION = 128
NUM_VECTORS = 10


def _make_vectors(n: int = NUM_VECTORS, d: int = DIMENSION):
    rng = np.random.default_rng(seed=1234)
    vectors = rng.random((n, d)).astype(np.float32)
    items = [
        {"id": str(i), "vector": vectors[i].tolist(), "metadata": {"idx": i}}
        for i in range(n)
    ]
    return items, vectors


class _KMSRoundTripBase:
    """Lifecycle + shared data-plane assertions for KMS round-trips.

    Concrete subclasses set ``kms_name`` to the registry slot under test
    and set ``needs_sdk_key`` True if the slot is ``provider: none`` (the
    SDK supplies the KEK on every call). They also define ``test_01`` and
    ``test_02`` for the create / load variants their slot type expects.
    """

    kms_name: str = ""
    needs_sdk_key: bool = False

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index_name = f"test_kms_{cls.kms_name}_{uuid.uuid4().hex[:8]}"
        cls.index_key = cyborgdb.Client.generate_key() if cls.needs_sdk_key else None
        cls.index = None

    @classmethod
    def tearDownClass(cls):
        if cls.index is not None:
            try:
                cls.index.delete_index()
            except Exception:
                pass

    def test_03_upsert_and_query(self):
        """Data-plane round-trip on the configured slot."""
        self.assertIsNotNone(self.index, "index not created in test_01")
        items, vectors = _make_vectors()

        self.index.upsert(items)

        results = self.index.query(
            query_vectors=vectors[0].tolist(),
            top_k=3,
            include=["distance", "metadata"],
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "0")  # closest match to itself
        self.assertIn("distance", results[0])
        self.assertIn("metadata", results[0])

    def test_04_other_data_plane_methods(self):
        """The remaining data-plane methods reach the service for this slot
        type. Real-KMS variants exercise these *without* an SDK-held key,
        which is the unique regression risk of the new keyless path."""
        self.assertIsNotNone(self.index, "index not created in test_01")

        all_ids = self.index.list_ids()
        self.assertGreaterEqual(len(all_ids), NUM_VECTORS)

        fetched = self.index.get(ids=["0"], include=["metadata"])
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0]["id"], "0")

        self.assertIsInstance(self.index.is_trained(), bool)
        self.assertIsInstance(self.index.is_training(), bool)

        self.index.delete(ids=["0"])
        remaining = self.index.list_ids()
        self.assertNotIn("0", remaining)


class _RealKMSRoundTrip(_KMSRoundTripBase):
    """Real-KMS slot: service generates and wraps the DEK; SDK holds no key."""

    needs_sdk_key = False

    def test_01_create_index_kms_only(self):
        index = self.client.create_index(
            index_name=self.index_name,
            kms_name=self.kms_name,
            dimension=DIMENSION,
            metric="euclidean",
        )
        self.assertIsInstance(index, cyborgdb.EncryptedIndex)
        self.assertEqual(index.index_name, self.index_name)
        self.assertIsNone(index._index_key)
        type(self).index = index

    def test_02_load_index_without_key(self):
        loaded = self.client.load_index(self.index_name)
        self.assertIsInstance(loaded, cyborgdb.EncryptedIndex)
        self.assertIsNone(loaded._index_key)


@unittest.skipUnless(
    KMS_NAME_REAL,
    "CYBORGDB_KMS_NAME_REAL not set — skipping aws-kms HSM round-trip.",
)
class TestKMSReal(_RealKMSRoundTrip, unittest.TestCase):
    """aws-kms (HSM): KEK lives in the HSM; service asks KMS to wrap the DEK."""

    kms_name = KMS_NAME_REAL or ""


@unittest.skipUnless(
    KMS_NAME_SM,
    "CYBORGDB_KMS_NAME_SM not set — skipping aws Secrets Manager round-trip.",
)
class TestKMSSecretsManager(_RealKMSRoundTrip, unittest.TestCase):
    """aws (Secrets Manager): KEK fetched from SM; service does AES-GCM wrap."""

    kms_name = KMS_NAME_SM or ""


@unittest.skipUnless(
    API_KEY,
    "CYBORGDB_API_KEY not set — skipping SDK-supplied KEK round-trip.",
)
class TestSDKSuppliedKEK(_KMSRoundTripBase, unittest.TestCase):
    """SDK-supplied KEK path: ``index_key`` alone, no ``kms_name``.

    The persisted envelope is ``provider="none"``; the SDK re-supplies
    the same key on every request. No KMS registry slot is referenced
    on the wire — strict mutex made the ``kms_name`` + ``index_key``
    combo a 400 regardless of slot type.
    """

    # ``kms_name`` is intentionally unset — exercising the no-slot wire
    # encoding is the whole point of this suite.
    kms_name = ""
    needs_sdk_key = True

    def test_01_create_index_with_sdk_key(self):
        index = self.client.create_index(
            index_name=self.index_name,
            index_key=self.index_key,
            dimension=DIMENSION,
            metric="euclidean",
        )
        self.assertIsInstance(index, cyborgdb.EncryptedIndex)
        self.assertEqual(index.index_name, self.index_name)
        self.assertEqual(index._index_key, self.index_key)
        type(self).index = index

    def test_02_load_index_with_key(self):
        loaded = self.client.load_index(self.index_name, self.index_key)
        self.assertIsInstance(loaded, cyborgdb.EncryptedIndex)
        self.assertEqual(loaded._index_key, self.index_key)


@unittest.skipUnless(
    KMS_NAME_REAL,
    "CYBORGDB_KMS_NAME_REAL not set — skipping real-provider negative test.",
)
class TestKMSRealRejectsSDKKey(unittest.TestCase):
    """A real-provider slot generates the KEK itself, so supplying
    ``index_key`` alongside ``kms_name`` is contradictory. The service
    rejects it with a 400, which the SDK surfaces as a ``ValueError``
    whose message includes the server's ``detail`` text. The SDK
    forwards both fields untouched — the rejection is the server's
    call, not the client's."""

    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index_name = f"test_kms_neg_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        # Best-effort: if the service unexpectedly created the index, clean up.
        try:
            self.client.load_index(self.index_name).delete_index()
        except Exception:
            pass

    def test_create_index_with_real_kms_and_key_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "index_key must not be supplied alongside"
        ):
            self.client.create_index(
                index_name=self.index_name,
                index_key=cyborgdb.Client.generate_key(),
                kms_name=KMS_NAME_REAL,
                dimension=DIMENSION,
                metric="euclidean",
            )


@unittest.skipUnless(
    API_KEY,
    "CYBORGDB_API_KEY not set — skipping strict-mutex coverage.",
)
class TestStrictMutexFiresBeforeSlotLookup(unittest.TestCase):
    """The ``kms_name`` + ``index_key`` mutex check runs before the
    registry lookup, so an unknown slot combined with an ``index_key``
    returns the *mutex* 400 (not an "unknown slot" 400). Pins down
    "mutex first, slot resolution second" so a future server refactor
    can't silently swap the ordering and let the combination through
    for an as-yet-unknown slot.

    Hits the endpoint directly via ``urllib`` — bypassing the SDK
    helper so we can inspect the server's ``detail`` field, which the
    generated client wraps inside a longer message. Direct probe keeps
    the assertion precise.
    """

    def test_unknown_slot_plus_index_key_returns_mutex_400(self):
        # 32-byte KEK as hex — same shape the SDK would put on the wire.
        index_key_hex = cyborgdb.Client.generate_key().hex()
        payload = json.dumps(
            {
                "index_name": f"test_kms_mutex_{uuid.uuid4().hex[:8]}",
                "index_key": index_key_hex,
                "kms_name": "definitely-not-a-registered-slot",
                "dimension": DIMENSION,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{BASE_URL}/v1/indexes/create",
            data=payload,
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                self.fail(f"expected 400, got {resp.status}")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            body = json.loads(exc.read().decode("utf-8"))
            self.assertIn("detail", body)
            self.assertIn("index_key must not be supplied alongside", body["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
