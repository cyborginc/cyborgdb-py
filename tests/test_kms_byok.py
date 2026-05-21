"""
KMS BYOK integration tests for the CyborgDB Python SDK.

These tests are gated on environment variables that name entries in the
running cyborgdb-service's `kms.registry`. Set the variable to opt the
corresponding registry slot in; leave it unset to skip.

- CYBORGDB_KMS_NAME_REAL — real-provider entry with `provider: aws-kms`
  (HSM-resident KEK; service generates the DEK and asks the HSM to wrap it).
- CYBORGDB_KMS_NAME_SM   — real-provider entry with `provider: aws`
  (Secrets Manager-resident KEK; service generates the DEK and AES-GCM-
  wraps it locally under the SM-fetched key).
- CYBORGDB_KMS_NAME_NONE — entry with `provider: none`. The SDK supplies
  the KEK on every request; service does no KMS round-trips.

All three exercise the SDK round-trip introduced when create_index and
load_index moved to optional index_key + kms_name routing. The two real-
KMS variants behave identically from the SDK's perspective and share a
common test body.
"""

import os
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
KMS_NAME_NONE = os.getenv("CYBORGDB_KMS_NAME_NONE")

DIMENSION = 128
NUM_VECTORS = 10


def _make_client() -> cyborgdb.Client:
    return cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)


def _make_vectors(n: int = NUM_VECTORS, d: int = DIMENSION):
    rng = np.random.default_rng(seed=1234)
    vectors = rng.random((n, d)).astype(np.float32)
    items = [
        {"id": str(i), "vector": vectors[i].tolist(), "metadata": {"idx": i}}
        for i in range(n)
    ]
    return items, vectors


class _RealKMSRoundTripMixin:
    """Shared body for real-KMS round-trips (aws-kms HSM, aws Secrets
    Manager). Subclasses set ``kms_name`` to the registry slot under test."""

    # Subclasses override:
    kms_name: str = ""

    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()
        cls.index_name = f"test_kms_{cls.kms_name}_{uuid.uuid4().hex[:8]}"
        cls.index = None

    @classmethod
    def tearDownClass(cls):
        if cls.index is not None:
            try:
                cls.index.delete_index()
            except Exception:
                pass

    def test_01_create_index_kms_only(self):
        """create_index with kms_name and no index_key succeeds; the returned
        EncryptedIndex carries no SDK-side key."""
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
        """load_index(name) without an index_key resolves the KEK via the
        index's KMSBlob and returns a usable handle."""
        loaded = self.client.load_index(self.index_name)
        self.assertIsInstance(loaded, cyborgdb.EncryptedIndex)
        self.assertIsNone(loaded._index_key)
        # Sanity: index_type is fetched via the keyless describe path.
        self.assertEqual(loaded.index_type, "disk_ivf")

    def test_03_upsert_and_query(self):
        """Data-plane operations work on a KMS-backed index without an SDK-
        side key (service resolves the DEK from the KMSBlob)."""
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


@unittest.skipUnless(
    KMS_NAME_REAL,
    "CYBORGDB_KMS_NAME_REAL not set — skipping aws-kms HSM round-trip.",
)
class TestKMSReal(_RealKMSRoundTripMixin, unittest.TestCase):
    """aws-kms (HSM): KEK lives in the HSM; service asks KMS to wrap the DEK."""

    kms_name = KMS_NAME_REAL or ""


@unittest.skipUnless(
    KMS_NAME_SM,
    "CYBORGDB_KMS_NAME_SM not set — skipping aws Secrets Manager round-trip.",
)
class TestKMSSecretsManager(_RealKMSRoundTripMixin, unittest.TestCase):
    """aws (Secrets Manager): KEK fetched from SM; service does AES-GCM wrap."""

    kms_name = KMS_NAME_SM or ""


@unittest.skipUnless(
    KMS_NAME_NONE,
    "CYBORGDB_KMS_NAME_NONE not set — skipping provider:none round-trip.",
)
class TestProviderNone(unittest.TestCase):
    """provider:none slot: SDK supplies the KEK; service does no KMS calls.
    Validates the mixed mode where both index_key and kms_name are passed."""

    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()
        cls.index_name = f"test_kms_none_{uuid.uuid4().hex[:8]}"
        cls.index_key = cyborgdb.Client.generate_key()
        cls.index = None

    @classmethod
    def tearDownClass(cls):
        if cls.index is not None:
            try:
                cls.index.delete_index()
            except Exception:
                pass

    def test_01_create_index_key_plus_kms_name(self):
        """create_index with both index_key and a provider:none kms_name
        succeeds; the SDK retains its key for subsequent calls."""
        index = self.client.create_index(
            index_name=self.index_name,
            index_key=self.index_key,
            kms_name=KMS_NAME_NONE,
            dimension=DIMENSION,
            metric="euclidean",
        )
        self.assertIsInstance(index, cyborgdb.EncryptedIndex)
        self.assertEqual(index.index_name, self.index_name)
        self.assertEqual(index._index_key, self.index_key)
        type(self).index = index

    def test_02_load_index_with_key(self):
        """load_index with the same key returns a usable handle.
        (provider:none indexes require the SDK key on every call.)"""
        loaded = self.client.load_index(self.index_name, self.index_key)
        self.assertIsInstance(loaded, cyborgdb.EncryptedIndex)
        self.assertEqual(loaded._index_key, self.index_key)
        self.assertEqual(loaded.index_type, "disk_ivf")

    def test_03_upsert_and_query(self):
        """Data-plane operations work end-to-end on a provider:none index."""
        self.assertIsNotNone(self.index, "index not created in test_01")
        items, vectors = _make_vectors()

        self.index.upsert(items)

        results = self.index.query(
            query_vectors=vectors[0].tolist(),
            top_k=3,
            include=["distance", "metadata"],
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "0")
        self.assertIn("distance", results[0])
        self.assertIn("metadata", results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
