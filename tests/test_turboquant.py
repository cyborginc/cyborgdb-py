"""TurboQuant storage precision: the `storage_precision` create-time knob and
its three new quantized tiers `tq8` / `tq6` / `tq4`.

`storage_precision` picks the on-disk rerank-vector format, chosen at create
and immutable. Alongside the existing `float32` / `float16`, the TurboQuant
tiers pack 8 / 6 / 4 bits per dimension, trading a little recall and latency
for a large storage saving. `tq4` is only valid with the cosine metric.

Two layers of coverage:

* Model-level (no service) — `CreateIndexRequest` accepts every valid tier,
  rejects anything else, and serializes the value through to the wire dict.
  These are the direct, deterministic checks that the tiers were wired in.
* End-to-end (live service on localhost:8000) — each tier survives the full
  create -> upsert -> train -> query round-trip and returns sane, high
  self-recall results. Skipped automatically when no service is reachable.

The index-info response does not echo `storage_precision` back, so the
end-to-end layer verifies the tiers by behavior, not by reading the value
back off the index.
"""

import os
import time
import unittest
import uuid

import numpy as np
import urllib3
from dotenv import load_dotenv
from pydantic import ValidationError

import cyborgdb
from cyborgdb.openapi_client.models import CreateIndexRequest

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")

VALID_PRECISIONS = ["float32", "float16", "tq8", "tq6", "tq4"]
TURBOQUANT_TIERS = ["tq8", "tq6", "tq4"]

# Enough vectors to clear the core training floor (train() silently no-ops
# below 10k vectors) while staying quick.
NUM_VECTORS = 10000
DIM = 64
N_LISTS = 8


def _service_up() -> bool:
    """True when a CyborgDB service answers /v1/health at BASE_URL."""
    try:
        resp = urllib3.PoolManager().request(
            "GET", f"{BASE_URL}/v1/health", timeout=2.0, retries=False
        )
        return resp.status == 200
    except Exception:
        return False


SERVICE_UP = _service_up()


class TurboQuantModelTest(unittest.TestCase):
    """Model-level contract for `storage_precision` — no service required."""

    def test_all_valid_precisions_accepted(self):
        for precision in VALID_PRECISIONS:
            with self.subTest(precision=precision):
                request = CreateIndexRequest(
                    index_name="idx", storage_precision=precision
                )
                self.assertEqual(request.storage_precision, precision)

    def test_turboquant_tiers_accepted(self):
        # The three tiers this change adds, called out explicitly.
        for tier in TURBOQUANT_TIERS:
            with self.subTest(tier=tier):
                request = CreateIndexRequest(index_name="idx", storage_precision=tier)
                self.assertEqual(request.storage_precision, tier)

    def test_storage_precision_optional(self):
        request = CreateIndexRequest(index_name="idx")
        self.assertIsNone(request.storage_precision)

    def test_invalid_precision_rejected(self):
        for bad in ["tq2", "tq16", "int8", "fp16", "float64", "TQ8", ""]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    CreateIndexRequest(index_name="idx", storage_precision=bad)

    def test_error_message_lists_valid_tiers(self):
        with self.assertRaises(ValidationError) as ctx:
            CreateIndexRequest(index_name="idx", storage_precision="tq5")
        message = str(ctx.exception)
        for tier in TURBOQUANT_TIERS:
            self.assertIn(tier, message)

    def test_precision_serialized_to_wire_dict(self):
        for tier in TURBOQUANT_TIERS:
            with self.subTest(tier=tier):
                payload = CreateIndexRequest(
                    index_name="idx", storage_precision=tier
                ).to_dict()
                self.assertEqual(payload["storage_precision"], tier)

    def test_precision_round_trips_through_from_dict(self):
        for tier in TURBOQUANT_TIERS:
            with self.subTest(tier=tier):
                restored = CreateIndexRequest.from_dict(
                    {"index_name": "idx", "storage_precision": tier}
                )
                self.assertEqual(restored.storage_precision, tier)


@unittest.skipUnless(SERVICE_UP, f"no CyborgDB service reachable at {BASE_URL}")
class TurboQuantIntegrationTest(unittest.TestCase):
    """End-to-end: each TurboQuant tier survives the full index lifecycle.

    One shared, cosine-metric corpus is built once (cosine is required by
    `tq4` and valid for every other tier). Each tier gets its own index so a
    failure names the tier that broke.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        vectors = rng.random((NUM_VECTORS, DIM), dtype=np.float32)
        # Normalize for the cosine metric so self-queries are unambiguous.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        cls.vectors = vectors / np.clip(norms, 1e-12, None)
        cls.ids = [str(i) for i in range(NUM_VECTORS)]
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)

    def _build_trained_index(self, precision):
        """Create a cosine index at `precision`, load it, train it, return it."""
        index = self.client.create_index(
            index_name=f"tq_{precision}_{uuid.uuid4().hex[:8]}",
            index_key=cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="cosine",
            storage_precision=precision,
        )
        self.addCleanup(self._safe_delete, index)

        index.upsert(self.ids, self.vectors)
        time.sleep(1)
        self.assertEqual(len(index.list_ids()), NUM_VECTORS)

        index.train(n_lists=N_LISTS)
        for _ in range(60):
            if not index.is_training() and index.is_trained():
                break
            time.sleep(2)
        self.assertTrue(index.is_trained(), f"{precision} index failed to train")
        return index

    @staticmethod
    def _safe_delete(index):
        try:
            index.delete_index()
        except Exception:
            pass

    def _assert_self_recall(self, index, precision, num_probe=50, min_recall=0.8):
        """Query with vectors that are in the index; each should find itself.

        Exhaustive search (n_probes == n_lists) removes IVF partitioning as a
        variable, so the only recall loss left is TurboQuant's quantization —
        which the threshold tolerates.
        """
        probe_ids = list(range(num_probe))
        results = index.query(
            query_vectors=self.vectors[probe_ids], top_k=10, n_probes=N_LISTS
        )
        self.assertEqual(len(results), num_probe)

        hits = 0
        for local_id, hit_list in zip(probe_ids, results):
            returned = {res["id"] for res in hit_list}
            if str(local_id) in returned:
                hits += 1
        recall = hits / num_probe
        self.assertGreaterEqual(
            recall,
            min_recall,
            f"{precision}: self-recall {recall:.2f} below {min_recall}",
        )

    def test_tq8_lifecycle(self):
        index = self._build_trained_index("tq8")
        self._assert_self_recall(index, "tq8", min_recall=0.9)

    def test_tq6_lifecycle(self):
        index = self._build_trained_index("tq6")
        self._assert_self_recall(index, "tq6", min_recall=0.85)

    def test_tq4_lifecycle(self):
        # tq4 is the most aggressive tier and is only valid with cosine.
        index = self._build_trained_index("tq4")
        self._assert_self_recall(index, "tq4", min_recall=0.7)

    def test_tq4_requires_cosine_metric(self):
        # tq4 with a non-cosine metric must be rejected by the service.
        with self.assertRaises(ValueError):
            index = self.client.create_index(
                index_name=f"tq4_bad_{uuid.uuid4().hex[:8]}",
                index_key=cyborgdb.Client.generate_key(),
                dimension=DIM,
                metric="euclidean",
                storage_precision="tq4",
            )
            self.addCleanup(self._safe_delete, index)


if __name__ == "__main__":
    unittest.main()
