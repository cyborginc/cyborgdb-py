"""Large-batch behaviour, binary/JSON path parity, and training boundaries.

These cost more wall-clock than the rest of the suite and are aimed at the
overnight run rather than per-PR CI. They exercise what small fixtures cannot:
the binary encoder, the auto-train threshold, and the accuracy cost of
quantised storage.
"""

import os
import time
import unittest
import uuid

import numpy as np
from dotenv import load_dotenv

import cyborgdb
from helpers import wait_for_ids

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
DIM = 64
SCALE_N = 2000

# Deterministic corpus: every assertion below must be reproducible across runs,
# so the vectors are generated from a fixed seed rather than fresh randomness.
_RNG = np.random.default_rng(20260918)
SCALE_VECTORS = _RNG.random((SCALE_N, DIM), dtype=np.float32)
SCALE_IDS = [f"v{i:05d}" for i in range(SCALE_N)]


def _client():
    return cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)


def _brute_force_nearest(query, vectors, k):
    """Exhaustive ground truth by squared euclidean distance."""
    d = ((vectors - query) ** 2).sum(axis=1)
    return [SCALE_IDS[i] for i in np.argsort(d, kind="stable")[:k]]


class TestBinaryPathParity(unittest.TestCase):
    """The binary and JSON encoders must be interchangeable.

    Two encodings of the same data with no differential test between them is
    where silent divergence lives. `upsert_binary`/`query_binary` were only
    exercised as "call it, it does not raise".
    """

    @classmethod
    def setUpClass(cls):
        cls.client = _client()
        cls.key = cyborgdb.Client.generate_key()
        cls.vectors = _RNG.random((200, DIM), dtype=np.float32)
        cls.ids = [f"b{i:03d}" for i in range(200)]

        cls.json_index = cls.client.create_index(
            f"parity_json_{uuid.uuid4().hex[:8]}",
            cls.key,
            dimension=DIM,
            metric="euclidean",
        )
        cls.json_index.upsert(
            [
                {"id": i, "vector": v.tolist(), "metadata": {"n": int(n)}}
                for n, (i, v) in enumerate(zip(cls.ids, cls.vectors))
            ]
        )

        cls.binary_index = cls.client.create_index(
            f"parity_bin_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        # upsert_binary explicitly: passing a list of dicts to upsert() routes
        # to the JSON encoder regardless of the vector type, so the earlier form
        # exercised binary on the query side only.
        cls.binary_index.upsert_binary(
            cls.ids,
            cls.vectors,
            metadata=[{"n": n} for n in range(len(cls.ids))],
        )

        wait_for_ids(cls.json_index, cls.ids)
        wait_for_ids(cls.binary_index, cls.ids)

    @classmethod
    def tearDownClass(cls):
        for index in (
            getattr(cls, "json_index", None),
            getattr(cls, "binary_index", None),
        ):
            try:
                index.delete_index()
            except Exception:
                pass

    def test_both_encoders_rank_identically(self):
        query = self.vectors[7]
        json_ids = [
            r["id"]
            for r in self.json_index.query(query_vectors=query.tolist(), top_k=20)
        ]
        binary_ids = [
            r["id"] for r in self.binary_index.query(query_vectors=query, top_k=20)
        ]
        # Anchored as well as compared: identical encoders that were both wrong
        # would agree with each other.
        self.assertEqual(json_ids, binary_ids)
        self.assertEqual(json_ids[0], self.ids[7], "a vector must be its own nearest")

    def test_both_encoders_return_the_same_result_shape(self):
        # The two paths build their result dicts differently, so compare keys
        # and not just ids. `include` is exercised across its supported values;
        # `vector`/`contents` are absent on query() either way — see
        # cyborgdb-core#2404.
        for include in ([], ["distance"], ["metadata"], ["distance", "metadata"]):
            with self.subTest(include=include):
                json_rows = self.json_index.query(
                    query_vectors=self.vectors[1].tolist(), top_k=3, include=include
                )
                binary_rows = self.binary_index.query(
                    query_vectors=self.vectors[1], top_k=3, include=include
                )
                self.assertEqual(
                    sorted(json_rows[0].keys()), sorted(binary_rows[0].keys())
                )

    def test_both_encoders_round_trip_vectors_identically(self):
        got_json = self.json_index.get([self.ids[3]], include=["vector"])[0]["vector"]
        got_binary = self.binary_index.get([self.ids[3]], include=["vector"])[0][
            "vector"
        ]
        np.testing.assert_allclose(got_json, got_binary, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got_json, self.vectors[3], rtol=1e-5, atol=1e-5)

    def test_metadata_filters_apply_identically_on_both(self):
        f = {"n": {"$lt": 50}}
        json_ids = {
            r["id"]
            for r in self.json_index.query(
                query_vectors=self.vectors[0].tolist(), top_k=200, filters=f
            )
        }
        binary_ids = {
            r["id"]
            for r in self.binary_index.query(
                query_vectors=self.vectors[0], top_k=200, filters=f
            )
        }
        self.assertEqual(json_ids, binary_ids)
        self.assertEqual(json_ids, set(self.ids[:50]))


class TestIncludeProjection(unittest.TestCase):
    """What `include` accepts and what it silently discards.

    Only the unknown-value case is asserted as a bug. Whether `query()` should
    return `vector`/`contents` the way `get()` does is an open question —
    cyborgdb-core#2404 asks for a decision rather than asserting one, so there
    is no test here pretending the answer is known.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = _client()
        cls.index = cls.client.create_index(
            f"include_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        cls.vector = _RNG.random(DIM, dtype=np.float32)
        cls.index.upsert(
            [
                {
                    "id": "only",
                    "vector": cls.vector,
                    "metadata": {"n": 1},
                    "contents": "hello",
                }
            ]
        )
        wait_for_ids(cls.index, ["only"])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_supported_include_values_are_honoured(self):
        rows = self.index.query(
            query_vectors=self.vector, top_k=1, include=["distance"]
        )
        self.assertIn("distance", rows[0])
        rows = self.index.query(
            query_vectors=self.vector, top_k=1, include=["metadata"]
        )
        self.assertEqual(rows[0]["metadata"], {"n": 1})

    def test_get_honours_vector_and_contents(self):
        # The asymmetry in cyborgdb-core#2404: these work on get() and are
        # discarded on query(). Asserted here only for get(), where the
        # contract is documented.
        row = self.index.get(["only"], include=["vector", "contents"])[0]
        self.assertIn("vector", row)
        self.assertEqual(row["contents"], "hello")

    def test_unknown_include_values_are_rejected(self):
        # KNOWN BUG — fails today. cyborgdb-core#2404: an unrecognised value is
        # silently discarded on both methods, so a typo such as "metdata" costs
        # the caller the field with no error. Unlike the vector/contents
        # question, this needs no documentation to be wrong.
        with self.assertRaises(ValueError):
            self.index.query(query_vectors=self.vector, top_k=1, include=["bogus"])
        with self.assertRaises(ValueError):
            self.index.get(["only"], include=["bogus"])


class TestLargeBatch(unittest.TestCase):
    """2000 vectors — well above the rest of the suite's 100, still exhaustive.

    Not enough to train: AUTO_TRAIN_MIN_VECTORS is 65536, so search here is
    still exact. That makes the assertions below verifiable against brute-force
    ground truth; the approximate path is covered by the trained fixture in
    test_trained_index.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = _client()
        cls.index = cls.client.create_index(
            f"scale_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        cls.index.upsert(
            [
                {"id": i, "vector": v, "metadata": {"bucket": int(n) % 10}}
                for n, (i, v) in enumerate(zip(SCALE_IDS, SCALE_VECTORS))
            ]
        )
        wait_for_ids(cls.index, SCALE_IDS[:1] + SCALE_IDS[-1:])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_every_vector_is_retrievable(self):
        ids = set(self.index.list_ids())
        missing = set(SCALE_IDS) - ids
        self.assertEqual(missing, set(), f"{len(missing)} of {SCALE_N} vectors missing")

    def test_exact_search_matches_brute_force(self):
        query = SCALE_VECTORS[123]
        got = [r["id"] for r in self.index.query(query_vectors=query, top_k=10)]
        self.assertEqual(got, _brute_force_nearest(query, SCALE_VECTORS, 10))

    def test_filtered_search_at_scale(self):
        got = {
            r["id"]
            for r in self.index.query(
                query_vectors=SCALE_VECTORS[0], top_k=SCALE_N, filters={"bucket": 3}
            )
        }
        self.assertEqual(got, {i for n, i in enumerate(SCALE_IDS) if n % 10 == 3})

    def test_top_k_prefix_invariant(self):
        # The first k of a larger result must equal the smaller result. Only
        # meaningful once the corpus exceeds top_k * window_mult, which the
        # small fixtures elsewhere never do.
        query = SCALE_VECTORS[500]
        wide = [r["id"] for r in self.index.query(query_vectors=query, top_k=50)]
        narrow = [r["id"] for r in self.index.query(query_vectors=query, top_k=10)]
        self.assertEqual(wide[:10], narrow)


class TestTrainingBoundaries(unittest.TestCase):
    """`train()` at and below the documented minimum."""

    def setUp(self):
        self.client = _client()
        self.index = self.client.create_index(
            f"trainmin_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def _seed(self, n):
        vectors = _RNG.random((n, DIM), dtype=np.float32)
        ids = [f"t{i:05d}" for i in range(n)]
        self.index.upsert([{"id": i, "vector": v} for i, v in zip(ids, vectors)])
        wait_for_ids(self.index, ids[:1])
        return ids

    def test_training_below_the_minimum_is_a_silent_no_op(self):
        # AUTO_TRAIN_MIN_VECTORS is 65536, so train() silently does nothing for
        # any index below it — returns successfully, leaves the index untrained,
        # and is_trained() is the only signal. Ticket item 7 assumed it errors.
        ids = self._seed(5)
        self.index.train(n_lists=64)
        self.assertFalse(self.index.is_trained())
        got = self.index.query(
            query_vectors=_RNG.random(DIM, dtype=np.float32), top_k=3
        )
        self.assertTrue({r["id"] for r in got} <= set(ids))

    def test_more_lists_than_vectors_is_a_silent_no_op(self):
        # Degenerate case, same contract: no error, no training, correct results.
        ids = self._seed(2)
        self.index.train(n_lists=2)
        self.assertFalse(self.index.is_trained())
        top = self.index.query(
            query_vectors=_RNG.random(DIM, dtype=np.float32), top_k=2
        )
        self.assertEqual({r["id"] for r in top}, set(ids))

    def test_untrained_index_still_queries(self):
        # Training is an optimisation, not a prerequisite: an untrained index
        # answers exactly via exhaustive search.
        ids = self._seed(20)
        got = self.index.query(
            query_vectors=_RNG.random(DIM, dtype=np.float32), top_k=5
        )
        self.assertEqual(len(got), 5)
        self.assertTrue({r["id"] for r in got} <= set(ids))
        self.assertFalse(self.index.is_trained())


class TestStoragePrecisionAccuracy(unittest.TestCase):
    """Quantised storage trades accuracy for size — bound the trade.

    test_storage_precision.py covers validation, serialisation and lifecycle
    across every tier, but nothing asserts that a quantised index still returns
    sensible results. These bound the loss instead of assuming it.
    """

    N = 500

    @classmethod
    def setUpClass(cls):
        cls.client = _client()
        cls.vectors = _RNG.random((cls.N, DIM), dtype=np.float32)
        cls.ids = [f"p{i:04d}" for i in range(cls.N)]
        cls.query = cls.vectors[42]
        cls.truth = [
            cls.ids[i]
            for i in np.argsort(
                ((cls.vectors - cls.query) ** 2).sum(axis=1), kind="stable"
            )[:10]
        ]
        cls.indexes = {}
        for precision in ("float32", "float16", "tq8"):
            index = cls.client.create_index(
                f"prec_{precision}_{uuid.uuid4().hex[:8]}",
                cyborgdb.Client.generate_key(),
                dimension=DIM,
                metric="euclidean",
                storage_precision=precision,
            )
            index.upsert([{"id": i, "vector": v} for i, v in zip(cls.ids, cls.vectors)])
            cls.indexes[precision] = index
        for index in cls.indexes.values():
            wait_for_ids(index, cls.ids[:1])
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        for index in getattr(cls, "indexes", {}).values():
            try:
                index.delete_index()
            except Exception:
                pass

    def _recall_at_10(self, precision):
        got = [
            r["id"]
            for r in self.indexes[precision].query(query_vectors=self.query, top_k=10)
        ]
        return len(set(got) & set(self.truth)) / len(self.truth)

    def test_float32_is_exact(self):
        # No quantisation, exhaustive search: the result must equal ground truth
        # outright, not merely approximate it.
        got = [
            r["id"]
            for r in self.indexes["float32"].query(query_vectors=self.query, top_k=10)
        ]
        self.assertEqual(got, self.truth)

    def test_quantised_tiers_stay_usable(self):
        # Deliberately loose floors. The purpose is to catch a tier that has
        # become badly wrong, not to police small accuracy movements — a tight
        # threshold here would be a flaky test rather than a useful one.
        for precision, floor in (("float16", 0.9), ("tq8", 0.5)):
            with self.subTest(precision=precision):
                recall = self._recall_at_10(precision)
                self.assertGreaterEqual(
                    recall, floor, f"{precision} recall@10 was {recall:.2f}"
                )

    def test_a_vector_finds_itself_at_every_precision(self):
        # The weakest possible accuracy guarantee, and the one that must hold
        # even at the most aggressive quantisation.
        for precision, index in self.indexes.items():
            with self.subTest(precision=precision):
                top = index.query(query_vectors=self.vectors[7], top_k=1)
                self.assertEqual(top[0]["id"], self.ids[7])


if __name__ == "__main__":
    unittest.main()
