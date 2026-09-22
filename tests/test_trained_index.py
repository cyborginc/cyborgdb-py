"""The approximate search path.

Everything else in the suite runs on untrained indexes, where search is
exhaustive and exact. The service only trains past `AUTO_TRAIN_MIN_VECTORS`
(65536 by default), so reaching the approximate path at all needs a corpus that
size — which is why nothing except quick_flow_test.py has ever tested it.

The corpus is `load_sample_dataset()` (quickstart-75k): 75,000 vectors, 100
queries, and ground-truth neighbours for both the trained and untrained cases.
Building the index takes a couple of minutes, so this file is aimed at the
overnight run.

Covers `rerank_mult`, which cannot be tested anywhere else: it widens the
candidate set before a final exact re-scoring pass, so on an exhaustive index
it is a no-op by construction.
"""

import os
import time
import unittest
import uuid

import numpy as np
from dotenv import load_dotenv

import cyborgdb

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
UPSERT_BATCH = 5000
TRAIN_TIMEOUT = 600


class TrainedIndexTestCase(unittest.TestCase):
    """Shared 75k trained index. Built once; every test here reads it."""

    @classmethod
    def setUpClass(cls):
        cls.data = cyborgdb.load_sample_dataset()
        cls.ids = list(cls.data.ids)
        cls.vectors = np.asarray(cls.data.vectors, dtype=np.float32)
        cls.queries = np.asarray(cls.data.queries, dtype=np.float32)
        cls.truth = np.asarray(cls.data.trained_neighbors)

        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        # `fruits` derives from the dataset's `list` field: ten terms, each in
        # ~35% of documents. Marking `string` full_text instead would make it
        # non-filterable and break the example-filter test below.
        cls.index = cls.client.create_index(
            f"trained_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=cls.vectors.shape[1],
            metric=str(cls.data.metric),
            text_fields=["fruits"],
        )

        total = len(cls.ids)
        for start in range(0, total, UPSERT_BATCH):
            stop = min(start + UPSERT_BATCH, total)
            cls.index.upsert(
                [
                    {
                        "id": cls.ids[i],
                        "vector": cls.vectors[i],
                        "metadata": {
                            **cls.data.metadata[i],
                            "fruits": " ".join(cls.data.metadata[i].get("list", [])),
                        },
                    }
                    for i in range(start, stop)
                ]
            )

        # Crossing AUTO_TRAIN_MIN_VECTORS queues training on a background
        # worker, so the index is not trained the moment the upsert returns.
        deadline = time.monotonic() + TRAIN_TIMEOUT
        while time.monotonic() < deadline:
            if cls.index.is_trained():
                break
            time.sleep(5)
        else:
            raise AssertionError(
                f"index did not train within {TRAIN_TIMEOUT}s of upserting {total} vectors"
            )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _recall_at_k(self, k, **query_kwargs):
        """Mean recall@k across every query in the dataset."""
        results = self.index.query(query_vectors=self.queries, top_k=k, **query_kwargs)
        scores = []
        for q, rows in enumerate(results):
            expected = {self.ids[i] for i in self.truth[q][:k]}
            got = {r["id"] for r in rows}
            scores.append(len(got & expected) / k)
        return float(np.mean(scores))


class TestTrainedIndex(TrainedIndexTestCase):
    """One class so the 75k index is built once rather than per class."""

    def test_index_is_trained(self):
        self.assertTrue(self.index.is_trained())
        self.assertGreater(self.index.n_lists, 0)

    def test_recall_meets_the_datasets_stated_expectation(self):
        # The dataset ships the recall its authors measured for the trained
        # case. Compared against that rather than a number invented here, with
        # headroom so ordinary index-build variation does not trip it.
        expected = float(self.data.trained_recall)
        recall = self._recall_at_k(100)
        # Measured 0.947 against the dataset's stated 0.940, so 2% of headroom
        # is enough; the earlier 10% was slack under a number that clears.
        self.assertGreaterEqual(
            recall,
            expected * 0.98,
            f"recall@100 was {recall:.3f}, dataset expects ~{expected:.3f}",
        )

    def test_every_vector_is_still_retrievable_after_training(self):
        # Training rebuilds the index; nothing may be lost in the process.
        sample = self.ids[::5000]
        got = self.index.get(sample, include=["vector"])
        self.assertEqual({r["id"] for r in got}, set(sample))

    # -- rerank_mult: only assertable where search is approximate ---------- #

    def test_wider_reranking_does_not_reduce_recall(self):
        # The real contract: a wider candidate set cannot make results worse.
        # Averaged over all 100 queries — on any single query the two can
        # legitimately tie, so a per-query strict inequality would be flaky.
        narrow = self._recall_at_k(10, rerank_mult=1)
        wide = self._recall_at_k(10, rerank_mult=8)
        # Measured sweep of recall@10: 0.783 / 0.944 / 0.971 / 0.977 / 0.977 for
        # rerank_mult 1 / 2 / 4 / 8 / 16. The effect is large and saturates
        # around 8, so assert a real improvement rather than merely "not worse"
        # — the latter would pass if rerank_mult were ignored entirely.
        self.assertGreaterEqual(
            wide,
            narrow + 0.05,
            f"recall@10 only moved {narrow:.3f} -> {wide:.3f} between rerank_mult 1 and 8",
        )
        self.assertGreater(narrow, 0.7, f"baseline recall@10 was only {narrow:.3f}")

    def test_rerank_mult_does_not_change_the_result_count(self):
        for mult in (1, 4, 16):
            with self.subTest(rerank_mult=mult):
                rows = self.index.query(
                    query_vectors=self.queries[0], top_k=10, rerank_mult=mult
                )
                self.assertEqual(len(rows), 10)

    def test_results_stay_ordered_by_distance(self):
        for mult in (1, 8):
            with self.subTest(rerank_mult=mult):
                rows = self.index.query(
                    query_vectors=self.queries[0],
                    top_k=20,
                    rerank_mult=mult,
                    include=["distance"],
                )
                distances = [r["distance"] for r in rows]
                self.assertEqual(distances, sorted(distances))

    def test_top_k_times_rerank_mult_ceiling_is_enforced(self):
        # Ticket item 9: nothing anywhere asserted the 10000 ceiling, or that
        # the error names the parameter responsible.
        with self.assertRaises(ValueError) as caught:
            self.index.query(query_vectors=self.queries[0], top_k=5000, rerank_mult=4)
        message = str(caught.exception)
        self.assertIn("10000", message)
        # KNOWN BUG — this assertion fails today. cyborgdb-core#2401: the
        # message says "top_k exceeds kMaxTopK" even though top_k=5000 is
        # itself under the limit; it is the product with rerank_mult that
        # breaches it. A caller reducing top_k to 2500 still fails.
        self.assertIn(
            "rerank_mult",
            message,
            f"the error should name the parameter responsible, got: {message}",
        )

    def test_the_ceiling_is_inclusive(self):
        # Exactly 10000 is accepted; only above it is rejected. Without this the
        # test above would still pass if the limit were off by one.
        for top_k, rerank_mult in ((2000, 5), (1000, 10), (100, 100)):
            with self.subTest(top_k=top_k, rerank_mult=rerank_mult):
                rows = self.index.query(
                    query_vectors=self.queries[0], top_k=top_k, rerank_mult=rerank_mult
                )
                self.assertTrue(rows)

    # -- metadata filtering against the approximate path ------------------- #

    def test_example_filters_from_the_dataset_all_resolve(self):
        # The dataset ships filters its authors consider representative. Each
        # must return something and every row must satisfy the filter.
        for example in self.data.example_filters:
            with self.subTest(example["name"]):
                rows = self.index.query(
                    query_vectors=self.queries[0],
                    top_k=50,
                    filters=example["filter"],
                    include=["metadata"],
                )
                self.assertTrue(rows, f"{example['name']} matched nothing")
                for row in rows:
                    self.assertTrue(
                        self._matches(row["metadata"], example["filter"]),
                        f"{row['id']} does not satisfy {example['filter']}",
                    )

    # -- hybrid on the approximate path (ticket item 10) ------------------- #
    #
    # Not relevance tests: every term sits in ~35% of documents, so the ranking
    # is mostly ties. These assert the wiring only, and are differential, so the
    # weak text does not matter.

    HYBRID_TEXT = "grape cherry"

    def test_hybrid_returns_fused_scores_on_a_trained_index(self):
        rows = self.index.query(
            query_vectors=self.queries[0], text=self.HYBRID_TEXT, top_k=10
        )
        self.assertTrue(rows)
        self.assertTrue(all("score" in r for r in rows))
        self.assertFalse(any("distance" in r for r in rows))

    def test_alpha_one_reproduces_the_approximate_vector_ranking(self):
        # The new ground covered here: at alpha=1 the fused result must match
        # the plain vector query, which on a trained index is the *approximate*
        # ranking. Nothing else checks that fusion leaves it intact.
        vector_only = [
            r["id"] for r in self.index.query(query_vectors=self.queries[0], top_k=10)
        ]
        fused = [
            r["id"]
            for r in self.index.query(
                query_vectors=self.queries[0],
                text=self.HYBRID_TEXT,
                alpha=1.0,
                top_k=10,
            )
        ]
        self.assertEqual(fused, vector_only)

    def test_alpha_zero_reproduces_the_pure_bm25_ranking(self):
        text_only = [
            r["id"] for r in self.index.query_metadata(text=self.HYBRID_TEXT, top_k=10)
        ]
        fused = [
            r["id"]
            for r in self.index.query(
                query_vectors=self.queries[0],
                text=self.HYBRID_TEXT,
                alpha=0.0,
                top_k=10,
            )
        ]
        self.assertEqual(fused, text_only)

    def test_alpha_endpoints_disagree_on_a_trained_index(self):
        # Guards the two above: if the vector and text rankings coincided they
        # would both pass while proving nothing.
        vector_only = [
            r["id"] for r in self.index.query(query_vectors=self.queries[0], top_k=10)
        ]
        text_only = [
            r["id"] for r in self.index.query_metadata(text=self.HYBRID_TEXT, top_k=10)
        ]
        self.assertNotEqual(vector_only, text_only)

    def test_hybrid_filter_prefilters_both_legs_when_trained(self):
        rows = self.index.query(
            query_vectors=self.queries[0],
            text=self.HYBRID_TEXT,
            filters={"number": {"$lt": 100}},
            top_k=20,
            include=["metadata"],
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertLess(row["metadata"]["number"], 100)

    @staticmethod
    def _matches(metadata, filters):
        """Evaluate the dataset's example filters locally, as an oracle."""
        for field, condition in filters.items():
            value = metadata.get(field)
            if not isinstance(condition, dict):
                if isinstance(value, list):
                    if condition not in value:
                        return False
                elif value != condition:
                    return False
                continue
            for op, operand in condition.items():
                if op == "$lt" and not value < operand:
                    return False
                if op == "$lte" and not value <= operand:
                    return False
                if op == "$gte" and not value >= operand:
                    return False
                if op == "$in":
                    candidates = value if isinstance(value, list) else [value]
                    if not set(candidates) & set(operand):
                        return False
        return True


if __name__ == "__main__":
    unittest.main()
