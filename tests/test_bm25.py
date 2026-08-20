"""BM25 full-text search: the `full_text` metadata policy, the `bm25` scorer
config, and the `text=...` legs on `query_metadata` (pure BM25) and `query`
(hybrid BM25 + vector).

Mirrors the create-time knobs (`text_fields`, `bm25_k1`, `bm25_b`) and the
two read paths that a full-text field unlocks. BM25 is opt-in and derived: an
index with at least one `full_text` field reports a `bm25` config and accepts
the `text=...` legs; an index with none reports `bm25 is None` and rejects them
server-side. Full-text search resolves from the metadata index and needs no
training, so these run on small untrained indexes.
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
DIM = 8

# `body` is analyzed by BM25; `topic` stays an exact-match filterable field so
# we can pre-filter the text leg. Docs 0/2/4 are about quantum computing to
# differing degrees; 1/3/5 are unrelated noise.
DOCS = [
    ("d0", "quantum computing breakthroughs in error correction", "physics"),
    ("d1", "classical machine learning models for tabular data", "ml"),
    ("d2", "quantum entanglement and superposition explained", "physics"),
    ("d3", "cooking pasta with fresh tomatoes and basil", "food"),
    ("d4", "advances in quantum computing hardware and qubits", "physics"),
    ("d5", "financial markets and stock trading strategies", "finance"),
]
# "quantum computing" — both terms in d0/d4, only "quantum" in d2.
BOTH_TERMS = {"d0", "d4"}
ANY_TERM = {"d0", "d2", "d4"}


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"bm25_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema={"topic": {"filterable": True}},
            text_fields=["body"],
            bm25_k1=1.5,
            bm25_b=0.7,
        )
        self.index.upsert(
            [
                {
                    "id": doc_id,
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"body": body, "topic": topic},
                }
                for doc_id, body, topic in DOCS
            ]
        )
        time.sleep(2)

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    # -- schema / config round-trip -------------------------------------- #

    def test_full_text_reported_in_schema(self):
        self.assertEqual(
            self.index.metadata_schema["body"],
            {"filterable": False, "pattern": False, "full_text": True},
        )

    def test_bm25_config_reports_tuning_params(self):
        config = self.index.bm25
        self.assertIsNotNone(config)
        self.assertAlmostEqual(config["k1"], 1.5)
        self.assertAlmostEqual(config["b"], 0.7)
        self.assertIn("analyzer_version", config)

    # -- query_metadata(text=...) : pure BM25 ---------------------------- #

    def test_text_search_returns_scored_dicts_ranked(self):
        results = self.index.query_metadata(text="quantum computing")
        self.assertTrue(results, "expected at least one match")
        # Scored dicts, not bare IDs, and sorted by descending score.
        self.assertTrue(all(set(r) == {"id", "score"} for r in results))
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # Every hit is a quantum doc; the top hit contains both query terms.
        self.assertTrue({r["id"] for r in results} <= ANY_TERM)
        self.assertIn(results[0]["id"], BOTH_TERMS)

    def test_require_all_terms_narrows_to_and(self):
        got = {
            r["id"]
            for r in self.index.query_metadata(
                text="quantum computing", require_all_terms=True
            )
        }
        self.assertEqual(got, BOTH_TERMS)

    def test_text_search_top_k_caps_results(self):
        results = self.index.query_metadata(text="quantum", top_k=1)
        self.assertEqual(len(results), 1)

    def test_text_fields_restricts_to_named_field(self):
        # `body` is the only full_text field; naming it explicitly is a no-op
        # but must be accepted.
        results = self.index.query_metadata(text="quantum", text_fields=["body"])
        self.assertTrue({r["id"] for r in results} <= ANY_TERM)

    def test_filter_prefilters_the_text_leg(self):
        # topic=food excludes every quantum doc, so the text leg scores nothing.
        results = self.index.query_metadata(text="quantum", filters={"topic": "food"})
        self.assertEqual(results, [])

    def test_no_text_still_returns_bare_ids(self):
        # Without text this stays a filter-only query: bare ID strings.
        ids = self.index.query_metadata(filters={"topic": "physics"})
        self.assertEqual(set(ids), {"d0", "d2", "d4"})
        self.assertTrue(all(isinstance(i, str) for i in ids))

    # -- query(text=...) : hybrid BM25 + vector -------------------------- #

    def test_hybrid_query_list_vector_carries_score(self):
        results = self.index.query(
            query_vectors=np.random.rand(DIM).astype(np.float32).tolist(),
            text="quantum computing",
            top_k=6,
        )
        self.assertTrue(results)
        # Hybrid rows are scored (fused), not distance-ranked.
        self.assertTrue(all("score" in r for r in results))
        self.assertFalse(any("distance" in r for r in results))

    def test_hybrid_query_numpy_vector_carries_score(self):
        # Numpy input routes through the binary path; it must forward the text
        # leg too.
        results = self.index.query(
            query_vectors=np.random.rand(DIM).astype(np.float32),
            text="quantum computing",
            top_k=6,
            alpha=0.5,
        )
        self.assertTrue(results)
        self.assertTrue(all("score" in r for r in results))

    def test_pure_vector_query_still_uses_distance(self):
        # `include` defaults to [] (IDs only); distance must be requested.
        results = self.index.query(
            query_vectors=np.random.rand(DIM).astype(np.float32).tolist(),
            top_k=6,
            include=["distance"],
        )
        self.assertTrue(results)
        self.assertTrue(all("distance" in r for r in results))
        self.assertFalse(any("score" in r for r in results))


class TestBM25NotConfigured(unittest.TestCase):
    """An index with no full_text field: BM25 is absent, not empty."""

    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"bm25_none_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        self.index.upsert(
            [
                {
                    "id": f"i{i}",
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"body": "quantum computing"},
                }
                for i in range(4)
            ]
        )
        time.sleep(2)

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def test_bm25_is_none(self):
        self.assertIsNone(self.index.bm25)

    def test_text_query_rejected_without_full_text_field(self):
        with self.assertRaises(ValueError):
            self.index.query_metadata(text="quantum")


if __name__ == "__main__":
    unittest.main()
