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

    def test_filter_operator_prefilters_the_text_leg(self):
        # An operator filter ($in) must pre-filter the text leg the same way an
        # equality filter does: only physics docs survive, so only quantum docs
        # can score — the food/ml/finance rows never reach the BM25 leg.
        results = self.index.query_metadata(
            text="quantum", filters={"topic": {"$in": ["physics"]}}
        )
        self.assertEqual({r["id"] for r in results}, ANY_TERM)
        self.assertTrue(all(set(r) == {"id", "score"} for r in results))

    def test_require_all_terms_with_filter_composes(self):
        # AND-matching and the pre-filter apply together: require_all_terms
        # narrows to {d0, d4}, and topic=physics keeps both (they are physics).
        got = {
            r["id"]
            for r in self.index.query_metadata(
                text="quantum computing",
                require_all_terms=True,
                filters={"topic": "physics"},
            )
        }
        self.assertEqual(got, BOTH_TERMS)

    def test_empty_text_is_filter_only(self):
        # Documented contract: an empty `text` keeps this a filter-only query —
        # {"id"} rows with no `score` — even though the SDK still forwards the
        # empty string to the service. Pins that "" is treated as "no text leg".
        rows = self.index.query_metadata(text="", filters={"topic": "physics"})
        self.assertEqual({r["id"] for r in rows}, {"d0", "d2", "d4"})
        self.assertTrue(all(r == {"id": r["id"]} for r in rows))

    def test_text_matching_no_document_returns_empty(self):
        # A term that appears in no `body` scores nothing: empty result, no error.
        self.assertEqual(self.index.query_metadata(text="zzzznonexistent"), [])

    def test_top_k_larger_than_matches_returns_all(self):
        # top_k above the match count is a cap, not a floor: all 3 quantum docs
        # come back, not padded to top_k.
        results = self.index.query_metadata(text="quantum", top_k=100)
        self.assertEqual({r["id"] for r in results}, ANY_TERM)

    def test_text_search_is_case_insensitive(self):
        # The BM25 analyzer lower-cases terms, so an upper-case query matches the
        # same docs as its lower-case form.
        upper = {r["id"] for r in self.index.query_metadata(text="QUANTUM COMPUTING")}
        lower = {r["id"] for r in self.index.query_metadata(text="quantum computing")}
        self.assertEqual(upper, lower)
        self.assertEqual(lower, ANY_TERM)

    def test_order_by_with_text_is_rejected(self):
        # Text results are relevance-ranked, so `order_by` alongside `text` is
        # unsupported and must raise rather than silently ignore one of them.
        with self.assertRaises(ValueError):
            self.index.query_metadata(text="quantum", order_by="topic")

    def test_non_filterable_field_rejected_even_with_text(self):
        # The metadata schema is enforced on the text path too: a pre-filter on
        # a non-filterable field raises, exactly as it does without `text`
        # (there is no post-filter fallback in query_metadata).
        with self.assertRaises(ValueError):
            self.index.query_metadata(text="quantum", filters={"body": "quantum"})

    # -- query(text=..., filters=...) : hybrid + pre-filter --------------- #

    def test_hybrid_query_applies_metadata_filter(self):
        # The metadata filter must pre-filter the hybrid candidate set: with
        # topic=food, no quantum doc survives and the text leg contributes
        # nothing, so only food docs (if any) can appear — never a quantum doc.
        results = self.index.query(
            query_vectors=np.random.rand(DIM).astype(np.float32).tolist(),
            text="quantum computing",
            filters={"topic": "food"},
            top_k=6,
        )
        self.assertTrue({r["id"] for r in results} <= {"d3"})
        self.assertTrue(all("distance" not in r for r in results))

    def test_no_text_returns_unscored_id_rows(self):
        # Without text this stays a filter-only query: {"id"} rows (no score),
        # matching core's list[MetadataResult].
        rows = self.index.query_metadata(filters={"topic": "physics"})
        self.assertEqual({r["id"] for r in rows}, {"d0", "d2", "d4"})
        self.assertTrue(all(r == {"id": r["id"]} for r in rows))

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

    def test_hybrid_scores_descending(self):
        # Fused (BM25 + vector) rows are ranked: scores come back
        # non-increasing. Vector inputs are random so the *ordering of ids*
        # isn't deterministic, but the score column must still be sorted.
        results = self.index.query(
            query_vectors=np.random.rand(DIM).astype(np.float32).tolist(),
            text="quantum computing",
            top_k=6,
        )
        self.assertTrue(results)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hybrid_alpha_forwarded_to_service(self):
        # `alpha` must reach the service: an out-of-[0, 1] value is rejected
        # there, proving the SDK forwards it rather than dropping it.
        with self.assertRaises(ValueError):
            self.index.query(
                query_vectors=np.random.rand(DIM).astype(np.float32).tolist(),
                text="quantum computing",
                alpha=5.0,
            )

    def test_hybrid_text_fields_forwarded_to_service(self):
        # `text_fields` must reach the service: naming a non-full-text field
        # (`topic`) is rejected there, proving forwarding on the hybrid path.
        with self.assertRaises(ValueError):
            self.index.query(
                query_vectors=np.random.rand(DIM).astype(np.float32).tolist(),
                text="quantum",
                text_fields=["topic"],
            )

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


class TestBM25MetadataFilterNarrowing(unittest.TestCase):
    """Two full_text fields (`title`, `body`) plus a discriminating filterable
    field (`lang`), so a single text term matches several docs and a metadata
    filter can narrow the hits to a *proper subset* — the case the single-topic
    fixture above can't express. Also lets `text_fields` genuinely exclude a hit
    (a term present only in the un-searched field)."""

    # "quantum" appears in different fields per doc; `lang` splits the matches.
    ROWS = [
        ("a", "quantum theory", "notes on physics", "en"),  # title
        ("b", "kitchen recipes", "a quantum leap forward", "en"),  # body only
        ("c", "quantum hardware", "qubit fabrication", "fr"),  # title
        ("d", "sourdough bread", "baking at home", "en"),  # no match
    ]
    QUANTUM_ANY_FIELD = {"a", "b", "c"}
    QUANTUM_IN_TITLE = {"a", "c"}

    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"bm25_filter_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema={"lang": {"filterable": True}},
            text_fields=["title", "body"],
        )
        self.index.upsert(
            [
                {
                    "id": doc_id,
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"title": title, "body": body, "lang": lang},
                }
                for doc_id, title, body, lang in self.ROWS
            ]
        )
        time.sleep(2)

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def test_text_matches_across_both_fields(self):
        # With no field restriction the term is found in either full_text field.
        got = {r["id"] for r in self.index.query_metadata(text="quantum")}
        self.assertEqual(got, self.QUANTUM_ANY_FIELD)

    def test_filter_narrows_text_matches_to_proper_subset(self):
        # text matches {a, b, c}; lang=en drops the French doc `c`, leaving a
        # strict subset — proving the pre-filter intersects rather than replaces.
        got = {
            r["id"]
            for r in self.index.query_metadata(text="quantum", filters={"lang": "en"})
        }
        self.assertEqual(got, {"a", "b"})
        self.assertTrue(got < self.QUANTUM_ANY_FIELD)

    def test_text_fields_excludes_match_in_unsearched_field(self):
        # Restricting to `title` drops `b`, whose only "quantum" is in `body`.
        got = {
            r["id"]
            for r in self.index.query_metadata(text="quantum", text_fields=["title"])
        }
        self.assertEqual(got, self.QUANTUM_IN_TITLE)

    def test_text_fields_and_filter_compose(self):
        # Both narrowings apply together: title-only → {a, c}, then lang=en drops
        # the French `c`, leaving just {a}.
        got = {
            r["id"]
            for r in self.index.query_metadata(
                text="quantum", text_fields=["title"], filters={"lang": "en"}
            )
        }
        self.assertEqual(got, {"a"})

    def test_field_weights_accepted_and_rank_stable(self):
        # Per-field weights (parallel to the searched fields) are forwarded and
        # accepted; the matched set is unchanged by re-weighting.
        got = {
            r["id"]
            for r in self.index.query_metadata(
                text="quantum",
                text_fields=["title", "body"],
                text_field_weights=[2.0, 1.0],
            )
        }
        self.assertEqual(got, self.QUANTUM_ANY_FIELD)


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


class TestMetadataResultContract(unittest.TestCase):
    """`query_metadata` returns plain-dict `MetadataResult` rows (matching core),
    so the public row type is a hand-written TypedDict rather than the generated
    wire model. These pin the TypedDict to the OpenAPI contract without importing
    core/service, so drift in core's shape (re-generated into the wire model)
    fails here instead of silently diverging. No service needed."""

    def test_metadata_result_is_public(self):
        # Exported at the top level so callers can annotate query_metadata rows.
        from cyborgdb import MetadataResult

        self.assertIs(MetadataResult, cyborgdb.MetadataResult)
        self.assertIn("MetadataResult", cyborgdb.__all__)

    def test_typed_dict_shape_is_id_required_score_optional(self):
        from cyborgdb import MetadataResult

        self.assertEqual(set(MetadataResult.__required_keys__), {"id"})
        self.assertEqual(set(MetadataResult.__optional_keys__), {"score"})
        self.assertEqual(MetadataResult.__annotations__["id"], str)
        self.assertEqual(MetadataResult.__annotations__["score"], float)

    def test_typed_dict_matches_wire_contract(self):
        # The wire model is generated from openapi.json (sourced from core), so
        # if core adds/renames a field the regenerated model changes and this
        # fails — flagging that the hand-written TypedDict needs the same update.
        from cyborgdb import MetadataResult
        from cyborgdb.openapi_client.models import MetadataResult as WireMetadataResult

        typed_keys = set(MetadataResult.__required_keys__) | set(
            MetadataResult.__optional_keys__
        )
        self.assertEqual(typed_keys, set(WireMetadataResult.model_fields))


if __name__ == "__main__":
    unittest.main()
