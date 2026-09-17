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
import unittest
import uuid

import numpy as np
from dotenv import load_dotenv

import cyborgdb
from helpers import DEFAULT_TIMEOUT, wait_for, wait_for_ids, wait_until_gone

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
DIM = 8

# Upserts become visible asynchronously; the suites poll rather than sleep.
# See tests/helpers.py for why a fixed delay was the wrong tool.
PROPAGATION_TIMEOUT = DEFAULT_TIMEOUT
_wait_for_ids = wait_for_ids
_wait_until_gone = wait_until_gone

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
    # Every test in this class is read-only, so the fixture is built once for
    # the class rather than once per test. As per-test setUp this was creating
    # an index, upserting, and waiting ~20 times over for no isolation benefit.
    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"bm25_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema={"topic": {"filterable": True}},
            text_fields=["body"],
            bm25_k1=1.5,
            bm25_b=0.7,
        )
        cls.index.upsert(
            [
                {
                    "id": doc_id,
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"body": body, "topic": topic},
                }
                for doc_id, body, topic in DOCS
            ]
        )
        _wait_for_ids(cls.index, [doc_id for doc_id, _, _ in DOCS])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
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

    # Read-only class: one fixture for all of it.
    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"bm25_filter_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema={"lang": {"filterable": True}},
            text_fields=["title", "body"],
        )
        cls.index.upsert(
            [
                {
                    "id": doc_id,
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"title": title, "body": body, "lang": lang},
                }
                for doc_id, title, body, lang in cls.ROWS
            ]
        )
        _wait_for_ids(cls.index, [row[0] for row in cls.ROWS])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
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

    def test_field_weights_flip_the_top_result(self):
        # Per-field weights change the *ranking* — that is the only thing they
        # do — so the assertion has to be on order, not on the matched set.
        # `a` and `c` match in `title` only and `b` matches in `body` only, so
        # weighting one field heavily must lift its documents above the other's.
        #
        # 10:1 against 1:10 is a 100x swing, deliberately far wider than any
        # term-frequency or field-length difference in this fixture, so the flip
        # does not depend on the exact per-field BM25 formula. A 2:1 weighting
        # would ride on those details and go flaky.
        title_heavy = [
            r["id"]
            for r in self.index.query_metadata(
                text="quantum",
                text_fields=["title", "body"],
                text_field_weights=[10.0, 1.0],
            )
        ]
        body_heavy = [
            r["id"]
            for r in self.index.query_metadata(
                text="quantum",
                text_fields=["title", "body"],
                text_field_weights=[1.0, 10.0],
            )
        ]
        # Re-weighting reorders; it never filters. Both directions still match
        # every document that contains the term in either field.
        self.assertEqual(set(title_heavy), self.QUANTUM_ANY_FIELD)
        self.assertEqual(set(body_heavy), self.QUANTUM_ANY_FIELD)
        # ...but the winner changes: a title-only match leads when `title` is
        # weighted, and the body-only match `b` leads when `body` is. If the
        # service ignored the weights, both lists would be identical and this
        # would fail — which the previous set-equality assertion could not.
        self.assertIn(title_heavy[0], self.QUANTUM_IN_TITLE)
        self.assertEqual(body_heavy[0], "b")
        self.assertNotEqual(title_heavy[0], body_heavy[0])


class TestBM25NotConfigured(unittest.TestCase):
    """An index with no full_text field: BM25 is absent, not empty."""

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"bm25_none_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        cls.index.upsert(
            [
                {
                    "id": f"i{i}",
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"body": "quantum computing"},
                }
                for i in range(4)
            ]
        )
        _wait_for_ids(cls.index, [f"i{i}" for i in range(4)])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
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


HYBRID_DIM = 4

# Ported from cyborgdb-core tests/bm25_api_test.py. Every other hybrid test in
# this file seeds random vectors, which makes the vector leg noise and leaves
# `alpha`, `rrf_k` and the fused ranking unassertable. Here the document vectors
# are basis vectors and the query sits at a fixed point, so squared-euclidean
# distances are strictly ordered —
#
#     d2 (0.0125) < d1 (1.8125) < d0 (1.9125) < d3 (2.0125)
#
# — and nothing below rests on a distance tie-break. `metric="euclidean"` and
# `dimension=4` are load-bearing: change either and the ordering above stops
# holding, taking the expected results with it.
HYBRID_DOCS = [
    ("d0", [1.0, 0.0, 0.0, 0.0], "apple banana", "date date elder", "ann"),
    ("d1", [0.0, 1.0, 0.0, 0.0], "banana", "date", "bob"),
    ("d2", [0.0, 0.0, 1.0, 0.0], "cherry", "elder", "ann"),
    ("d3", [0.0, 0.0, 0.0, 1.0], "apple", "fig", "bob"),
]
HYBRID_QUERY_VECTOR = [0.05, 0.1, 1.0, 0.0]
HYBRID_TEXT = "apple date"


class TestHybridFusionDeterministic(unittest.TestCase):
    """Hybrid fusion with hand-chosen vectors, so the fused ranking is a fact
    rather than noise.

    Core proves the fusion maths (tests/bm25_api_test.py). What these prove is
    the wiring: that each knob crosses the wire intact, reaches core, and that
    the ordering survives JSON serialisation and the SDK's result mapping. A
    disagreement between one of these and its core twin is a transport bug.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"hybrid_fusion_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            # `full_text` set directly rather than via the `text_fields` sugar,
            # which the shortcut-based fixtures above never exercise.
            #
            # `filterable: False` is spelled out only because the SDK cannot
            # currently send `{"full_text": True}` on its own — see
            # test_full_text_alone_is_rejected_by_the_sdk below.
            metadata_schema={
                "title": {"full_text": True, "filterable": False},
                "body": {"full_text": True, "filterable": False},
                "author": {"filterable": True},
            },
        )
        cls.index.upsert(
            [
                {
                    "id": doc_id,
                    "vector": vector,
                    "metadata": {"title": title, "body": body, "author": author},
                }
                for doc_id, vector, title, body, author in HYBRID_DOCS
            ]
        )
        _wait_for_ids(cls.index, [doc[0] for doc in HYBRID_DOCS])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _hybrid(self, **kwargs):
        kwargs.setdefault("text", HYBRID_TEXT)
        kwargs.setdefault("top_k", 4)
        return self.index.query(query_vectors=HYBRID_QUERY_VECTOR, **kwargs)

    def _hybrid_ids(self, **kwargs):
        return [r["id"] for r in self._hybrid(**kwargs)]

    def _text_only_ids(self):
        return [r["id"] for r in self.index.query_metadata(text=HYBRID_TEXT, top_k=4)]

    def _vector_only_ids(self):
        return [
            r["id"]
            for r in self.index.query(query_vectors=HYBRID_QUERY_VECTOR, top_k=4)
        ]

    # -- alpha: tested against each leg rather than against arithmetic ---- #

    def test_alpha_zero_reproduces_the_pure_bm25_ranking(self):
        # alpha=0 is pure BM25 by definition, so the fused order must equal the
        # text-only order exactly. Comparing one call against another means no
        # score is ever computed here, and the random-vector problem disappears:
        # at alpha=0 the vectors genuinely cannot matter.
        self.assertEqual(self._hybrid_ids(alpha=0.0), self._text_only_ids())

    def test_alpha_one_reproduces_the_pure_vector_ranking(self):
        # The mirror image: alpha=1 drops the text leg, so the fused order must
        # equal a plain vector query over the same vector.
        self.assertEqual(self._hybrid_ids(alpha=1.0), self._vector_only_ids())

    def test_alpha_endpoints_disagree(self):
        # Guards the two tests above. If the BM25 and vector rankings happened
        # to coincide on this corpus, both would pass while proving nothing.
        # d2 is the vector winner and has no text match at all, so the two
        # orderings genuinely differ.
        self.assertNotEqual(self._text_only_ids(), self._vector_only_ids())

    # -- fusion ------------------------------------------------------------ #

    def test_fusion_promotes_a_document_neither_leg_ranked_first(self):
        # Vector ranking (by distance to HYBRID_QUERY_VECTOR): d2, d1, d0, d3.
        # Text ranking for "apple date":                       d0, then d1/d3.
        #
        # At the defaults (alpha 0.5, rrf_k 60) d0 wins on agreement across both
        # legs — 0.5/61 + 0.5/63 — ahead of d1 at 0.5/62 + 0.5/62, while d2,
        # rank 1 in the vector leg but absent from the text leg, falls to last
        # on 0.5/61 alone. Both orderings of the d1/d3 text tie fuse the same
        # way, so this does not depend on how that tie breaks.
        #
        # This is the assertion random vectors make impossible: it is only
        # meaningful because the distances above are fixed.
        self.assertEqual(self._hybrid_ids(), ["d0", "d1", "d3", "d2"])

    def test_rrf_k_reaches_the_fusion(self):
        # RRF contributes 1/(k + rank) per leg, so shrinking k raises every
        # fused score. Asserting on scores rather than order keeps this robust:
        # on a four-document corpus the order changes only by knife-edge
        # margins (<1%), which is exactly how a flaky test gets written.
        small = {r["id"]: r["score"] for r in self._hybrid(rrf_k=1.0)}
        large = {r["id"]: r["score"] for r in self._hybrid(rrf_k=60.0)}
        self.assertEqual(set(small), set(large), "rrf_k must not change matches")
        # If rrf_k were dropped in transit, these would be identical.
        self.assertNotEqual(small, large)
        for doc_id in small:
            self.assertGreater(
                small[doc_id],
                large[doc_id],
                f"{doc_id}: smaller rrf_k must raise the fused score",
            )

    def test_window_mult_bounds_and_monotonicity(self):
        # window_mult is per-leg candidate depth as a multiple of top_k. On a
        # four-document corpus every document is always a candidate, so no
        # honest assertion about it changing the *ranking* is available here —
        # claiming otherwise would be a test that passes for the wrong reason.
        # What can be asserted: the bound is enforced, and widening the window
        # never returns fewer results.
        with self.assertRaises(ValueError):
            self._hybrid(window_mult=0)
        narrow = self._hybrid_ids(top_k=2, window_mult=1)
        wide = self._hybrid_ids(top_k=2, window_mult=4)
        self.assertEqual(len(narrow), len(wide))
        self.assertLessEqual(len(narrow), 2)

    # -- projections and batching ------------------------------------------ #

    def test_include_metadata_returns_the_fused_winners_metadata(self):
        # `include` has only ever been tested on get(); this is the query()
        # path, and specifically the hybrid query() path.
        results = self._hybrid(include=["metadata"], top_k=2)
        self.assertEqual(len(results), 2)
        for row in results:
            self.assertIn("metadata", row)
            self.assertIn("title", row["metadata"])
        # d0 wins the fusion (see above) and d0's author is "ann".
        self.assertEqual(results[0]["id"], "d0")
        self.assertEqual(results[0]["metadata"]["author"], "ann")

    def test_batch_hybrid_fuses_each_row_independently(self):
        # Batch + hybrid exists only in Go today. Two identical query vectors
        # must produce two identical fused rankings, each matching the
        # single-vector result — proving the text leg is applied per row rather
        # than once for the whole batch.
        expected = self._hybrid_ids()
        batched = self.index.query(
            query_vectors=np.array(
                [HYBRID_QUERY_VECTOR, HYBRID_QUERY_VECTOR], dtype=np.float32
            ),
            text=HYBRID_TEXT,
            top_k=4,
        )
        self.assertEqual(len(batched), 2)
        for row_results in batched:
            self.assertEqual([r["id"] for r in row_results], expected)

    def test_filter_prefilters_both_legs_of_the_hybrid(self):
        # author=ann keeps d0 and d2 only. The surviving order must be the
        # fused order restricted to those two, not an arbitrary subset.
        ids = self._hybrid_ids(filters={"author": "ann"})
        self.assertEqual(ids, ["d0", "d2"])

    # -- determinism -------------------------------------------------------- #

    def test_repeated_queries_are_identical(self):
        # Cheap, and the precondition for every order-based assertion above:
        # if ties broke arbitrarily between calls, those would flake instead of
        # failing honestly.
        first = self._hybrid()
        second = self._hybrid()
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertEqual([r["score"] for r in first], [r["score"] for r in second])


class TestMetadataFieldPolicyDefaults(unittest.TestCase):
    """The `full_text` shorthand the SDK documents but cannot currently send."""

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)

    def _create(self, metadata_schema):
        index = self.client.create_index(
            f"policy_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            metadata_schema=metadata_schema,
        )
        self.addCleanup(lambda: self._safe_delete(index))
        return index

    @staticmethod
    def _safe_delete(index):
        try:
            index.delete_index()
        except Exception:
            pass

    @unittest.expectedFailure
    def test_full_text_alone_is_rejected_by_the_sdk(self):
        # `create_index`'s own docstring states that `full_text=True` "implies
        # filterable=False", and core accepts `{"full_text": True}` on its own.
        # Through this SDK it cannot work: the generated MetadataFieldPolicy
        # model declares `filterable: Optional[StrictBool] = True` and always
        # serialises it, so the request carries
        #     {"filterable": true, "pattern": false, "full_text": true}
        # and the service rejects the combination with a 422.
        #
        # Marked expectedFailure rather than deleted so the contract stays
        # written down: when the default is fixed this test passes, unittest
        # reports an unexpected success, and the marker gets removed. Fixing it
        # is out of scope here (the ticket's non-goals put bug fixes in a
        # separate change).
        self._create({"title": {"full_text": True}})

    def test_full_text_works_when_filterable_is_spelled_out(self):
        # The workaround callers currently need.
        index = self._create({"title": {"full_text": True, "filterable": False}})
        self.assertEqual(
            index.metadata_schema["title"],
            {"filterable": False, "pattern": False, "full_text": True},
        )

    def test_text_fields_sugar_is_equivalent(self):
        # The documented shortcut produces the same policy, and does work.
        index = self.client.create_index(
            f"policy_sugar_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            text_fields=["title"],
        )
        self.addCleanup(lambda: self._safe_delete(index))
        self.assertEqual(
            index.metadata_schema["title"],
            {"filterable": False, "pattern": False, "full_text": True},
        )


class TestBM25Analyzer(unittest.TestCase):
    """Observable behaviour of the tokenizer/stemmer pipeline.

    The pipeline is not configurable from the SDK — `index.bm25` only reports an
    `analyzer_version` — which is exactly why its behaviour should be pinned:
    that version can change underneath us, and nothing else would notice. Every
    expectation below was measured against the running service rather than
    assumed, so a failure here means the analyzer changed, not that the test
    guessed wrong.
    """

    ROWS = {
        "stem": "running runner runs",
        "punct": "mind-killer, fear! (really)",
        "accent": "café résumé naïve",
        "stop": "the a an and or but of",
        "num": "version 42 build 7",
        "case": "MixedCase WORD",
        "plural": "boxes churches",
    }

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"bm25_analyzer_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            text_fields=["body"],
        )
        cls.index.upsert(
            [
                {
                    "id": doc_id,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "metadata": {"body": body},
                }
                for doc_id, body in cls.ROWS.items()
            ]
        )
        _wait_for_ids(cls.index, list(cls.ROWS))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _ids(self, text):
        return {r["id"] for r in self.index.query_metadata(text=text)}

    def test_terms_are_stemmed(self):
        # "running runner runs" is reachable from each of its inflections, so
        # the analyzer stems rather than matching raw tokens.
        for term in ("run", "runs", "runner", "running"):
            with self.subTest(term=term):
                self.assertIn("stem", self._ids(term))

    def test_plurals_stem_to_their_singular(self):
        self.assertIn("plural", self._ids("box"))
        self.assertIn("plural", self._ids("church"))

    def test_punctuation_is_stripped_and_hyphens_split(self):
        # "mind-killer" indexes as two terms, and trailing punctuation on
        # "fear!" does not become part of the token.
        self.assertIn("punct", self._ids("mind"))
        self.assertIn("punct", self._ids("killer"))
        self.assertIn("punct", self._ids("fear"))

    def test_case_is_folded_both_ways(self):
        self.assertIn("case", self._ids("mixedcase"))
        self.assertIn("case", self._ids("WORD"))

    def test_stop_words_are_dropped(self):
        # A document made entirely of stop words contributes no searchable
        # terms, so querying one matches nothing at all.
        for term in ("the", "and", "of"):
            with self.subTest(term=term):
                self.assertEqual(self._ids(term), set())

    def test_numeric_tokens_are_indexed(self):
        self.assertIn("num", self._ids("42"))

    def test_accents_are_not_folded(self):
        # Measured behaviour, and the one that most often surprises callers:
        # "café" matches only its exact accented form. If accent folding is ever
        # added this test fails, which is the point — it is a user-visible
        # search behaviour change that should be a deliberate decision.
        self.assertIn("accent", self._ids("café"))
        self.assertEqual(self._ids("cafe"), set())


# Scoring-property fixtures. Unlike the uniform synthetic rows elsewhere in this
# file, these corpora are shaped so that one BM25 property at a time is the only
# thing separating two documents — the approach Qdrant takes by seeding its BM25
# fixture with stop-word-heavy documents alongside real sentences.
#
#   IDF      "zeppelin" occurs in one document, "common" in five. Both candidate
#            documents are the same length and match exactly one query term, so
#            only the term's rarity can separate them.
#   LENGTH   Same term, same term-frequency, very different document lengths.
#   TF       Same length, different term-frequency.
SCORING_DOCS = [
    ("idf_rare", "zeppelin padding padding padding"),
    ("idf_common", "common padding padding padding"),
    ("c1", "common padding padding padding"),
    ("c2", "common padding padding padding"),
    ("c3", "common padding padding padding"),
    ("c4", "common padding padding padding"),
    ("len_short", "target"),
    ("len_long", "target " + "filler " * 24),
    ("tf_one", "saturate alpha beta gamma delta"),
    ("tf_many", "saturate saturate saturate saturate saturate"),
]


def _scoring_items():
    return [
        {
            "id": doc_id,
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"body": body},
        }
        for doc_id, body in SCORING_DOCS
    ]


class TestBM25ScoringProperties(unittest.TestCase):
    """BM25's three defining behaviours, asserted through the SDK.

    Core proves the arithmetic (bm25_score_test.cpp). These assert the
    consequences survive the wire: a ranking that depends on IDF, on document
    length, and on term frequency. Only relative order is asserted — never an
    absolute score — because scores shift legitimately with `analyzer_version`.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"bm25_scoring_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            text_fields=["body"],
        )
        cls.index.upsert(_scoring_items())
        _wait_for_ids(cls.index, [doc_id for doc_id, _ in SCORING_DOCS])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _ranked(self, text):
        return [r["id"] for r in self.index.query_metadata(text=text)]

    def _scores(self, text):
        return {r["id"]: r["score"] for r in self.index.query_metadata(text=text)}

    def test_a_rare_term_outranks_a_common_one(self):
        # Both documents match exactly one query term, with the same term
        # frequency and the same length. The only difference is that "zeppelin"
        # appears in one document and "common" in five, so IDF alone decides.
        ranked = self._ranked("zeppelin common")
        self.assertIn("idf_rare", ranked)
        self.assertIn("idf_common", ranked)
        self.assertLess(
            ranked.index("idf_rare"),
            ranked.index("idf_common"),
            "a term matching 1 of 10 documents must outrank one matching 5",
        )

    def test_a_shorter_document_outranks_a_longer_one(self):
        # Same term, same term-frequency; `len_long` simply buries it in 24
        # filler words. Length normalisation must penalise it.
        ranked = self._ranked("target")
        self.assertEqual(ranked[:2], ["len_short", "len_long"])

    def test_higher_term_frequency_scores_higher(self):
        ranked = self._ranked("saturate")
        self.assertEqual(ranked[:2], ["tf_many", "tf_one"])

    def test_a_term_in_every_document_still_scores(self):
        # "common" is in five of ten documents. IDF shrinks but must not go
        # negative or zero the row out — every matching document still comes
        # back with a score.
        scores = self._scores("common")
        self.assertEqual(set(scores), {"idf_common", "c1", "c2", "c3", "c4"})
        for doc_id, score in scores.items():
            self.assertGreater(score, 0.0, f"{doc_id} scored non-positive")


class TestBM25TuningParameters(unittest.TestCase):
    """`bm25_k1` and `bm25_b` change ranking, not just `describe` output.

    Both were previously tier-2: created with non-default values and asserted
    only via the config round-trip, which would pass if the engine ignored them.
    Each test here builds a second index differing in exactly one parameter and
    asserts the ranking difference that parameter is responsible for.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.indexes = []

    @classmethod
    def tearDownClass(cls):
        for index in cls.indexes:
            try:
                index.delete_index()
            except Exception:
                pass

    @classmethod
    def _seeded(cls, label, **create_kwargs):
        index = cls.client.create_index(
            f"bm25_tune_{label}_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            text_fields=["body"],
            **create_kwargs,
        )
        cls.indexes.append(index)
        index.upsert(_scoring_items())
        _wait_for_ids(index, [doc_id for doc_id, _ in SCORING_DOCS])
        return index

    @staticmethod
    def _ranked(index, text):
        return [r["id"] for r in index.query_metadata(text=text)]

    def test_b_zero_removes_the_length_penalty(self):
        # `b` controls how much document length matters. At the default (0.75)
        # the short document wins; at b=0 length is ignored entirely, so two
        # documents with the same term-frequency must score equally and the
        # ordering between them stops being decided by length.
        default_b = self._seeded("bdefault")
        no_length = self._seeded("bzero", bm25_b=0.0)

        self.assertEqual(
            self._ranked(default_b, "target")[:2], ["len_short", "len_long"]
        )

        scores = {r["id"]: r["score"] for r in no_length.query_metadata(text="target")}
        self.assertEqual(set(scores), {"len_short", "len_long"})
        self.assertAlmostEqual(
            scores["len_short"],
            scores["len_long"],
            places=5,
            msg="with b=0 document length must not affect the score",
        )

    def test_k1_zero_makes_scoring_binary(self):
        # `k1` controls term-frequency saturation. At k1=0 the tf component
        # collapses to presence/absence, so a document containing the term five
        # times scores the same as one containing it once. At the default they
        # differ — asserted alongside so the comparison is meaningful.
        default_k1 = self._seeded("kdefault")
        binary = self._seeded("kzero", bm25_k1=0.0)

        default_scores = {
            r["id"]: r["score"] for r in default_k1.query_metadata(text="saturate")
        }
        self.assertGreater(
            default_scores["tf_many"],
            default_scores["tf_one"],
            "at the default k1, repeating a term must raise the score",
        )

        binary_scores = {
            r["id"]: r["score"] for r in binary.query_metadata(text="saturate")
        }
        self.assertAlmostEqual(
            binary_scores["tf_many"],
            binary_scores["tf_one"],
            places=5,
            msg="with k1=0 term frequency must stop mattering",
        )


class TestBM25Lifecycle(unittest.TestCase):
    """BM25 after mutation — the behaviour no SDK test covers today.

    BM25 scores depend on corpus-wide statistics (document count, total document
    length) that feed IDF and length normalisation. cyborg-encrypted-index tests
    these hard at its own layer (ReUpsertReplacesNotDoubleCounts,
    DeleteSubtractsFromGlobals). Nothing until now checked they are wired
    through core -> service -> SDK, where stale statistics would silently skew
    every subsequent score with no error surface.
    """

    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"bm25_lifecycle_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=HYBRID_DIM,
            metric="euclidean",
            text_fields=["body"],
        )
        self.index.upsert(
            [
                {
                    "id": f"m{i}",
                    "vector": [float(i == j) for j in range(HYBRID_DIM)],
                    "metadata": {"body": body},
                }
                for i, body in enumerate(
                    ["alpha beta", "alpha gamma", "delta epsilon", "alpha zeta"]
                )
            ]
        )
        _wait_for_ids(self.index, ["m0", "m1", "m2", "m3"])

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def _ids(self, text):
        return {r["id"] for r in self.index.query_metadata(text=text)}

    def test_delete_removes_a_document_from_text_results(self):
        self.assertEqual(self._ids("alpha"), {"m0", "m1", "m3"})
        self.index.delete(["m1"])
        _wait_until_gone(self.index, ["m1"])
        self.assertEqual(self._ids("alpha"), {"m0", "m3"})

    def test_deleted_document_never_comes_back(self):
        # Ported from core's BM25QueryTest::DeletedDocumentsNeverComeBack. The
        # delete has to travel the same wire as the query, so the core version
        # cannot substitute for this one.
        self.index.delete(["m0"])
        _wait_until_gone(self.index, ["m0"])
        for text in ("alpha", "alpha beta", "beta"):
            self.assertNotIn("m0", self._ids(text), f"m0 resurfaced for {text!r}")

    def test_updating_a_text_field_moves_the_document_between_results(self):
        # The update path, not just insert: m2 does not match "alpha" until its
        # body is rewritten, and stops matching "delta" once it is.
        self.assertNotIn("m2", self._ids("alpha"))
        self.index.upsert(
            [
                {
                    "id": "m2",
                    "vector": [0.0, 0.0, 1.0, 0.0],
                    "metadata": {"body": "alpha omega"},
                }
            ]
        )
        wait_for(
            lambda: "m2" in self._ids("alpha"),
            "m2 becomes searchable for 'alpha' after its body was rewritten",
        )
        # ...and the old term no longer matches it: the update replaced the
        # document's postings rather than adding to them.
        self.assertNotIn("m2", self._ids("delta"))

    def test_reupsert_does_not_double_count(self):
        # Re-upserting an unchanged document must leave scores untouched. If
        # the corpus statistics were double-counted, IDF and the length
        # normaliser would shift and every score would move.
        before = {r["id"]: r["score"] for r in self.index.query_metadata(text="alpha")}
        # `body` is byte-identical to the original, so BM25 must be unaffected.
        # A `marker` field rides along purely so there is something observable
        # to poll for — otherwise this would need a blind sleep, which is the
        # habit these helpers exist to remove.
        self.index.upsert(
            [
                {
                    "id": "m0",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"body": "alpha beta", "marker": "reupserted"},
                }
            ]
        )
        wait_for(
            lambda: (
                {r["id"] for r in self.index.query_metadata({"marker": "reupserted"})}
                == {"m0"}
            ),
            "re-upsert of m0 becomes visible",
        )
        after = {r["id"]: r["score"] for r in self.index.query_metadata(text="alpha")}
        self.assertEqual(set(before), set(after))
        for doc_id in before:
            self.assertAlmostEqual(
                before[doc_id],
                after[doc_id],
                places=5,
                msg=f"{doc_id}: re-upserting an unchanged document moved its score",
            )


if __name__ == "__main__":
    unittest.main()
