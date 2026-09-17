"""Metadata-only query (`index.query_metadata`) and the per-field indexing
policy it enforces (`create_index(metadata_schema=...)`).

Mirrors go query_metadata_test.go and js query_metadata.test.ts.

The point of these tests is the asymmetry between the two read paths. `query()`
can always fall back to a post-filter over the decrypted metadata, so there the
policy only affects speed. `query_metadata()` resolves everything from the
index with no fallback, so the policy is enforced — `$regex`/`$contains` need a
`pattern` field and a non-filterable field cannot be filtered at all. Each
rejection is paired with the same filter succeeding via `query()`, so a failure
points at the policy rather than at a broken filter.
"""

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
from dotenv import load_dotenv

import cyborgdb
from helpers import wait_for_ids

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
DIM = 8
N = 6

# `color` opts into the regex dictionary, `shape` is indexed but not pattern,
# `hidden` opts out of indexing entirely. Even ids are red/square/secret.
SCHEMA = {
    "color": {"filterable": True, "pattern": True},
    "shape": {"filterable": True, "pattern": False},
    "hidden": {"filterable": False},
}
EVEN = {f"i{i}" for i in range(0, N, 2)}
ODD = {f"i{i}" for i in range(1, N, 2)}


def _ids(rows):
    """Pull the ids out of query_metadata's `{"id"}` rows (core's shape)."""
    return [row["id"] for row in rows]


class TestQueryMetadata(unittest.TestCase):
    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"query_metadata_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema=SCHEMA,
        )
        self.index.upsert(
            [
                {
                    "id": f"i{i}",
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {
                        "color": "red" if i % 2 == 0 else "green",
                        "shape": "square" if i % 2 == 0 else "circle",
                        "hidden": "secret" if i % 2 == 0 else "public",
                        "rank": i,
                        "loc": {"city": "paris" if i % 2 == 0 else "lyon"},
                    },
                }
                for i in range(N)
            ]
        )
        wait_for_ids(self.index, [f"i{i}" for i in range(N)])

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def _query_ids(self, filters):
        """Same filter through the vector path, for comparison."""
        results = self.index.query(
            query_vectors=np.random.rand(DIM).astype(np.float32),
            top_k=N,
            filters=filters,
        )
        return {r["id"] for r in results}

    # -- schema round-trip ------------------------------------------------ #

    def test_metadata_schema_round_trips(self):
        self.assertEqual(
            self.index.metadata_schema,
            {
                "color": {"filterable": True, "pattern": True, "full_text": False},
                "shape": {"filterable": True, "pattern": False, "full_text": False},
                "hidden": {"filterable": False, "pattern": False, "full_text": False},
            },
        )

    # -- happy paths ------------------------------------------------------ #

    def test_no_filters_matches_all(self):
        self.assertEqual(set(_ids(self.index.query_metadata())), EVEN | ODD)

    def test_rows_are_id_dicts_without_score(self):
        # Filter-only rows match core's list[MetadataResult]: {"id"} only,
        # no `score` key (nothing to score without `text`).
        rows = self.index.query_metadata({"color": "red"})
        self.assertTrue(all(row == {"id": row["id"]} for row in rows))

    def test_equality(self):
        self.assertEqual(set(_ids(self.index.query_metadata({"color": "red"}))), EVEN)

    def test_nested_dot_path(self):
        self.assertEqual(
            set(_ids(self.index.query_metadata({"loc.city": "paris"}))), EVEN
        )

    def test_regex_on_pattern_field(self):
        self.assertEqual(
            set(_ids(self.index.query_metadata({"color": {"$regex": "^r"}}))), EVEN
        )

    def test_contains_on_pattern_field(self):
        self.assertEqual(
            set(_ids(self.index.query_metadata({"color": {"$contains": "ree"}}))), ODD
        )

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.query_metadata({"color": "mauve"}), [])

    # -- ordering and paging ---------------------------------------------- #

    def test_order_by_ascending_and_descending(self):
        all_ranks = {"rank": {"$gte": 0}}
        self.assertEqual(
            _ids(self.index.query_metadata(all_ranks, order_by="rank")),
            [f"i{i}" for i in range(N)],
        )
        self.assertEqual(
            _ids(
                self.index.query_metadata(all_ranks, order_by="rank", ascending=False)
            ),
            [f"i{i}" for i in reversed(range(N))],
        )

    def test_order_by_mongo_style_dict(self):
        # {field: -1} is core's form; the wrapper normalizes it for the service.
        self.assertEqual(
            _ids(
                self.index.query_metadata({"rank": {"$gte": 0}}, order_by={"rank": -1})
            ),
            [f"i{i}" for i in reversed(range(N))],
        )

    def test_order_by_dict_with_two_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.query_metadata(order_by={"rank": 1, "color": -1})

    def test_top_k_applies_after_sort(self):
        self.assertEqual(
            _ids(
                self.index.query_metadata(
                    {"rank": {"$gte": 0}}, order_by="rank", top_k=2
                )
            ),
            ["i0", "i1"],
        )

    # -- policy enforcement ----------------------------------------------- #

    def test_regex_on_non_pattern_field_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.query_metadata({"shape": {"$regex": "^sq"}})
        # ...but the same filter is fine on the vector path, which post-filters.
        self.assertEqual(self._query_ids({"shape": {"$regex": "^sq"}}), EVEN)

    def test_non_filterable_field_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.query_metadata({"hidden": "secret"})
        self.assertEqual(self._query_ids({"hidden": "secret"}), EVEN)

    def test_unsupported_operator_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.query_metadata({"rank": {"$type": "number"}})


class TestQueryMetadataDefaultPosture(unittest.TestCase):
    """No metadata_schema — everything is filterable, nothing is a pattern."""

    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"query_metadata_default_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        self.index.upsert(
            [
                {
                    "id": f"i{i}",
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {"color": "red" if i % 2 == 0 else "green"},
                }
                for i in range(N)
            ]
        )
        wait_for_ids(self.index, [f"i{i}" for i in range(N)])

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def test_describe_reports_empty_schema(self):
        self.assertEqual(self.index.metadata_schema, {})

    def test_equality_works_without_opt_in(self):
        self.assertEqual(set(_ids(self.index.query_metadata({"color": "red"}))), EVEN)

    def test_regex_needs_a_pattern_field(self):
        # Default posture indexes every field but builds no regex dictionary,
        # so query_metadata cannot resolve $regex on any of them.
        with self.assertRaises(ValueError):
            self.index.query_metadata({"color": {"$regex": "^r"}})


OPERATOR_SCHEMA = {
    "color": {"filterable": True, "pattern": True},
    "rank": {"filterable": True},
    "tags": {"filterable": True},
    "author": {"filterable": True},
}

# One fixture covering every operator, plus the two cases that make operator
# semantics ambiguous: documents that omit a field entirely (o2, o4 have no
# `author`) and an array-valued field (`tags`, including an empty array on o3).
#
#   id  color  rank  tags                      author
#   o0  red     0    [design, search]          ada
#   o1  green  10    [design]                  bob
#   o2  blue   20    [search]                  <missing>
#   o3  red    30    []                        ada
#   o4  green  40    [design, search, ml]      <missing>
OPERATOR_ROWS = [
    ("o0", "red", 0, ["design", "search"], "ada"),
    ("o1", "green", 10, ["design"], "bob"),
    ("o2", "blue", 20, ["search"], None),
    ("o3", "red", 30, [], "ada"),
    ("o4", "green", 40, ["design", "search", "ml"], None),
]
ALL_OPS = {"o0", "o1", "o2", "o3", "o4"}

# Every operator in openapi.json's documented set, each with an expected answer
# that is a proper subset of the corpus — so a filter that silently matched
# everything, or nothing, fails rather than passing by luck.
OPERATOR_CASES = [
    ("$eq", {"color": {"$eq": "red"}}, {"o0", "o3"}),
    ("$ne", {"color": {"$ne": "red"}}, {"o1", "o2", "o4"}),
    ("$in", {"color": {"$in": ["red", "blue"]}}, {"o0", "o2", "o3"}),
    ("$nin", {"color": {"$nin": ["red"]}}, {"o1", "o2", "o4"}),
    ("$gt", {"rank": {"$gt": 20}}, {"o3", "o4"}),
    ("$gte", {"rank": {"$gte": 20}}, {"o2", "o3", "o4"}),
    ("$lt", {"rank": {"$lt": 20}}, {"o0", "o1"}),
    ("$lte", {"rank": {"$lte": 20}}, {"o0", "o1", "o2"}),
    ("$exists-true", {"author": {"$exists": True}}, {"o0", "o1", "o3"}),
    ("$exists-false", {"author": {"$exists": False}}, {"o2", "o4"}),
    ("$and", {"$and": [{"color": "red"}, {"rank": {"$gte": 30}}]}, {"o3"}),
    ("$or", {"$or": [{"color": "blue"}, {"rank": {"$lt": 10}}]}, {"o0", "o2"}),
    ("$nor", {"$nor": [{"color": "red"}, {"color": "green"}]}, {"o2"}),
    # `$not` is deliberately absent — openapi.json documents it, but the engine
    # rejects it on both read paths. See test_not_is_documented_but_unsupported.
    ("$regex", {"color": {"$regex": "^r"}}, {"o0", "o3"}),
    ("$contains", {"color": {"$contains": "ree"}}, {"o1", "o4"}),
]


class TestFilterOperators(unittest.TestCase):
    """All fifteen documented operators, on both read paths.

    Only four (`$in`, `$gte`, `$regex`, `$contains`) were exercised anywhere in
    this SDK before, and `$gte` only incidentally as a match-all inside the
    order_by tests. Read-only, so one fixture serves the whole class.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"operators_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema=OPERATOR_SCHEMA,
        )
        items = []
        for doc_id, color, rank, tags, author in OPERATOR_ROWS:
            metadata = {"color": color, "rank": rank, "tags": tags}
            # `author` is omitted entirely rather than set to null, so these
            # exercise absence rather than a stored null.
            if author is not None:
                metadata["author"] = author
            items.append(
                {
                    "id": doc_id,
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": metadata,
                }
            )
        cls.index.upsert(items)
        wait_for_ids(cls.index, ALL_OPS)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _meta_ids(self, filters):
        return {row["id"] for row in self.index.query_metadata(filters)}

    def _vector_ids(self, filters):
        return {
            r["id"]
            for r in self.index.query(
                query_vectors=np.random.rand(DIM).astype(np.float32),
                top_k=len(ALL_OPS),
                filters=filters,
            )
        }

    def test_every_operator_on_the_metadata_path(self):
        for name, filters, expected in OPERATOR_CASES:
            with self.subTest(operator=name):
                self.assertEqual(self._meta_ids(filters), expected)

    def test_every_operator_on_the_vector_path(self):
        # The same operators through query(), which post-filters over decrypted
        # metadata rather than resolving from the index. Same answers required.
        for name, filters, expected in OPERATOR_CASES:
            with self.subTest(operator=name):
                self.assertEqual(self._vector_ids(filters), expected)

    def test_both_read_paths_agree(self):
        # The congruence check: whatever the right answer is, the two paths must
        # not disagree. This is the cheapest broad guard we have — it catches a
        # divergence between the indexed path and the post-filter fallback even
        # for operators whose expected value above turns out to be wrong.
        for name, filters, _ in OPERATOR_CASES:
            with self.subTest(operator=name):
                self.assertEqual(
                    self._meta_ids(filters),
                    self._vector_ids(filters),
                    f"{name}: query_metadata and query disagree",
                )

    # -- missing fields ---------------------------------------------------- #

    def test_missing_field_is_excluded_by_ne_but_included_by_nin(self):
        # The asymmetry the design doc specifies: `$ne` drops documents lacking
        # the field, `$nin` keeps them. Both are defensible in isolation; what
        # matters is that the contract is pinned rather than inferred.
        self.assertEqual(self._meta_ids({"author": {"$ne": "ada"}}), {"o1"})
        self.assertEqual(
            self._meta_ids({"author": {"$nin": ["ada"]}}), {"o1", "o2", "o4"}
        )

    def test_missing_field_is_included_by_nor(self):
        self.assertEqual(
            self._meta_ids({"$nor": [{"author": "ada"}]}), {"o1", "o2", "o4"}
        )

    def test_not_is_documented_but_unsupported(self):
        # openapi.json lists `$not` among the supported operators and the SDK
        # docstrings repeat it, but the engine rejects it on BOTH read paths:
        #   "Invalid input: Unsupported metadata operator: $not"
        #
        # Pinned as current behaviour so nobody rediscovers it the hard way.
        # Either the operator gets implemented or it comes out of the documented
        # set; whichever happens, this test fails and forces the decision to be
        # made explicitly rather than drifting.
        filters = {"color": {"$not": {"$eq": "red"}}}
        with self.assertRaises(ValueError):
            self.index.query_metadata(filters)
        with self.assertRaises(ValueError):
            self.index.query(
                query_vectors=np.random.rand(DIM).astype(np.float32),
                top_k=len(ALL_OPS),
                filters=filters,
            )

    # -- arrays ------------------------------------------------------------- #

    def test_bare_value_on_an_array_field_means_contains(self):
        self.assertEqual(self._meta_ids({"tags": "design"}), {"o0", "o1", "o4"})

    def test_in_on_an_array_field_means_any_of(self):
        self.assertEqual(
            self._meta_ids({"tags": {"$in": ["ml", "search"]}}), {"o0", "o2", "o4"}
        )

    def test_has_all_of_these_via_and_of_two_memberships(self):
        # "contains all" has no dedicated operator; it is expressed as $and of
        # two membership conditions.
        self.assertEqual(
            self._meta_ids({"$and": [{"tags": "design"}, {"tags": "search"}]}),
            {"o0", "o4"},
        )

    def test_empty_array_matches_no_membership(self):
        # o3's tags are [], so it can never satisfy a membership condition.
        for filters in ({"tags": "design"}, {"tags": {"$in": ["design", "ml"]}}):
            with self.subTest(filters=filters):
                self.assertNotIn("o3", self._meta_ids(filters))

    # -- degenerate operands ------------------------------------------------ #

    def test_empty_filter_matches_everything(self):
        self.assertEqual(self._meta_ids({}), ALL_OPS)

    def test_empty_in_list_matches_nothing(self):
        # Qdrant ships dedicated regression tests for empty match-any/match-none
        # because both were real reported bugs. Ours were untested entirely.
        self.assertEqual(self._meta_ids({"color": {"$in": []}}), set())

    def test_empty_nin_list_matches_everything(self):
        self.assertEqual(self._meta_ids({"color": {"$nin": []}}), ALL_OPS)

    def test_empty_boolean_operands(self):
        # $and over nothing is vacuously true; $or over nothing is vacuously
        # false. Both are easy to get backwards in a query planner.
        self.assertEqual(self._meta_ids({"$and": []}), ALL_OPS)
        self.assertEqual(self._meta_ids({"$or": []}), set())

    # -- type handling ------------------------------------------------------ #

    def test_int_and_float_are_the_same_key(self):
        # All numbers share one index, so 20 and 20.0 must resolve identically
        # on both equality and range bounds.
        self.assertEqual(self._meta_ids({"rank": 20}), self._meta_ids({"rank": 20.0}))
        self.assertEqual(
            self._meta_ids({"rank": {"$gte": 20}}),
            self._meta_ids({"rank": {"$gte": 20.0}}),
        )

    def test_cross_type_comparison_does_not_match_silently(self):
        # `rank` holds numbers; filtering it with a string must either raise or
        # return nothing. What it must not do is match — a silent wrong answer
        # is the failure mode nobody reports as a bug.
        try:
            got = self._meta_ids({"rank": "20"})
        except ValueError:
            return  # raising is an acceptable contract
        self.assertEqual(got, set(), "a string filter matched a numeric field")


class TestDatetimeHandling(unittest.TestCase):
    """What actually happens to a native `datetime` passed as metadata.

    This is the boundary the design doc flagged as the one place documentation
    and code may disagree, and they do. cyborgdb-core converts datetimes to
    epoch millis (its metadata_datetime_test.py has `test_stored_as_epoch_millis`
    and `test_query_metadata_datetime_range`), so range filters on a date work
    there. This SDK serialises to an ISO 8601 string instead, so the value round
    trips and equality matches, but every range comparison fails.

    These tests pin the SDK's real behaviour rather than the intended contract,
    so the gap is visible and a fix would surface here as a failure.
    """

    BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

    @classmethod
    def setUpClass(cls):
        cls.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        cls.index = cls.client.create_index(
            f"datetime_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
            metadata_schema={
                "created": {"filterable": True},
                "created_ms": {"filterable": True},
            },
        )
        cls.index.upsert(
            [
                {
                    "id": f"t{i}",
                    "vector": np.random.rand(DIM).astype(np.float32).tolist(),
                    "metadata": {
                        "created": cls.BASE + timedelta(days=10 * i),
                        # The workaround callers currently need for ranges.
                        "created_ms": int(
                            (cls.BASE + timedelta(days=10 * i)).timestamp() * 1000
                        ),
                    },
                }
                for i in range(3)
            ]
        )
        wait_for_ids(cls.index, ["t0", "t1", "t2"])

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_datetime_is_stored_as_an_iso_string(self):
        # Not epoch millis, which is what core stores. The SDK hands the
        # datetime to JSON serialisation and the ISO form is what lands.
        row = self.index.get(["t0"], include=["metadata"])[0]
        self.assertEqual(row["metadata"]["created"], "2026-01-01T00:00:00+00:00")

    def test_equality_on_a_datetime_matches(self):
        # Equality survives because it degenerates to string comparison.
        got = {r["id"] for r in self.index.query_metadata({"created": self.BASE})}
        self.assertEqual(got, {"t0"})

    def test_range_on_a_datetime_is_rejected(self):
        # The consequence of ISO-string storage: a range comparison against a
        # string is invalid, and the service says so explicitly —
        #   "$gte requires a numeric value, got: \"2026-01-06T00:00:00+00:00\""
        #
        # Core supports this query. Pinned here as the SDK's current behaviour;
        # if the SDK starts converting to epoch millis this test fails and
        # should be replaced with the range assertion core already has.
        with self.assertRaises(ValueError) as caught:
            self.index.query_metadata(
                {"created": {"$gte": self.BASE + timedelta(days=5)}}
            )
        self.assertIn("numeric", str(caught.exception))

    def test_epoch_millis_supports_ranges(self):
        # The workaround: convert to epoch millis yourself and ranges work.
        cutoff = int((self.BASE + timedelta(days=5)).timestamp() * 1000)
        got = {
            r["id"] for r in self.index.query_metadata({"created_ms": {"$gte": cutoff}})
        }
        self.assertEqual(got, {"t1", "t2"})

    def test_epoch_millis_round_trips_exactly(self):
        # Millisecond precision must survive the JSON round trip — a float
        # conversion anywhere would corrupt the low digits.
        expected = int(self.BASE.timestamp() * 1000)
        row = self.index.get(["t0"], include=["metadata"])[0]
        self.assertEqual(row["metadata"]["created_ms"], expected)


if __name__ == "__main__":
    unittest.main()
