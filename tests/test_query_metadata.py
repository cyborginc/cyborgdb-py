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
import time
import unittest
import uuid
from datetime import date, datetime, timedelta, timezone

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

# o2 and o4 omit `author` entirely, and o3's `tags` is empty — the two cases
# that make operator semantics ambiguous.
OPERATOR_ROWS = [
    ("o0", "red", 0, ["design", "search"], "ada"),
    ("o1", "green", 10, ["design"], "bob"),
    ("o2", "blue", 20, ["search"], None),
    ("o3", "red", 30, [], "ada"),
    ("o4", "green", 40, ["design", "search", "ml"], None),
]
ALL_OPS = {"o0", "o1", "o2", "o3", "o4"}

# Each expected answer is a proper subset of the corpus, so a filter that
# silently matched everything or nothing fails rather than passing by luck.
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
    """All fifteen documented operators, on both read paths."""

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
            # Omitted rather than null: these exercise absence.
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
        # query() post-filters over decrypted metadata rather than resolving
        # from the index; the answers must still match.
        for name, filters, expected in OPERATOR_CASES:
            with self.subTest(operator=name):
                self.assertEqual(self._vector_ids(filters), expected)

    def test_both_read_paths_agree(self):
        # Anchored as well as compared: a bug in the shared filter parser would
        # break both paths identically and slip past an agreement-only check.
        for name, filters, expected in OPERATOR_CASES:
            with self.subTest(operator=name):
                meta, vector = self._meta_ids(filters), self._vector_ids(filters)
                self.assertEqual(meta, vector, f"{name}: paths disagree")
                self.assertEqual(meta, expected, f"{name}: both paths wrong")

    # -- missing fields ---------------------------------------------------- #

    def test_missing_field_is_excluded_by_ne_but_included_by_nin(self):
        # `$ne` drops documents lacking the field, `$nin` keeps them. Both are
        # defensible; the point is that the contract is pinned, not inferred.
        self.assertEqual(self._meta_ids({"author": {"$ne": "ada"}}), {"o1"})
        self.assertEqual(
            self._meta_ids({"author": {"$nin": ["ada"]}}), {"o1", "o2", "o4"}
        )

    def test_missing_field_is_included_by_nor(self):
        self.assertEqual(
            self._meta_ids({"$nor": [{"author": "ada"}]}), {"o1", "o2", "o4"}
        )

    def test_not_operator_works_on_both_paths(self):
        # KNOWN BUG — fails today. cyborgdb-core#2395: the engine rejects `$not`
        # on both read paths although openapi.json documents it.
        filters = {"color": {"$not": {"$eq": "red"}}}
        self.assertEqual(self._meta_ids(filters), {"o1", "o2", "o4"})
        self.assertEqual(self._vector_ids(filters), {"o1", "o2", "o4"})

    # -- arrays ------------------------------------------------------------- #

    def test_bare_value_on_an_array_field_means_contains(self):
        self.assertEqual(self._meta_ids({"tags": "design"}), {"o0", "o1", "o4"})

    def test_in_on_an_array_field_means_any_of(self):
        self.assertEqual(
            self._meta_ids({"tags": {"$in": ["ml", "search"]}}), {"o0", "o2", "o4"}
        )

    def test_has_all_of_these_via_and_of_two_memberships(self):
        # No dedicated operator; expressed as $and of two memberships.
        self.assertEqual(
            self._meta_ids({"$and": [{"tags": "design"}, {"tags": "search"}]}),
            {"o0", "o4"},
        )

    def test_empty_array_matches_no_membership(self):
        for filters in ({"tags": "design"}, {"tags": {"$in": ["design", "ml"]}}):
            with self.subTest(filters=filters):
                self.assertNotIn("o3", self._meta_ids(filters))

    # -- degenerate operands ------------------------------------------------ #

    def test_empty_filter_matches_everything(self):
        self.assertEqual(self._meta_ids({}), ALL_OPS)

    def test_empty_in_list_matches_nothing(self):
        self.assertEqual(self._meta_ids({"color": {"$in": []}}), set())

    def test_empty_nin_list_matches_everything(self):
        self.assertEqual(self._meta_ids({"color": {"$nin": []}}), ALL_OPS)

    def test_empty_boolean_operands(self):
        # $and over nothing is vacuously true, $or vacuously false.
        self.assertEqual(self._meta_ids({"$and": []}), ALL_OPS)
        self.assertEqual(self._meta_ids({"$or": []}), set())

    # -- type handling ------------------------------------------------------ #

    def test_int_and_float_are_the_same_key(self):
        # Anchored to the expected answer, not just compared to each other:
        # two empty results would otherwise satisfy the comparison.
        self.assertEqual(self._meta_ids({"rank": 20}), {"o2"})
        self.assertEqual(self._meta_ids({"rank": 20.0}), {"o2"})
        self.assertEqual(self._meta_ids({"rank": {"$gte": 20}}), {"o2", "o3", "o4"})
        self.assertEqual(self._meta_ids({"rank": {"$gte": 20.0}}), {"o2", "o3", "o4"})

    def test_cross_type_comparison_does_not_match_silently(self):
        # Either contract is acceptable; matching is not.
        try:
            got = self._meta_ids({"rank": "20"})
        except ValueError:
            return  # raising is an acceptable contract
        self.assertEqual(got, set(), "a string filter matched a numeric field")


class TestDatetimeHandling(unittest.TestCase):
    """datetime/date objects are coerced to epoch milliseconds before reaching the engine.

    Mirrors cyborgdb-core tests/metadata_datetime_test.py.
    """

    # 2026-01-01T00:00:00Z in epoch milliseconds
    _BASE_MS = 1767225600000
    _BASE_DT = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def setUp(self):
        from cyborgdb.client.encrypted_index import _coerce_datetimes

        self._coerce = _coerce_datetimes
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"dt_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        self.index.upsert(
            [
                {
                    "id": f"t{i}",
                    "vector": [float(i + 1) / 10] * DIM,
                    "metadata": {"created": self._BASE_DT + timedelta(days=i)},
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

    # -- unit conversion (no network) -------------------------------------- #

    def test_utc_aware_datetime_to_millis(self):
        self.assertEqual(self._coerce(self._BASE_DT), self._BASE_MS)

    def test_naive_datetime_treated_as_utc(self):
        self.assertEqual(self._coerce(datetime(2026, 1, 1)), self._BASE_MS)

    def test_date_object_midnight_utc(self):
        self.assertEqual(self._coerce(date(2026, 1, 1)), self._BASE_MS)

    def test_offset_datetime_normalised_to_utc(self):
        tz_plus5 = timezone(timedelta(hours=5))
        dt = datetime(2026, 1, 1, 5, 0, 0, tzinfo=tz_plus5)
        self.assertEqual(self._coerce(dt), self._BASE_MS)

    def test_millisecond_precision_truncated(self):
        dt = datetime(2026, 1, 1, 0, 0, 0, 500999, tzinfo=timezone.utc)
        self.assertEqual(self._coerce(dt), self._BASE_MS + 500)

    def test_plain_string_passthrough(self):
        self.assertEqual(self._coerce("hello"), "hello")

    # -- integration: read-back and filter paths --------------------------- #

    def test_get_returns_epoch_millis_integer(self):
        results = self.index.get(["t0"], include=["metadata"])
        val = results[0]["metadata"]["created"]
        self.assertIsInstance(val, int)
        self.assertEqual(val, self._BASE_MS)

    def test_range_on_a_datetime_works(self):
        low = self._BASE_DT + timedelta(days=1)
        high = self._BASE_DT + timedelta(days=3)
        rows = self.index.query_metadata({"created": {"$gte": low, "$lt": high}})
        self.assertEqual(set(_ids(rows)), {"t1", "t2"})

    def test_query_metadata_nested_and_with_datetime_predicates(self):
        low = self._BASE_DT + timedelta(days=1)
        high = self._BASE_DT + timedelta(days=3)
        rows = self.index.query_metadata(
            {"$and": [{"created": {"$gte": low}}, {"created": {"$lt": high}}]}
        )
        self.assertEqual(set(_ids(rows)), {"t1", "t2"})

    def test_query_metadata_in_with_datetime_list(self):
        rows = self.index.query_metadata(
            {"created": {"$in": [self._BASE_DT, self._BASE_DT + timedelta(days=2)]}}
        )
        self.assertEqual(set(_ids(rows)), {"t0", "t2"})

    def test_query_metadata_or_with_datetime_predicates(self):
        rows = self.index.query_metadata(
            {
                "$or": [
                    {"created": self._BASE_DT},
                    {"created": self._BASE_DT + timedelta(days=2)},
                ]
            }
        )
        self.assertEqual(set(_ids(rows)), {"t0", "t2"})

    def test_query_vector_path_filter_with_datetime(self):
        cutoff = self._BASE_DT + timedelta(days=2)
        results = self.index.query(
            query_vectors=np.array([0.1] * DIM, dtype=np.float32),
            top_k=N,
            filters={"created": {"$gte": cutoff}},
        )
        self.assertEqual({r["id"] for r in results}, {"t2", "t3"})


if __name__ == "__main__":
    unittest.main()
