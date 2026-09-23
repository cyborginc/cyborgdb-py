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
        time.sleep(2)

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
        time.sleep(2)

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
