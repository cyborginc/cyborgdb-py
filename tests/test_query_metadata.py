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
        self.assertEqual(set(self.index.query_metadata()), EVEN | ODD)

    def test_equality(self):
        self.assertEqual(set(self.index.query_metadata({"color": "red"})), EVEN)

    def test_nested_dot_path(self):
        self.assertEqual(set(self.index.query_metadata({"loc.city": "paris"})), EVEN)

    def test_regex_on_pattern_field(self):
        self.assertEqual(
            set(self.index.query_metadata({"color": {"$regex": "^r"}})), EVEN
        )

    def test_contains_on_pattern_field(self):
        self.assertEqual(
            set(self.index.query_metadata({"color": {"$contains": "ree"}})), ODD
        )

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.query_metadata({"color": "mauve"}), [])

    # -- ordering and paging ---------------------------------------------- #

    def test_order_by_ascending_and_descending(self):
        all_ranks = {"rank": {"$gte": 0}}
        self.assertEqual(
            self.index.query_metadata(all_ranks, order_by="rank"),
            [f"i{i}" for i in range(N)],
        )
        self.assertEqual(
            self.index.query_metadata(all_ranks, order_by="rank", ascending=False),
            [f"i{i}" for i in reversed(range(N))],
        )

    def test_order_by_mongo_style_dict(self):
        # {field: -1} is core's form; the wrapper normalizes it for the service.
        self.assertEqual(
            self.index.query_metadata({"rank": {"$gte": 0}}, order_by={"rank": -1}),
            [f"i{i}" for i in reversed(range(N))],
        )

    def test_order_by_dict_with_two_fields_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.query_metadata(order_by={"rank": 1, "color": -1})

    def test_top_k_applies_after_sort(self):
        self.assertEqual(
            self.index.query_metadata({"rank": {"$gte": 0}}, order_by="rank", top_k=2),
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
        self.assertEqual(set(self.index.query_metadata({"color": "red"})), EVEN)

    def test_regex_needs_a_pattern_field(self):
        # Default posture indexes every field but builds no regex dictionary,
        # so query_metadata cannot resolve $regex on any of them.
        with self.assertRaises(ValueError):
            self.index.query_metadata({"color": {"$regex": "^r"}})


if __name__ == "__main__":
    unittest.main()
