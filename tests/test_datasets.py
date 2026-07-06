import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import cyborgdb.datasets as datasets_mod
from cyborgdb import DEFAULT_SAMPLE_DATASET, load_sample_dataset
from cyborgdb.datasets import _DatasetEntry

# The mocked payload is the *raw* hosted shape (no items/sample_queries);
# those convenience fields are rebuilt by the loader's hydrate step.
FAKE_RAW = {
    "name": "quickstart-75k",
    "version": 1,
    "description": "test fixture",
    "dimension": 3,
    "metric": "euclidean",
    "count": 2,
    "exampleFilters": [
        {"name": "eq", "filter": {"string": "a"}, "demonstrates": "equality"},
    ],
    "ids": ["item_0", "item_1"],
    "vectors": [[1, 2, 3], [4, 5, 6]],
    "metadata": [{"number": 0, "string": "a"}, {"number": 1, "string": "b"}],
    "queries": [[1, 2, 3]],
    "metadata_queries": [{"string": "a"}],
    "metadata_query_names": ["eq string a"],
    "untrained_neighbors": [[0]],
    "trained_neighbors": [[0]],
    "untrained_metadata_matches": [[1]],
    "trained_metadata_matches": [[1]],
    "untrained_metadata_neighbors": [[[0]]],
    "trained_metadata_neighbors": [[[0]]],
    "untrained_recall": 1.0,
    "trained_recall": 0.94,
    "num_untrained_vectors": 1,
    "num_trained_vectors": 1,
}


def _raw_bytes(raw):
    """The decompressed JSON bytes the loader hashes."""
    return json.dumps(raw).encode("utf-8")


def _gzip_response(raw):
    """A MagicMock standing in for a requests.Response with gzipped content."""
    resp = MagicMock()
    resp.content = gzip.compress(_raw_bytes(raw))
    resp.raise_for_status.return_value = None
    return resp


# SHA-256 of the decompressed fixture, matching what the loader verifies.
_FAKE_SHA256 = hashlib.sha256(_raw_bytes(FAKE_RAW)).hexdigest()


class TestLoadSampleDataset(unittest.TestCase):
    """Offline unit tests for the sample dataset loader (requests mocked)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name
        # Pin the catalog digest to the fixture so integrity verification passes.
        self._orig_entry = datasets_mod._DATASETS["quickstart-75k"]
        datasets_mod._DATASETS["quickstart-75k"] = _DatasetEntry(
            object_path=self._orig_entry.object_path, sha256=_FAKE_SHA256
        )

    def tearDown(self):
        datasets_mod._DATASETS["quickstart-75k"] = self._orig_entry
        self._tmp.cleanup()

    @patch("cyborgdb.datasets.requests.get")
    def test_download_and_hydrate(self, mock_get):
        mock_get.return_value = _gzip_response(FAKE_RAW)

        ds = load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(ds.count, 2)
        self.assertEqual(ds.dimension, 3)
        # items are built from ids + vectors + metadata
        self.assertEqual(len(ds.items), 2)
        self.assertEqual(ds.items[0]["id"], "item_0")
        self.assertEqual(ds.items[0]["vector"], [1, 2, 3])
        self.assertEqual(ds.items[1]["metadata"], {"number": 1, "string": "b"})
        # sample_queries are the leading queries
        self.assertEqual(ds.sample_queries, [[1, 2, 3]])
        self.assertEqual(ds.example_filters[0]["filter"], {"string": "a"})
        # raw ground-truth fields pass through unchanged
        self.assertEqual(ds.trained_recall, 0.94)
        self.assertEqual(ds.untrained_neighbors, [[0]])

    @patch("cyborgdb.datasets.requests.get")
    def test_second_call_uses_cache(self, mock_get):
        mock_get.return_value = _gzip_response(FAKE_RAW)
        load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)
        load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)
        self.assertEqual(mock_get.call_count, 1)

    @patch("cyborgdb.datasets.requests.get")
    def test_force_download_refetches(self, mock_get):
        mock_get.return_value = _gzip_response(FAKE_RAW)
        load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)
        load_sample_dataset(
            DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir, force_download=True
        )
        self.assertEqual(mock_get.call_count, 2)

    @patch("cyborgdb.datasets.requests.get")
    def test_unknown_dataset_raises(self, mock_get):
        with self.assertRaises(ValueError):
            load_sample_dataset("does-not-exist", cache_dir=self.cache_dir)
        mock_get.assert_not_called()

    @patch("cyborgdb.datasets.requests.get")
    def test_download_failure_raises(self, mock_get):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = resp
        with self.assertRaises(RuntimeError):
            load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)

    @patch("cyborgdb.datasets.requests.get")
    def test_integrity_mismatch_raises(self, mock_get):
        mock_get.return_value = _gzip_response(FAKE_RAW)
        datasets_mod._DATASETS["quickstart-75k"] = _DatasetEntry(
            object_path=self._orig_entry.object_path, sha256="0" * 64
        )
        with self.assertRaisesRegex(RuntimeError, "Integrity check failed"):
            load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)

    @patch("cyborgdb.datasets.requests.get")
    def test_tampered_cache_refetches(self, mock_get):
        mock_get.return_value = _gzip_response(FAKE_RAW)
        load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)
        self.assertEqual(mock_get.call_count, 1)

        # Tamper with the cached file; the pinned digest no longer matches.
        cache_file = Path(self.cache_dir) / "quickstart-75k_v1_dataset.json"
        cache_file.write_text("tampered", "utf-8")
        load_sample_dataset(DEFAULT_SAMPLE_DATASET, cache_dir=self.cache_dir)
        self.assertEqual(mock_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
