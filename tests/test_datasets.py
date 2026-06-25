import gzip
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from cyborgdb import DEFAULT_SAMPLE_DATASET, load_sample_dataset

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


def _gzip_response(raw):
    """A MagicMock standing in for a requests.Response with gzipped content."""
    resp = MagicMock()
    resp.content = gzip.compress(json.dumps(raw).encode("utf-8"))
    resp.raise_for_status.return_value = None
    return resp


class TestLoadSampleDataset(unittest.TestCase):
    """Offline unit tests for the sample dataset loader (requests mocked)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
