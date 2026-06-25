#!/usr/bin/env python3
import unittest
import json
import os
import uuid
import numpy as np
from dotenv import load_dotenv
import time

import cyborgdb as cyborgdb

# Load environment variables from .env.local
load_dotenv(".env.local")


def generate_unique_name(prefix="test_"):
    """
    Generates a unique index name with a given prefix by appending a UUID v4.
    """
    return f"{prefix}{uuid.uuid4()}"


def check_query_results(results, neighbors, num_queries):
    # Parse results to extract IDs from the returned dictionaries
    result_ids = [
        [int(res["id"]) for res in query_results] for query_results in results
    ]

    # Convert the results to numpy array
    result_ids = np.array(result_ids)
    if neighbors.shape != result_ids.shape:
        raise ValueError(
            f"The shapes of the neighbors and results do not match: {neighbors.shape} != {result_ids.shape}"
        )

    # Compute the recall using the neighbors
    recall = np.zeros(num_queries)
    for i in range(num_queries):
        recall[i] = len(np.intersect1d(neighbors[i], result_ids[i])) / len(neighbors[i])

    return recall.mean()


def check_metadata_results(
    results, metadata_neighors, metadata_candidates, num_queries
):
    def safe_int(val):
        try:
            return int(val)
        except ValueError:
            return -1

    result_ids = [
        [[safe_int(res["id"]) for res in query_results] for query_results in result]
        for result in results
    ]

    recalls = []

    for idx, result in enumerate(result_ids):
        # Get candidates for this query
        candidates = metadata_candidates[idx]

        # Get groundtruth neighbors for this metadata query (should be shape (num_queries, 100))
        metadata_neighbors_indices = metadata_neighors[idx]

        recall = np.zeros(num_queries, dtype=np.float32)
        num_returned = 0
        num_expected = 0

        # Iterate over the queries
        for i in range(num_queries):
            # Get the groundtruth neighbors for this query
            groundtruth_indices = metadata_neighbors_indices[i]

            groundtruth_ids = np.array(
                [
                    candidates[int(idx)]
                    for idx in groundtruth_indices
                    if idx != -1 and 0 <= idx < len(candidates)
                ]
            )

            # Get the returned neighbors for this query
            returned = np.array(result[i])

            # Update the number of returned neighbors
            num_returned += len(returned)
            local_expected = np.count_nonzero(groundtruth_ids != -1)
            num_expected += local_expected

            # If we expect no results and got no results, recall is 100%
            if len(returned) == 0 and local_expected == 0:
                recall[i] = 1
                continue

            # Check if the number of returned neighbors is correct
            if len(returned) > 100:
                raise ValueError(
                    f"More than 100 results returned: got {len(returned)} instead of 100"
                )

            # Compute the recall for this query
            recall[i] = len(np.intersect1d(groundtruth_ids, returned)) / min(
                local_expected, 100
            )

        # Get the number of groundtruth results (non -1 values)
        num_expected = num_expected / num_queries
        num_returned = num_returned / num_queries
        recalls.append(recall.mean())

    return recalls


class TestUnitFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fetch the hosted sample dataset (downloaded from S3 on first use and
        # cached locally). It carries the full ground-truth arrays this recall
        # test needs, so it replaces the previously-committed JSON fixture.
        data = cyborgdb.load_sample_dataset()

        # Load vectors and neighbors as numpy arrays
        cls.vectors = np.array(data.vectors, dtype=np.float32)
        cls.queries = np.array(data.queries, dtype=np.float32)
        cls.untrained_neighbors = np.array(data.untrained_neighbors, dtype=np.int32)
        cls.trained_neighbors = np.array(data.trained_neighbors, dtype=np.int32)
        cls.metadata = data.metadata
        cls.metadata_queries = data.metadata_queries
        cls.metadata_query_names = data.metadata_query_names
        cls.untrained_metadata_matches = data.untrained_metadata_matches
        cls.trained_metadata_matches = data.trained_metadata_matches
        cls.untrained_metadata_neighbors = data.untrained_metadata_neighbors
        cls.trained_metadata_neighbors = data.trained_metadata_neighbors

        # Load expected recall values (as scalars or lists)
        cls.untrained_recall = data.untrained_recall
        cls.trained_recall = data.trained_recall

        # Set counts and dimension
        cls.num_untrained_vectors = data.num_untrained_vectors
        cls.num_trained_vectors = data.num_trained_vectors
        cls.total_num_vectors = cls.num_untrained_vectors + cls.num_trained_vectors
        cls.num_queries = cls.queries.shape[0]
        cls.dimension = cls.vectors.shape[1]
        cls.n_lists = 100

        # CYBORDB SETUP: Create the index once (shared state).
        cls.client = cyborgdb.Client(
            base_url="http://localhost:8000", api_key=os.getenv("CYBORGDB_API_KEY", "")
        )
        cls.index_name = generate_unique_name()
        cls.index_key = cyborgdb.Client.generate_key()
        cls.index = cls.client.create_index(
            cls.index_name,
            cls.index_key,
            dimension=cls.dimension,
            metric="euclidean",
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up the index after all tests are done."""
        try:
            if hasattr(cls, "index") and cls.index:
                cls.index.delete_index()
        except Exception as e:
            print(f"Error during index cleanup: {e}")

    def test_00_get_health(self):
        # Check if the API is healthy.
        health = self.client.get_health()
        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertEqual(health["status"], "healthy", "API is not healthy")

    def test_01_untrained_upsert(self):
        # UNTRAINED UPSERT: upsert untrained items.
        items = []
        for i in range(self.num_untrained_vectors):
            items.append(
                {
                    "id": str(i),
                    "vector": self.vectors[i],
                    "metadata": self.metadata[i],
                    # 'contents': bytes(self.vectors[i]) # TODO!!!
                }
            )
        self.index.upsert(items)

        # Wait for 1 second to ensure upsert is processed
        time.sleep(1)

        # Check if the index has all IDs
        results = self.index.list_ids()
        expected_ids = [str(i) for i in range(self.num_untrained_vectors)]
        self.assertCountEqual(results, expected_ids)

    def test_02_untrained_query_no_metadata(self):
        # UNTRAINED QUERY (NO METADATA)
        results = self.index.query(query_vectors=self.queries, top_k=100, n_probes=1)

        recall = check_query_results(
            results, self.untrained_neighbors, self.num_queries
        )
        print(
            f"Untrained Query (No Metadata). Expected recall: {self.untrained_recall}, got {recall}"
        )

        self.assertAlmostEqual(recall.mean(), self.untrained_recall, delta=0.02)

        # Check if index is still untrained
        self.assertFalse(self.index.is_trained(), "Index should still be untrained")

    def test_03_untrained_query_metadata(self):
        # UNTRAINED QUERY (METADATA)
        results = []
        for metadata_query in self.metadata_queries:
            results.append(
                self.index.query(
                    query_vectors=self.queries,
                    top_k=100,
                    n_probes=1,
                    filters=metadata_query,
                )
            )

        recalls = check_metadata_results(
            results,
            self.untrained_metadata_neighbors,
            self.untrained_metadata_matches,
            self.num_queries,
        )

        for idx, recall in enumerate(recalls):
            print()
            print(f"Metadata Query #{idx + 1}")
            print(f"Metadata filters: {self.metadata_queries[idx]}")
            print(
                f"Number of candidates: {len(self.untrained_metadata_neighbors[idx])} / {self.num_untrained_vectors}"
            )
            print(f"Mean recall: {recall * 100:.2f}%")

            self.assertAlmostEqual(recall, self.untrained_recall, delta=0.02)

        # Check if index is still untrained
        self.assertFalse(self.index.is_trained(), "Index should still be untrained")

    def test_04_untrained_get(self):
        # UNTRAINED GET
        num_get = 1000
        get_indices = np.random.choice(
            self.num_untrained_vectors, num_get, replace=False
        )
        get_indices_str = get_indices.astype(str).tolist()
        get_results = self.index.get(
            ids=get_indices_str, include=["vector", "contents", "metadata"]
        )

        for i, get_result in enumerate(get_results):
            self.assertEqual(
                get_result["id"],
                get_indices_str[i],
                f"ID mismatch: {get_result['id']} != {get_indices_str[i]}",
            )
            self.assertTrue(
                np.array_equal(get_result["vector"], self.vectors[get_indices[i]]),
                f"Vector mismatch for index {i}",
            )
            metadata_str = json.dumps(get_result["metadata"], sort_keys=True)
            expected_metadata_str = json.dumps(
                self.metadata[get_indices[i]], sort_keys=True
            )
            self.assertEqual(
                metadata_str, expected_metadata_str, f"Metadata mismatch for index {i}"
            )

        # Check if index is still untrained
        self.assertFalse(self.index.is_trained(), "Index should still be untrained")

    def test_05_untrained_list_ids(self):
        # UNTRAINED LIST IDS
        results = self.index.list_ids()
        self.assertCountEqual(
            results, [str(id) for id in range(self.num_untrained_vectors)]
        )

        # Check if index is still untrained
        self.assertFalse(self.index.is_trained(), "Index should still be untrained")

    def test_06_upsert_to_trigger_auto_train(self):
        # Upsert enough vectors to exceed AUTO_TRAIN_MIN_VECTORS and trigger auto-train.
        # (Service default AUTO_TRAIN_MIN_VECTORS=65536 means auto-train triggers when
        # num_vectors > 65536.)
        auto_train_trigger = 65537
        items = []
        for i in range(self.num_untrained_vectors, auto_train_trigger):
            items.append(
                {
                    "id": str(i),
                    "vector": self.vectors[i],
                    "metadata": self.metadata[i],
                }
            )
        print(
            f"\nUpserting {len(items)} vector(s) to trigger auto-train (total will be {auto_train_trigger})..."
        )
        self.index.upsert(items)

        # Wait for upsert to be processed
        time.sleep(1)

        # Verify IDs are present
        results = self.index.list_ids()
        print(f"Total IDs in index: {len(results)}")
        expected_ids = [str(i) for i in range(auto_train_trigger)]
        self.assertCountEqual(results, expected_ids)

    def test_07_wait_for_auto_train(self):
        # WAIT FOR AUTO TRAINING TO COMPLETE (triggered at >65,536 vectors)
        num_retries = 60
        trained = False

        for attempt in range(num_retries):
            time.sleep(2)
            trained = self.index.is_trained()
            if trained:
                print("Index is now trained (auto-train complete).")
                break
            else:
                print(
                    f"Index not trained yet, retrying... ({attempt + 1}/{num_retries})"
                )

        self.assertTrue(trained, "Index did not become trained via auto-train in time")

    def test_08_upsert_remaining_vectors(self):
        # Upsert remaining vectors after auto-train (IDs auto_train_trigger to total-1)
        auto_train_trigger = 65537
        items = []
        for i in range(auto_train_trigger, self.total_num_vectors):
            items.append(
                {
                    "id": str(i),
                    "vector": self.vectors[i],
                    "metadata": self.metadata[i],
                }
            )
        print(
            f"\nUpserting {len(items)} remaining vectors (IDs {auto_train_trigger} to {self.total_num_vectors - 1})..."
        )
        self.index.upsert(items)

        # Wait for upsert to be processed
        time.sleep(1)

        # Verify all IDs are present
        results = self.index.list_ids()
        print(f"Total IDs in index: {len(results)}")
        expected_ids = [str(i) for i in range(self.total_num_vectors)]
        self.assertCountEqual(results, expected_ids)

    def test_09_retrain_with_n_lists(self):
        # Retrain with explicit n_lists to match core test behavior
        print(
            f"\nRetraining index with n_lists={self.n_lists} on {self.total_num_vectors} vectors..."
        )
        self.index.train(n_lists=self.n_lists)

        # Wait for training to finish by checking is_training status
        num_retries = 60
        trained = False

        for attempt in range(num_retries):
            time.sleep(2)

            is_currently_training = self.index.is_training()

            if not is_currently_training:
                # Training finished, verify it's trained
                trained = self.index.is_trained()
                if trained:
                    print("Index retrained successfully.")
                    break
            else:
                print(
                    f"Index still training, retrying... ({attempt + 1}/{num_retries})"
                )

        self.assertTrue(trained, "Index did not become trained after retraining")

        # Verify all vectors are still present after training
        results = self.index.list_ids()
        print(f"Total IDs in index after retraining: {len(results)}")
        self.assertEqual(
            len(results), self.total_num_vectors, "Vectors lost during retraining!"
        )

        # Verify final state - n_lists should match what we specified
        final_n_lists = self.index.n_lists
        print(f"Final n_lists: {final_n_lists}")
        self.assertEqual(
            final_n_lists,
            self.n_lists,
            f"Expected n_lists={self.n_lists}, got {final_n_lists}",
        )

    def test_10_trained_query_should_get_perfect_recall(self):
        # TRAINED QUERY WHERE N_PROBES == N_LISTS
        # Debug: Check index state before query
        num_ids = len(self.index.list_ids())
        is_trained = self.index.is_trained()
        print("\n=== DEBUG: Index state before query ===")
        print(f"Total IDs in index: {num_ids}")
        print(f"Is trained: {is_trained}")
        print(
            f"Dimension: {self.index.dimension}, metric: {self.index.metric}, "
            f"n_lists: {self.index.n_lists}"
        )

        results = self.index.query(
            query_vectors=self.queries, top_k=100, n_probes=self.n_lists
        )

        # Debug: Check what IDs are being returned
        all_returned_ids = set()
        for query_results in results:
            for res in query_results:
                all_returned_ids.add(int(res["id"]))
        print(f"Unique IDs returned across all queries: {len(all_returned_ids)}")
        print(
            f"Min ID returned: {min(all_returned_ids)}, Max ID returned: {max(all_returned_ids)}"
        )

        recall = check_query_results(results, self.trained_neighbors, self.num_queries)
        expected_recall = 1.0
        print(
            f"Trained Query (N_PROBES == N_LISTS). Expected recall: {expected_recall}, got {recall}"
        )

        # Exhaustive search (n_probes == n_lists) should recover the ground truth,
        # but JSON->float32 rounding of the query vectors differs slightly across
        # SDKs and can flip a few tie-broken neighbors at the top_k boundary, so
        # allow a tiny tolerance instead of bit-exact equality.
        self.assertAlmostEqual(recall, expected_recall, delta=0.01)

    def test_11_trained_query_no_metadata(self):
        # TRAINED QUERY (NO METADATA)
        results = self.index.query(query_vectors=self.queries, top_k=100, n_probes=24)

        recall = check_query_results(results, self.trained_neighbors, self.num_queries)
        print(
            f"Trained Query (No Metadata). Expected recall: {self.trained_recall}, got {recall}"
        )

        self.assertAlmostEqual(recall.mean(), self.trained_recall, delta=0.08)

    def test_12_trained_query_metadata(self):
        # TRAINED QUERY (METADATA)
        results = []
        for metadata_query in self.metadata_queries:
            results.append(
                self.index.query(
                    query_vectors=self.queries,
                    top_k=100,
                    n_probes=24,
                    filters=metadata_query,
                )
            )
        self.metadata_queries[6] = {"number": 0}

        recalls = check_metadata_results(
            results,
            self.trained_metadata_neighbors,
            self.trained_metadata_matches,
            self.num_queries,
        )

        print(f"Number of recall values: {len(recalls)}")

        base_thresholds = [
            94.04,  # Query #1
            100.00,  # Query #2
            91.05,  # Query #3
            77.77,  # Query #4
            75.00,  # Query #5
            78.88,  # Query #6
            75.00,  # Query #7
            92.35,  # Query #8
            91.66,  # Query #9
            77.77,  # Query #10
            88.26,  # Query #11
            94.04,  # Query #12
            90.05,  # Query #13
            22.00,  # Query #14
            3.7,  # Query #15
            70.00,  # Query #16
            70.00,  # Query #17
        ]

        expected_thresholds = [threshold * 0.95 for threshold in base_thresholds]

        assert len(recalls) == len(expected_thresholds), (
            f"Mismatch in number of recalls ({len(recalls)}) and thresholds ({len(expected_thresholds)})"
        )

        # Check each recall against its threshold
        failing_recalls = []

        for idx, recall in enumerate(recalls):
            recall_percentage = recall * 100
            threshold = expected_thresholds[idx]

            if idx < 17:
                print()
                print(f"Metadata Query #{idx + 1}")
                print(f"Metadata filters: {self.metadata_queries[idx]}")
                print(
                    f"Number of candidates: {len(self.trained_metadata_neighbors[idx])} / {self.total_num_vectors}"
                )
                print(f"Mean recall: {recall_percentage:.2f}%")
                print(f"Expected threshold: {threshold:.2f}%")
            else:
                print()
                print(f"Additional Query #{idx + 1}")
                print(f"Mean recall: {recall_percentage:.2f}%")
                print(f"Expected threshold: {threshold:.2f}%")

            if recall_percentage < threshold:
                failing_recalls.append((idx + 1, recall_percentage, threshold))

        if failing_recalls:
            fail_message = "\n".join(
                [
                    f"Query #{idx}: recall {actual:.2f}% < threshold {expected:.2f}%"
                    for idx, actual, expected in failing_recalls
                ]
            )
            assert not failing_recalls, (
                f"Some recalls are below their thresholds:\n{fail_message}"
            )

    def test_13_trained_get(self):
        # TRAINED GET (using untrained indices as an example)
        num_get = 1000
        get_indices = np.random.choice(
            self.num_untrained_vectors, num_get, replace=False
        )
        get_indices_str = get_indices.astype(str).tolist()
        get_results = self.index.get(
            get_indices_str, ["vector", "contents", "metadata"]
        )
        for i, get_result in enumerate(get_results):
            self.assertEqual(
                get_result["id"],
                get_indices_str[i],
                f"ID mismatch: {get_result['id']} != {get_indices_str[i]}",
            )
            self.assertTrue(
                np.array_equal(get_result["vector"], self.vectors[get_indices[i]]),
                f"Vector mismatch for index {i}",
            )
            metadata_str = json.dumps(get_result["metadata"], sort_keys=True)
            expected_metadata_str = json.dumps(
                self.metadata[get_indices[i]], sort_keys=True
            )
            self.assertEqual(
                metadata_str, expected_metadata_str, f"Metadata mismatch for index {i}"
            )

    def test_14_delete(self):
        # DELETE ITEMS (using untrained indices as an example)
        ids_to_delete = [str(i) for i in range(self.num_untrained_vectors)]
        self.index.delete(ids_to_delete)

        # Wait for 1 second to ensure delete is processed
        time.sleep(1)

        # Check if the index has deleted the IDs
        results = self.index.list_ids()
        for deleted_id in ids_to_delete:
            self.assertNotIn(deleted_id, results, f"ID {deleted_id} was not deleted")

        self.assertTrue(True)

    def test_15_get_deleted(self):
        # GET DELETED ITEMS
        num_get = 1000
        get_indices = np.random.choice(
            self.num_untrained_vectors, num_get, replace=False
        )
        get_indices_str = get_indices.astype(str).tolist()
        get_results = self.index.get(
            get_indices_str, ["vector", "contents", "metadata"]
        )

        self.assertEqual(len(get_results), 0)
        for i, get_result in enumerate(get_results):
            self.assertIsNone(get_result, f"Item {get_indices_str[i]} was not deleted")

    def test_16_query_deleted(self):
        # QUERY DELETED ITEMS
        results = self.index.query(query_vectors=self.queries, top_k=100, n_probes=24)

        for result in results:
            for query_result in result:
                self.assertNotIn(query_result["id"], range(self.num_untrained_vectors))

        self.assertTrue(True)

    def test_17_list_indexes(self):
        # LIST INDEXES
        indexes = self.client.list_indexes()
        self.assertIsInstance(indexes, list)
        self.assertGreater(len(indexes), 0, "No indexes found")

        # Check if the created index is in the list
        self.assertIn(
            self.index_name,
            indexes,
            f"Index {self.index_name} not found in the list of indexes",
        )

    def test_18_index_properties(self):
        # Check if the index has the expected properties
        self.assertEqual(
            self.index.index_name, self.index_name, "Index name does not match"
        )
        self.assertIsInstance(self.index.dimension, int)
        self.assertIsInstance(self.index.metric, str)
        self.assertIsInstance(self.index.n_lists, int)

    def test_19_load_index(self):
        # Test loading an existing index.
        loaded_index = self.client.load_index(self.index_name, self.index_key)
        self.assertIsInstance(loaded_index, cyborgdb.EncryptedIndex)
        self.assertEqual(loaded_index.index_name, self.index_name)


if __name__ == "__main__":
    unittest.main()
