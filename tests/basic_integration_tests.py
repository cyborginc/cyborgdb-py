#!/usr/bin/env python3
import unittest
import pickle
import json
import numpy as np
import time
import os
import cyborgdb as cyborgdb
from compress_utils import decompress
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

def check_query_results(results, neighbors, num_queries):

    # Parse results to extract IDs from the returned dictionaries
    result_ids = [
        [int(res["id"]) for res in query_results] for query_results in results
    ]

    # Convert the results to numpy array
    result_ids = np.array(result_ids)
    if neighbors.shape != result_ids.shape:
        raise ValueError(f"The shapes of the neighbors and results do not match: {neighbors.shape} != {result_ids.shape}")

    # Compute the recall using the neighbors
    recall = np.zeros(num_queries)
    for i in range(num_queries):
        recall[i] = len(np.intersect1d(neighbors[i], result_ids[i])) / len(neighbors[i])

    return recall.mean()


def check_metadata_results(results, metadata_neighors, metadata_candidates, num_queries):
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
            
            groundtruth_ids = np.array([
                candidates[int(idx)] 
                for idx in groundtruth_indices 
                if idx != -1 and 0 <= idx < len(candidates)
            ])
            
            # Get the returned neighbors for this query
            returned = np.array(result[i])

            # Update the number of returned neighbors
            num_returned += len(returned)
            local_expected = np.count_nonzero(groundtruth_ids != -1) 
            num_expected += local_expected

            # If we expect no results and got no results, recall is 100%
            if len(returned) == 0 and local_expected== 0:
                recall[i] = 1
                continue

            # Check if the number of returned neighbors is correct
            if len(returned) > 100:
                raise ValueError(f"More than 100 results returned: got {len(returned)} instead of 100")
            
            # Compute the recall for this query 
            recall[i] = len(np.intersect1d(groundtruth_ids, returned)) / min(local_expected, 100)

        # Get the number of groundtruth results (non -1 values)
        num_expected = num_expected / num_queries
        num_returned = num_returned / num_queries
        recalls.append(recall.mean())

    return recalls

class TestUnitFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):      
        # Construct the path to the pickle file
        import os
        test_dir = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(test_dir, "constants", "unit_test_flow_data.pkl.zst")
        
        # Load all the test data from the pickle file.
        with open(pickle_path, "rb") as f:
            compressed_data = f.read()
            decompressed_data = decompress(compressed_data, 'zstd')
            data = pickle.loads(decompressed_data)
            
        cls.vectors = data["vectors"]
        cls.untrained_vectors = data["untrained_vectors"]
        cls.training_vectors = data["training_vectors"]
        cls.queries = data["queries"]
        cls.untrained_neighbors = data["untrained_neighbors"]
        cls.trained_neighbors = data["trained_neighbors"]
        cls.metadata = data["metadata"]
        cls.metadata_queries = data["metadata_queries"]
        cls.metadata_query_names = data["metadata_query_names"]
        cls.untrained_metadata_matches = data["untrained_metadata_matches"]
        cls.trained_metadata_matches = data["trained_metadata_matches"]
        cls.untrained_metadata_neighbors = data["untrained_metadata_neighbors"]
        cls.trained_metadata_neighbors = data["trained_metadata_neighbors"]

        # Load expected recall values (as scalars or lists)
        cls.untrained_recall = data.get("untrained_recall")
        cls.trained_recall = data.get("trained_recall")

        # Set counts and dimension.
        cls.num_untrained_vectors = cls.untrained_vectors.shape[0]
        cls.total_num_vectors = cls.untrained_vectors.shape[0] + cls.training_vectors.shape[0]
        cls.num_queries = cls.queries.shape[0] if isinstance(cls.queries, np.ndarray) else len(cls.queries)
        cls.dimension = cls.vectors.shape[1] if isinstance(cls.vectors, np.ndarray) else len(cls.vectors[0])
        
        # CYBORDB SETUP: Create the index once (shared state).
        cls.index_config = cyborgdb.IndexIVFFlat(dimension=cls.dimension, n_lists=512, metric="euclidean")
        cls.client = cyborgdb.Client(
            api_url="https://localhost:8000",
            api_key=os.getenv("CYBORG_API_KEY")
        )
        cls.index_name = "memory_example_index19"
        cls.index_key = cyborgdb. generate_key() #bytes([1] * 32)
        cls.index = cls.client.create_index(cls.index_name, cls.index_key, cls.index_config)

    @classmethod
    def tearDownClass(cls):
        """Clean up the index after all tests are done."""
        try:
            if hasattr(cls, 'index') and cls.index:
                cls.index.delete_index()
        except Exception as e:
            print(f"Error during index cleanup: {e}")

    def test_01_untrained_upsert(self):
        # UNTRAINED UPSERT: upsert untrained items.
        items = []
        for i in range(self.num_untrained_vectors):
            items.append({
                'id': str(i),
                'vector': self.vectors[i],
                'metadata': self.metadata[i],
                # 'contents': bytes(self.vectors[i]) # not supported by the service yet
            })
        self.index.upsert(items)
        self.assertTrue(True)

    def test_02_untrained_query_no_metadata(self):
        # UNTRAINED QUERY (NO METADATA)
        results = self.index.query(query_vectors=self.queries, top_k=100, n_probes=1)
        recall = check_query_results(results, self.untrained_neighbors, self.num_queries)

        self.assertAlmostEqual(recall.mean(), 
                               self.untrained_recall, 
                               delta=0.02)


    def test_03_untrained_query_metadata(self):
        # UNTRAINED QUERY (METADATA)
        results = [] 
        for metadata_query in self.metadata_queries:
            results.append(self.index.query(query_vectors=self.queries, top_k=100, n_probes=1, filters=metadata_query))

        recalls = check_metadata_results(results, self.untrained_metadata_neighbors, self.untrained_metadata_matches, self.num_queries)
        
        for idx, recall in enumerate(recalls):
            print()
            print(f"Metadata Query #{idx+1}")
            print(f"Metadata filters: {self.metadata_queries[idx]}")
            print(f"Number of candidates: {len(self.untrained_metadata_neighbors[idx])} / {self.num_untrained_vectors}")
            print(f"Mean recall: {recall * 100:.2f}%")
            
            self.assertAlmostEqual(recall, self.untrained_recall, delta=0.02)

    def test_04_untrained_get(self):
        # UNTRAINED GET
        num_get = 1000
        get_indices = np.random.choice(self.num_untrained_vectors, num_get, replace=False)
        get_indices_str = get_indices.astype(str).tolist()
        get_results = self.index.get(ids=get_indices_str, include=['vector', 'contents', 'metadata'])
        for i, get_result in enumerate(get_results):
            self.assertEqual(get_result["id"], get_indices_str[i],
                             f"ID mismatch: {get_result['id']} != {get_indices_str[i]}")
            self.assertTrue(np.array_equal(get_result["vector"], self.vectors[get_indices[i]]),
                            f"Vector mismatch for index {i}")
            metadata_str = json.dumps(get_result["metadata"], sort_keys=True)
            expected_metadata_str = json.dumps(self.metadata[get_indices[i]], sort_keys=True)
            self.assertEqual(metadata_str, expected_metadata_str,
                             f"Metadata mismatch for index {i}")

    def test_05_train_index(self):
        # TRAIN INDEX
        self.index.train()
        self.assertTrue(True)
        self.assertTrue(self.index.is_trained())

    def test_06_trained_upsert(self):
        # TRAINED UPSERT: upsert training vectors.
        items = []
        for i in range(self.num_untrained_vectors, self.total_num_vectors):
            items.append({
                'id': str(i),
                'vector': self.vectors[i],
                'metadata': self.metadata[i],
                # 'contents': bytes(self.vectors[i])
            })
        self.index.upsert(items)
        self.assertTrue(True)

    def test_07_trained_query_no_metadata(self):
        # TRAINED QUERY (NO METADATA)
        results = self.index.query(query_vectors=self.queries, top_k=100, n_probes=24)
        
        recall = check_query_results(results, self.trained_neighbors, self.num_queries)
        print(f"Trained Query (No Metadata). Expected recall: {self.trained_recall}, got {recall}")

        self.assertAlmostEqual(recall.mean(),
                               self.trained_recall,
                               delta=0.02)
        
    def test_08_trained_query_metadata(self):
        # TRAINED QUERY (METADATA)
        results = []
        for metadata_query in self.metadata_queries:
            results.append(self.index.query(query_vectors=self.queries, top_k=100, n_probes=24, filters=metadata_query))
        self.metadata_queries[6] = {'number': 0}

        recalls = check_metadata_results(results, self.trained_metadata_neighbors, self.trained_metadata_matches, self.num_queries)

        print(f"Number of recall values: {len(recalls)}")

        base_thresholds = [
            94.04,  # Query #1
            100.00, # Query #2
            91.05,  # Query #3
            88.24,  # Query #4
            100.00, # Query #5
            78.88,  # Query #6
            100.00, # Query #7
            92.35,  # Query #8
            91.66,  # Query #9
            88.38,  # Query #10
            88.26,  # Query #11
            94.04,  # Query #12
            90.05,  # Query #13
            74.09,  # Query #14
            9.00,  # Query #15 
        ]

        # For the additional 2 recalls, we'll use a default threshold of 70%
        for i in range(len(base_thresholds), len(recalls)):
            base_thresholds.append(70.00)

        expected_thresholds = [threshold * 0.95 for threshold in base_thresholds]

        assert len(recalls) == len(expected_thresholds), \
            f"Mismatch in number of recalls ({len(recalls)}) and thresholds ({len(expected_thresholds)})"

        # Check each recall against its threshold
        failing_recalls = []

        for idx, recall in enumerate(recalls):
            recall_percentage = recall * 100
            threshold = expected_thresholds[idx]

            if idx < 15:
                print()
                print(f"Metadata Query #{idx+1}")
                print(f"Metadata filters: {self.metadata_queries[idx]}")
                print(f"Number of candidates: {len(self.trained_metadata_neighbors[idx])} / {self.total_num_vectors}")
                print(f"Mean recall: {recall_percentage:.2f}%")
                print(f"Expected threshold: {threshold:.2f}%")
            else:
                print()
                print(f"Additional Query #{idx+1}")
                print(f"Mean recall: {recall_percentage:.2f}%")
                print(f"Expected threshold: {threshold:.2f}%")

            if recall_percentage < threshold:
                failing_recalls.append((idx+1, recall_percentage, threshold))

        if failing_recalls:
            fail_message = "\n".join([
                f"Query #{idx}: recall {actual:.2f}% < threshold {expected:.2f}%"
                for idx, actual, expected in failing_recalls
            ])
            assert not failing_recalls, f"Some recalls are below their thresholds:\n{fail_message}"

    def test_09_trained_get(self):
        # TRAINED GET (using untrained indices as an example)
        num_get = 1000
        get_indices = np.random.choice(self.num_untrained_vectors, num_get, replace=False)
        get_indices_str = get_indices.astype(str).tolist()
        get_results = self.index.get(get_indices_str, ['vector', 'contents', 'metadata'])
        for i, get_result in enumerate(get_results):
            self.assertEqual(get_result["id"], get_indices_str[i],
                             f"ID mismatch: {get_result['id']} != {get_indices_str[i]}")
            self.assertTrue(np.array_equal(get_result["vector"], self.vectors[get_indices[i]]),
                            f"Vector mismatch for index {i}")
            metadata_str = json.dumps(get_result["metadata"], sort_keys=True)
            expected_metadata_str = json.dumps(self.metadata[get_indices[i]], sort_keys=True)
            self.assertEqual(metadata_str, expected_metadata_str,
                             f"Metadata mismatch for index {i}")
            
    def test_10_delete(self):
        # DELETE ITEMS (using untrained indices as an example)
        ids_to_delete = [str(i) for i in range(self.num_untrained_vectors)]
        self.index.delete(ids_to_delete)
        self.assertTrue(True)

    def test_11_get_deleted(self):
        # GET DELETED ITEMS
        num_get = 1000
        get_indices = np.random.choice(self.num_untrained_vectors, num_get, replace=False)
        get_indices_str = get_indices.astype(str).tolist()
        get_results = self.index.get(get_indices_str, ['vector', 'contents', 'metadata'])
        self.assertEqual(len(get_results), 0)
        for i, get_result in enumerate(get_results):
            self.assertIsNone(get_result, f"Item {get_indices_str[i]} was not deleted")

    def test_12_query_deleted(self):
        # QUERY DELETED ITEMS
        results = self.index.query(query_vectors=self.queries, top_k=100, n_probes=24)

        for result in results:
            for query_result in result:
                self.assertNotIn(query_result["id"], range(self.num_untrained_vectors))
        
        self.assertTrue(True)
    
    def test_13_list_indexes(self):
        # LIST INDEXES
        indexes = self.client.list_indexes()
        self.assertIsInstance(indexes, list)
        self.assertGreater(len(indexes), 0, "No indexes found")
        
        # Check if the created index is in the list
        self.assertIn(self.index_name, indexes, f"Index {self.index_name} not found in the list of indexes")

    def test_14_index_properies(self):
        # Check if the index has the expected properties
        self.assertEqual(self.index.index_name, self.index_name, "Index name does not match")
        self.assertIsInstance(self.index.index_config, dict, "Index config is not a dictionary")
        self.assertEqual(self.index.index_type, "ivfflat", "Index type is not IVFFlat")

    def test_15_load_index(self):
        # Test loading an existing index.
        loaded_index = self.client.load_index(self.index_name, self.test_key)
        self.assertIsInstance(loaded_index, cyborgdb.EncryptedIndex)
        self.assertEqual(loaded_index.index_name, self.index_name)

if __name__ == '__main__':
    unittest.main()
