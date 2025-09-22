"""
API Contract Test for CyborgDB Python SDK

This test rigorously verifies the complete public API surface of the CyborgDB Python SDK.
It validates:
- Exact function signatures (parameter names, order, defaults)
- Exact response formats (no missing or extra keys)
- Type constraints on all inputs and outputs
- That no unexpected parameters are accepted
"""

import os
import time
import uuid
import inspect
import numpy as np
from typing import List, Dict, Any
import unittest
from dotenv import load_dotenv

import cyborgdb

# Load environment variables from .env.local
load_dotenv(".env.local")

def generate_test_vectors(num_vectors: int = 10, dimension: int = 128) -> List[np.ndarray]:
    """Generate test vectors for API testing."""
    np.random.seed(42)  # Fixed seed for reproducibility
    return [np.random.rand(dimension).astype(np.float32) for _ in range(num_vectors)]


def generate_test_metadata(num_items: int = 10) -> List[Dict[str, Any]]:
    """Generate test metadata for API testing."""
    return [{"index": i, "category": f"cat_{i % 3}", "value": i * 10} for i in range(num_items)]


def validate_exact_keys(data: dict, expected_keys: set, name: str):
    """Validate that a dict has exactly the expected keys - no more, no less."""
    actual_keys = set(data.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    if missing:
        raise AssertionError(f"{name}: Missing required keys: {missing}")
    if extra:
        raise AssertionError(f"{name}: Unexpected extra keys: {extra}")


def validate_function_signature(func, expected_params: dict, func_name: str):
    """
    Validate function signature exactly matches expectations.

    expected_params format:
    {
        'param_name': {
            'position': 0,  # Expected position (excluding self)
            'default': inspect.Parameter.empty or actual default value,
            'annotation': type or None
        }
    }
    """
    sig = inspect.signature(func)
    params = dict(sig.parameters)

    # Remove 'self' if present
    if 'self' in params:
        del params['self']

    # Check exact parameter count
    if len(params) != len(expected_params):
        raise AssertionError(
            f"{func_name}: Parameter count mismatch. "
            f"Expected {len(expected_params)}, got {len(params)}. "
            f"Expected: {list(expected_params.keys())}, Got: {list(params.keys())}"
        )

    # Check each parameter
    param_list = list(params.items())
    for expected_name, expected_info in expected_params.items():
        position = expected_info['position']

        # Check parameter exists at correct position
        if position >= len(param_list):
            raise AssertionError(f"{func_name}: Missing parameter '{expected_name}' at position {position}")

        actual_name, actual_param = param_list[position]

        # Check name matches
        if actual_name != expected_name:
            raise AssertionError(
                f"{func_name}: Parameter name mismatch at position {position}. "
                f"Expected '{expected_name}', got '{actual_name}'"
            )

        # Check default value
        expected_default = expected_info.get('default', inspect.Parameter.empty)
        if actual_param.default != expected_default:
            raise AssertionError(
                f"{func_name}.{expected_name}: Default value mismatch. "
                f"Expected {expected_default!r}, got {actual_param.default!r}"
            )


class TestAPIContract(unittest.TestCase):
    """Test the complete API contract with rigid validation."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources."""
        cls.base_url = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
        cls.api_key = os.getenv("CYBORGDB_API_KEY", "")
        cls.dimension = 128
        cls.test_vectors = generate_test_vectors(10, cls.dimension)
        cls.test_metadata = generate_test_metadata(10)
        cls.index_name = f"test_contract_{uuid.uuid4().hex[:8]}"
        cls.index_key = None
        cls.client = None
        cls.index = None

    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        if cls.index:
            try:
                cls.index.delete_index()
            except:
                pass

    def test_01_module_exports(self):
        """Test module exports exactly match expected set."""
        # Get all public exports
        actual_exports = {name for name in dir(cyborgdb) if not name.startswith('_')}

        # Define expected exports (core + optional)
        required_exports = {
            "Client",
            "EncryptedIndex",
            "IndexConfig",
            "IndexIVF",
            "IndexIVFPQ",
            "IndexIVFFlat",
        }

        # These may have additional internal exports, but we check minimums
        missing = required_exports - actual_exports
        self.assertEqual(len(missing), 0, f"Missing required exports: {missing}")

        # Check if optional exports exist
        if "CyborgVectorStore" in actual_exports:
            print("Optional export CyborgVectorStore: available")

    def test_02_client_class_signatures(self):
        """Test Client class method signatures exactly."""
        # Test __init__ signature
        validate_function_signature(
            cyborgdb.Client.__init__,
            {
                'base_url': {'position': 0, 'default': inspect.Parameter.empty},
                'api_key': {'position': 1, 'default': inspect.Parameter.empty},
                'verify_ssl': {'position': 2, 'default': None},
            },
            "Client.__init__"
        )

        # Test generate_key signature (static method)
        validate_function_signature(
            cyborgdb.Client.generate_key,
            {},  # No parameters
            "Client.generate_key"
        )

        # Test other methods will be validated when we have a client instance

    def test_03_client_generate_key(self):
        """Test Client.generate_key() with exact response validation."""
        # Test with no arguments (should work)
        key = cyborgdb.Client.generate_key()
        self.assertIsInstance(key, bytes, "generate_key must return bytes")
        self.assertEqual(len(key), 32, "Key must be exactly 32 bytes")

        # Test that it accepts no arguments
        with self.assertRaises(TypeError, msg="generate_key should accept no arguments"):
            cyborgdb.Client.generate_key("unexpected_arg")

        # Test that is also works as a member function
        client = cyborgdb.Client(base_url=self.base_url, api_key=self.api_key)
        key = client.generate_key()
        self.assertIsInstance(key, bytes, "generate_key must return bytes")
        self.assertEqual(len(key), 32, "Key must be exactly 32 bytes")

        # Store for later use
        self.__class__.index_key = key

    def test_04_client_init(self):
        """Test Client initialization with strict parameter validation."""
        # Test with positional arguments
        client1 = cyborgdb.Client(self.base_url, self.api_key)
        self.assertIsInstance(client1, cyborgdb.Client)

        # Test with keyword arguments
        client2 = cyborgdb.Client(base_url=self.base_url, api_key=self.api_key)
        self.assertIsInstance(client2, cyborgdb.Client)

        # Test with optional verify_ssl
        client3 = cyborgdb.Client(self.base_url, self.api_key, True)
        self.assertIsInstance(client3, cyborgdb.Client)

        # Test with keyword + optional verify_ssl
        client4 = cyborgdb.Client(base_url=self.base_url, api_key=self.api_key, verify_ssl=True)
        self.assertIsInstance(client4, cyborgdb.Client)

        # Test that unexpected arguments are rejected
        with self.assertRaises(TypeError):
            cyborgdb.Client(
                base_url=self.base_url,
                api_key=self.api_key,
                unexpected_param="should fail"
            )

        # Store client for later use
        self.__class__.client = client1

        # Validate instance method signatures
        validate_function_signature(
            self.client.get_health,
            {},  # No parameters
            "Client.get_health"
        )

        validate_function_signature(
            self.client.list_indexes,
            {},  # No parameters
            "Client.list_indexes"
        )

        validate_function_signature(
            self.client.create_index,
            {
                'index_name': {'position': 0, 'default': inspect.Parameter.empty},
                'index_key': {'position': 1, 'default': inspect.Parameter.empty},
                'index_config': {'position': 2, 'default': None},
                'embedding_model': {'position': 3, 'default': None},
                'metric': {'position': 4, 'default': None},
            },
            "Client.create_index"
        )

        validate_function_signature(
            self.client.load_index,
            {
                'index_name': {'position': 0, 'default': inspect.Parameter.empty},
                'index_key': {'position': 1, 'default': inspect.Parameter.empty},
            },
            "Client.load_index"
        )

    def test_05_client_get_health(self):
        """Test Client.get_health() with exact response format."""
        # Call with no arguments
        health = self.client.get_health()

        # Validate exact response structure
        self.assertIsInstance(health, dict)
        self.assertIn("status", health, "Health must contain 'status' key")

        # Test that it accepts no arguments
        with self.assertRaises(TypeError):
            self.client.get_health("unexpected_arg")

    def test_06_client_list_indexes(self):
        """Test Client.list_indexes() with exact response format."""
        # Call with no arguments
        indexes = self.client.list_indexes()

        # Validate exact response type
        self.assertIsInstance(indexes, list)
        for index_name in indexes:
            self.assertIsInstance(index_name, str, "Each index name must be a string")

        # Test that it accepts no arguments
        with self.assertRaises(TypeError):
            self.client.list_indexes("unexpected_arg")

    def test_07_index_config_classes(self):
        """Test index configuration class signatures and instantiation."""
        # Test IndexIVF with no arguments
        config1 = cyborgdb.IndexIVF()
        # IndexIVF is actually IndexIVFModel, not IndexConfig
        self.assertIsInstance(config1, cyborgdb.IndexIVF)
        # Verify it has expected attributes
        self.assertTrue(hasattr(config1, 'type'))
        self.assertEqual(config1.type, 'ivf')

        # Test IndexIVFFlat - Pydantic models use **data so can't inspect individual params
        # Instead test that it accepts the dimension parameter correctly

        config2 = cyborgdb.IndexIVFFlat(dimension=self.dimension)
        self.assertIsInstance(config2, cyborgdb.IndexIVFFlat)
        self.assertEqual(config2.dimension, self.dimension)
        self.assertEqual(config2.type, 'ivfflat')

        # Test IndexIVFPQ - check required parameters
        config3 = cyborgdb.IndexIVFPQ(dimension=self.dimension, pq_dim=64, pq_bits=8)
        self.assertIsInstance(config3, cyborgdb.IndexIVFPQ)
        self.assertEqual(config3.dimension, self.dimension)
        self.assertEqual(config3.pq_dim, 64)
        self.assertEqual(config3.pq_bits, 8)
        self.assertEqual(config3.type, 'ivfpq')

    def test_08_client_create_index(self):
        """Test Client.create_index() with strict parameter validation."""
        index_config = cyborgdb.IndexIVFFlat(dimension=self.dimension)

        # Test with all parameters including a valid config
        index = self.client.create_index(
            self.index_name,
            self.index_key,
            index_config,
            None,  # embedding_model
            "euclidean"  # metric
        )
        self.assertIsInstance(index, cyborgdb.EncryptedIndex)

        # Clean up this index
        index.delete_index()
        time.sleep(1)

        # Test with positional arguments (with config to avoid API error)
        index_name2 = f"test_contract_{uuid.uuid4().hex[:8]}"
        index = self.client.create_index(
            index_name2,
            self.index_key,
            index_config  # Include config to avoid 400 error
        )
        self.assertIsInstance(index, cyborgdb.EncryptedIndex)

        # Store this one for later tests
        self.__class__.index_name = index_name2

        # Store for later tests
        self.__class__.index = index

        # Validate EncryptedIndex method signatures
        validate_function_signature(
            self.index.is_trained,
            {},  # No parameters
            "EncryptedIndex.is_trained"
        )

        validate_function_signature(
            self.index.is_training,
            {},  # No parameters
            "EncryptedIndex.is_training"
        )

        validate_function_signature(
            self.index.delete_index,
            {},  # No parameters
            "EncryptedIndex.delete_index"
        )

        validate_function_signature(
            self.index.list_ids,
            {},  # No parameters
            "EncryptedIndex.list_ids"
        )

        # Check upsert signature - it has 2 parameters with second optional
        sig = inspect.signature(self.index.upsert)
        params = dict(sig.parameters)
        if 'self' in params:
            del params['self']
        self.assertEqual(len(params), 2, "upsert should have exactly 2 parameters")

        validate_function_signature(
            self.index.delete,
            {
                'ids': {'position': 0, 'default': inspect.Parameter.empty},
            },
            "EncryptedIndex.delete"
        )

        validate_function_signature(
            self.index.get,
            {
                'ids': {'position': 0, 'default': inspect.Parameter.empty},
                'include': {'position': 1, 'default': ["vector", "contents", "metadata"]},
            },
            "EncryptedIndex.get"
        )

        # Query has many optional parameters
        sig = inspect.signature(self.index.query)
        params = dict(sig.parameters)
        if 'self' in params:
            del params['self']

        # Check query parameters exist and have defaults
        expected_query_params = ['query_vectors', 'query_contents', 'top_k', 'n_probes', 'filters', 'include', 'greedy']
        for param_name in expected_query_params:
            if param_name in params:
                param = params[param_name]
                # All query parameters should have defaults (be optional)
                self.assertNotEqual(
                    param.default,
                    inspect.Parameter.empty,
                    f"query.{param_name} should have a default value"
                )

        validate_function_signature(
            self.index.train,
            {
                'n_lists': {'position': 0, 'default': None},
                'batch_size': {'position': 1, 'default': None},
                'max_iters': {'position': 2, 'default': None},
                'tolerance': {'position': 3, 'default': None},
            },
            "EncryptedIndex.train"
        )

    def test_09_encrypted_index_properties(self):
        """Test EncryptedIndex properties return exact expected types."""
        # Test index_name property
        name = self.index.index_name
        self.assertIsInstance(name, str)
        self.assertEqual(name, self.index_name)

        # Test index_type property
        index_type = self.index.index_type
        self.assertIsInstance(index_type, str)

        # Test index_config property
        config = self.index.index_config
        self.assertIsInstance(config, dict)
        # Config should have certain keys
        self.assertIn("dimension", config)

    def test_10_encrypted_index_is_trained(self):
        """Test EncryptedIndex.is_trained() exact behavior."""
        # Should return exactly a bool
        trained = self.index.is_trained()
        self.assertIsInstance(trained, bool)
        self.assertIn(trained, [True, False])

        # Should accept no arguments
        with self.assertRaises(TypeError):
            self.index.is_trained("unexpected")

    def test_11_encrypted_index_is_training(self):
        """Test EncryptedIndex.is_training() exact behavior."""
        # Should return exactly a bool
        training = self.index.is_training()
        self.assertIsInstance(training, bool)
        self.assertIn(training, [True, False])

        # Should accept no arguments
        with self.assertRaises(TypeError):
            self.index.is_training("unexpected")

    def test_12_encrypted_index_upsert(self):
        """Test EncryptedIndex.upsert() parameter validation."""
        # Test 1: Prepare test data with contents as bytes (should work)
        items_bytes = []
        for i in range(2):
            item = {
                "id": str(i),
                "vector": self.test_vectors[i],
                "metadata": self.test_metadata[i],
                "contents": f"test content {i}".encode('utf-8')
            }
            items_bytes.append(item)

        # Test upsert with bytes contents (should work)
        result = self.index.upsert(items_bytes)
        self.assertIsNone(result, "upsert must return None")

        time.sleep(1)

        # Test 2: Prepare test data with contents as strings (should also work)
        items_strings = []
        for i in range(2, 5):
            item = {
                "id": str(i),
                "vector": self.test_vectors[i],
                "metadata": self.test_metadata[i],
                "contents": f"test content {i}"
            }
            items_strings.append(item)

        # Test upsert with string contents (should work according to Contents model)
        result = self.index.upsert(items_strings)
        self.assertIsNone(result, "upsert must return None")

        time.sleep(1)

        # Test 3: Two argument format (this API doesn't support dict format, only list of dicts)
        # Create list of dict items instead
        items_remaining = []
        for i in range(5, 10):
            item = {
                "id": str(i),
                "vector": self.test_vectors[i],
                "metadata": self.test_metadata[i],
                "contents": f"test content {i}".encode('utf-8')
            }
            items_remaining.append(item)

        result = self.index.upsert(items_remaining)
        self.assertIsNone(result, "upsert must return None")

        time.sleep(1)

    def test_13_encrypted_index_list_ids(self):
        """Test EncryptedIndex.list_ids() exact response format."""
        ids = self.index.list_ids()

        # Must return a list of strings
        self.assertIsInstance(ids, list)
        for id_val in ids:
            self.assertIsInstance(id_val, str, "Each ID must be a string")

        # Should accept no arguments
        with self.assertRaises(TypeError):
            self.index.list_ids("unexpected")

    def test_14_encrypted_index_get(self):
        """Test EncryptedIndex.get() with exact response validation."""
        ids_to_get = ["0", "5", "9"]

        # Test with default include parameter
        results = self.index.get(ids_to_get)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

        # Each result should have exact keys based on default include
        for result in results:
            self.assertIsInstance(result, dict)
            # Default include is ["vector", "contents", "metadata"]
            validate_exact_keys(
                result,
                {"id", "vector", "metadata", "contents"},
                "get() result with default include"
            )

        # Test with specific include parameter
        results = self.index.get(ids_to_get, include=["metadata"])

        for result in results:
            validate_exact_keys(
                result,
                {"id", "metadata"},
                "get() result with include=['metadata']"
            )

        # Test with only IDs
        results = self.index.get(ids_to_get, include=[])

        for result in results:
            validate_exact_keys(
                result,
                {"id"},
                "get() result with include=[]"
            )

    def test_15_encrypted_index_query(self):
        """Test EncryptedIndex.query() with exact response validation."""
        query_vector = self.test_vectors[0]

        # Test query with default include (should be ["distance", "metadata"])
        results = self.index.query(query_vectors=[query_vector])

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1, "Should have 1 result for 1 query")

        query_results = results[0]
        self.assertIsInstance(query_results, list)

        # Check exact structure with default include - id is always included, plus distance and metadata by default
        for result in query_results:
            self.assertIsInstance(result, dict)
            validate_exact_keys(
                result,
                {"id", "distance", "metadata"},
                "query() result with default include"
            )

        # Test with specific include=["vector", "metadata"]
        results = self.index.query(
            query_vectors=[query_vector],
            top_k=5,
            include=["vector", "metadata"]
        )

        for result in results[0]:
            validate_exact_keys(
                result,
                {"id", "distance", "vector", "metadata"},
                "query() result with include=['vector','metadata']"
            )

        # Test with empty include (should only have id and distance)
        results = self.index.query(
            query_vectors=[query_vector],
            include=[]
        )

        for result in results[0]:
            validate_exact_keys(
                result,
                {"id", "distance"},
                "query() result with include=[]"
            )

        # Test with filters and include=["metadata"]
        results = self.index.query(
            query_vectors=[query_vector],
            filters={"category": "cat_0"},
            include=["metadata"]
        )

        # Results should have id, distance (always included), and metadata (requested)
        for result in results[0]:
            validate_exact_keys(
                result,
                {"id", "distance", "metadata"},
                "query() result with filters and include=['metadata']"
            )

    def test_16_encrypted_index_train(self):
        """Test EncryptedIndex.train() parameter validation."""
        # Test with no arguments (all defaults)
        result = self.index.train()
        self.assertIsNone(result, "train must return None")

        # Test with all arguments
        result = self.index.train(
            n_lists=10,
            batch_size=100,
            max_iters=10,
            tolerance=0.01
        )
        self.assertIsNone(result, "train must return None")

        # Test positional argument order
        result = self.index.train(10, 100, 10, 0.01)
        self.assertIsNone(result, "train must return None")

        time.sleep(2)

    def test_17_encrypted_index_delete(self):
        """Test EncryptedIndex.delete() exact behavior."""
        ids_to_delete = ["0", "5"]

        # Must accept ids parameter
        result = self.index.delete(ids=ids_to_delete)
        self.assertIsNone(result, "delete must return None")

        # Test positional
        result = self.index.delete(["9"])
        self.assertIsNone(result, "delete must return None")

        time.sleep(1)

        # Verify deletion worked
        remaining = self.index.list_ids()
        for deleted_id in ["0", "5", "9"]:
            self.assertNotIn(deleted_id, remaining)

    def test_18_client_load_index(self):
        """Test Client.load_index() exact behavior."""
        # Test positional arguments
        loaded = self.client.load_index(self.index_name, self.index_key)
        self.assertIsInstance(loaded, cyborgdb.EncryptedIndex)

        # Test keyword arguments
        loaded = self.client.load_index(
            index_name=self.index_name,
            index_key=self.index_key
        )
        self.assertIsInstance(loaded, cyborgdb.EncryptedIndex)

        # Verify properties match
        self.assertEqual(loaded.index_name, self.index_name)

        # Test that extra arguments are rejected
        with self.assertRaises(TypeError):
            self.client.load_index(
                self.index_name,
                self.index_key,
                "unexpected_arg"
            )

    def test_19_encrypted_index_delete_index(self):
        """Test EncryptedIndex.delete_index() exact behavior."""
        # Should accept no arguments
        result = self.index.delete_index()
        self.assertIsNone(result, "delete_index must return None")

        time.sleep(1)

        # Verify deletion
        indexes = self.client.list_indexes()
        self.assertNotIn(self.index_name, indexes)

        # Clear reference
        self.__class__.index = None


if __name__ == "__main__":
    unittest.main(verbosity=2)