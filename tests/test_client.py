"""
Unit tests for the CyborgDB REST API client wrapper.
"""

import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the REST client
from cyborgdb.client.rest_client import (
    Client, 
    EncryptedIndex,
    DBConfig, 
    DBLocation,
    IndexIVF, 
    IndexIVFPQ,
    IndexIVFFlat,
    generate_key
)

# Test constants
API_URL = os.getenv("CYBORGDB_API_URL", "http://0.0.0.0:8000")
REDIS_CONNECTION_STRING = os.getenv("REDIS_CONNECTION_STRING", "redis://localhost:6379")


class MockApiResponse:
    """Mock API response for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestRestClient(unittest.TestCase):
    """Test the REST API client."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create patches for API client
        self.api_patcher = patch('cyborgdb.client.rest_client.DefaultApi')
        self.api_client_patcher = patch('cyborgdb.client.rest_client.ApiClient')
        self.config_patcher = patch('cyborgdb.client.rest_client.Configuration')
        
        # Start patches
        self.mock_api = self.api_patcher.start()
        self.mock_api_client = self.api_client_patcher.start()
        self.mock_config = self.config_patcher.start()
        
        # Setup mock API instance
        self.api_instance = MagicMock()
        self.mock_api.return_value = self.api_instance
        
        # Setup basic client
        self.client = Client(
            api_url=API_URL,
            index_location=DBConfig(DBLocation.REDIS, connection_string=REDIS_CONNECTION_STRING),
            config_location=DBConfig(DBLocation.REDIS, connection_string=REDIS_CONNECTION_STRING)
        )
        
        # Create a test key
        self.test_key = generate_key()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.api_patcher.stop()
        self.api_client_patcher.stop()
        self.config_patcher.stop()
    
    def test_client_initialization(self):
        """Test client initialization."""
        # The client was initialized in setUp, so we just check it exists
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.config.host, API_URL)
    
    def test_list_indexes(self):
        """Test listing indexes."""
        # Setup mock response
        self.api_instance.list_indexes.return_value = MockApiResponse(
            indexes=["index1", "index2"]
        )
        
        # Call the method
        indexes = self.client.list_indexes()
        
        # Verify results
        self.api_instance.list_indexes.assert_called_once()
        self.assertEqual(indexes, ["index1", "index2"])
    
    def test_create_index(self):
        """Test creating an index."""
        # Setup test data
        index_name = "test_index"
        index_config = IndexIVF(dimension=128, n_lists=64)
        
        # Setup mock response
        self.api_instance.create_index.return_value = None  # Successful creation
        
        # Call the method
        index = self.client.create_index(index_name, self.test_key, index_config)
        
        # Verify results
        self.api_instance.create_index.assert_called_once()
        self.assertIsInstance(index, EncryptedIndex)
        self.assertEqual(index.index_name, index_name)
    
    def test_load_index(self):
        """Test loading an index."""
        # Setup test data
        index_name = "existing_index"
        
        # Setup mock responses
        self.api_instance.list_indexes.return_value = MockApiResponse(
            indexes=[index_name, "other_index"]
        )
        
        # Call the method
        index = self.client.load_index(index_name, self.test_key)
        
        # Verify results
        self.api_instance.list_indexes.assert_called_once()
        self.assertIsInstance(index, EncryptedIndex)
        self.assertEqual(index.index_name, index_name)
    
    def test_load_nonexistent_index(self):
        """Test loading a nonexistent index."""
        # Setup test data
        index_name = "nonexistent_index"
        
        # Setup mock responses
        self.api_instance.list_indexes.return_value = MockApiResponse(
            indexes=["other_index"]
        )
        
        # Call the method and verify exception
        with self.assertRaises(ValueError):
            self.client.load_index(index_name, self.test_key)


class TestEncryptedIndex(unittest.TestCase):
    """Test the EncryptedIndex class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create patches
        self.api_patcher = patch('cyborgdb.client.rest_client.DefaultApi')
        self.api_client_patcher = patch('cyborgdb.client.rest_client.ApiClient')
        
        # Start patches
        self.mock_api_class = self.api_patcher.start()
        self.mock_api_client_class = self.api_client_patcher.start()
        
        # Create mock instances
        self.mock_api = MagicMock()
        self.mock_api_client = MagicMock()
        
        self.mock_api_class.return_value = self.mock_api
        self.mock_api_client_class.return_value = self.mock_api_client
        
        # Create test objects
        self.index_name = "test_index"
        self.index_key = generate_key()
        
        # Create the index
        self.index = EncryptedIndex(
            index_name=self.index_name,
            index_key=self.index_key,
            api=self.mock_api,
            api_client=self.mock_api_client
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.api_patcher.stop()
        self.api_client_patcher.stop()
    
    def test_index_properties(self):
        """Test index properties."""
        # Setup mock responses
        self.mock_api.get_index_info.return_value = MockApiResponse(
            index_type="ivf",
            config={
                "dimension": 128,
                "metric": "cosine",
                "indexType": "ivf",
                "nLists": 64
            }
        )
        
        # Get properties
        self.assertEqual(self.index.index_name, self.index_name)
        self.assertEqual(self.index.index_type, "ivf")
        self.assertEqual(self.index.index_config["dimension"], 128)
    
    def test_is_trained(self):
        """Test checking if index is trained."""
        # Setup mock response
        self.mock_api.get_index_status.return_value = MockApiResponse(
            trained=True
        )
        
        # Check trained status
        self.assertTrue(self.index.is_trained())
        self.mock_api.get_index_status.assert_called_once()
    
    def test_delete_index(self):
        """Test deleting an index."""
        # Setup mock response
        self.mock_api.delete_index.return_value = None  # Successful deletion
        
        # Delete the index
        self.index.delete_index()
        
        # Verify API call
        self.mock_api.delete_index.assert_called_once_with(
            self.index_name,
            key=self.index._key_to_hex()
        )
    
    def test_get_items(self):
        """Test retrieving items."""
        # Setup test data
        ids = ["item1", "item2"]
        include = ["vector", "metadata"]
        
        # Setup mock response
        self.mock_api.get_items.return_value = MockApiResponse(
            items=[
                MockApiResponse(
                    id="item1",
                    vector=[0.1, 0.2, 0.3],
                    metadata={"category": "A"}
                ),
                MockApiResponse(
                    id="item2",
                    vector=[0.4, 0.5, 0.6],
                    metadata={"category": "B"}
                )
            ]
        )
        
        # Get items
        items = self.index.get(ids, include)
        
        # Verify API call
        self.mock_api.get_items.assert_called_once_with(
            self.index_name,
            ids=ids,
            key=self.index._key_to_hex(),
            include=include
        )
        
        # Verify results
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "item1")
        self.assertEqual(items[1]["id"], "item2")
        self.assertEqual(items[0]["metadata"]["category"], "A")
    
    def test_train(self):
        """Test training the index."""
        # Setup mock response
        self.mock_api.train_index.return_value = None  # Successful training
        
        # Train the index
        self.index.train(batch_size=1024, max_iters=50)
        
        # Verify API call
        self.mock_api.train_index.assert_called_once()
        
        # Check the request details
        args, kwargs = self.mock_api.train_index.call_args
        self.assertEqual(args[0], self.index_name)
        self.assertEqual(kwargs["request"].batch_size, 1024)
        self.assertEqual(kwargs["request"].max_iters, 50)
    
    def test_upsert_dict_list(self):
        """Test upserting a list of dictionaries."""
        # Setup test data
        items = [
            {"id": "item1", "vector": [0.1, 0.2, 0.3], "metadata": {"category": "A"}},
            {"id": "item2", "vector": [0.4, 0.5, 0.6], "metadata": {"category": "B"}}
        ]
        
        # Setup mock response
        self.mock_api.upsert_items.return_value = None  # Successful upsert
        
        # Upsert items
        self.index.upsert(items)
        
        # Verify API call
        self.mock_api.upsert_items.assert_called_once()
        
        # Check request details
        args, kwargs = self.mock_api.upsert_items.call_args
        self.assertEqual(args[0], self.index_name)
        self.assertEqual(len(kwargs["request"].items), 2)
    
    def test_upsert_ids_vectors(self):
        """Test upserting with separate IDs and vectors."""
        # Setup test data
        ids = ["item1", "item2"]
        vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Setup mock response
        self.mock_api.upsert_items.return_value = None  # Successful upsert
        
        # Upsert items
        self.index.upsert(ids, vectors)
        
        # Verify API call
        self.mock_api.upsert_items.assert_called_once()
        
        # Check request details
        args, kwargs = self.mock_api.upsert_items.call_args
        self.assertEqual(args[0], self.index_name)
        self.assertEqual(len(kwargs["request"].items), 2)
    
    def test_delete_items(self):
        """Test deleting items."""
        # Setup test data
        ids = ["item1", "item2"]
        
        # Setup mock response
        self.mock_api.delete_items.return_value = None  # Successful deletion
        
        # Delete items
        self.index.delete(ids)
        
        # Verify API call
        self.mock_api.delete_items.assert_called_once_with(
            self.index_name,
            ids=ids,
            key=self.index._key_to_hex()
        )
    
    def test_query_single(self):
        """Test querying with a single vector."""
        # Setup test data
        query_vector = np.array([0.1, 0.2, 0.3])
        
        # Setup mock response
        self.mock_api.query_index.return_value = MockApiResponse(
            results=[
                [
                    MockApiResponse(id="item1", distance=0.1, metadata={"category": "A"}),
                    MockApiResponse(id="item2", distance=0.2, metadata={"category": "B"})
                ]
            ]
        )
        
        # Query the index
        results = self.index.query(query_vector=query_vector, top_k=2)
        
        # Verify API call
        self.mock_api.query_index.assert_called_once()
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "item1")
        self.assertEqual(results[1]["id"], "item2")
        self.assertEqual(results[0]["distance"], 0.1)
    
    def test_query_multiple(self):
        """Test querying with multiple vectors."""
        # Setup test data
        query_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Setup mock response
        self.mock_api.query_index.return_value = MockApiResponse(
            results=[
                [
                    MockApiResponse(id="item1", distance=0.1, metadata={"category": "A"}),
                    MockApiResponse(id="item2", distance=0.2, metadata={"category": "B"})
                ],
                [
                    MockApiResponse(id="item3", distance=0.3, metadata={"category": "C"}),
                    MockApiResponse(id="item4", distance=0.4, metadata={"category": "D"})
                ]
            ]
        )
        
        # Query the index
        results = self.index.query(query_vector=query_vectors, top_k=2)
        
        # Verify API call
        self.mock_api.query_index.assert_called_once()
        
        # Check results
        self.assertEqual(len(results), 2)  # Two result sets
        self.assertEqual(len(results[0]), 2)  # Two results per query
        self.assertEqual(results[0][0]["id"], "item1")
        self.assertEqual(results[1][0]["id"], "item3")
    
    def test_query_with_filters(self):
        """Test querying with metadata filters."""
        # Setup test data
        query_vector = np.array([0.1, 0.2, 0.3])
        filters = {"category": "A"}
        
        # Setup mock response
        self.mock_api.query_index.return_value = MockApiResponse(
            results=[
                [
                    MockApiResponse(id="item1", distance=0.1, metadata={"category": "A"}),
                ]
            ]
        )
        
        # Query the index
        results = self.index.query(query_vector=query_vector, filters=filters)
        
        # Verify API call
        self.mock_api.query_index.assert_called_once()
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "item1")
        self.assertEqual(results[0]["metadata"]["category"], "A")


if __name__ == "__main__":
    unittest.main()