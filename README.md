# CyborgDB Python SDK

The **CyborgDB Python SDK** provides a comprehensive client library for interacting with [CyborgDB](https://www.cyborg.co), the first Confidential Vector Database. This SDK enables you to perform encrypted vector operations including ingestion, search, and retrieval while maintaining end-to-end encryption of your vector embeddings. Built for Python applications, it offers seamless integration into modern Python applications and services.

This SDK provides an interface to `cyborgdb-service` which you will need to separately install and run in order to use the SDK. For more info, please see our [docs](https://docs.cyborg.co)

**Why CyborgDB?**

Vector Search powers critical AI applications like RAG systems, recommendation engines, and semantic search. The CyborgDB Python SDK brings confidential computing to your Python applications and services, ensuring vector embeddings remain encrypted throughout their entire lifecycle while providing fast, accurate search capabilities.

**Key Features**

* **End-to-End Encryption**: All vector operations maintain encryption with client-side keys
* **Batch Operations**: Efficient batch queries and upserts for high-throughput applications
* **Flexible Indexing**: Support for multiple index types (IVFFlat, IVFPQ, etc.) with customizable parameters

**Installation**

1. Install `cyborgdb-service`

```bash
# Install the CyborgDB Service
pip install cyborgdb-service
```

2. Install `cyborgdb` SDK:

```bash
# Install the CyborgDB Python SDK
pip install cyborgdb
```

**Usage**

```python
from cyborgdb import Client, IndexIVFFlat
import secrets

# Initialize the client
client = Client('https://localhost:8000', 'your-api-key')

# Generate a 32-byte encryption key
index_key = secrets.token_bytes(32)

# Create an encrypted index
index = await client.create_index('my-index', index_key, IndexIVFFlat(128, 1024))

# Add encrypted vector items
items = [
    {
        'id': 'doc1',
        'vector': [([0.1] * 128)],  # Replace with real embeddings
        'contents': 'Hello world!',
        'metadata': {'category': 'greeting', 'language': 'en'}
    },
    {
        'id': 'doc2',
        'vector': [([0.1] * 128)],  # Replace with real embeddings
        'contents': 'Bonjour le monde!',
        'metadata': {'category': 'greeting', 'language': 'fr'}
    }
]

await index.upsert(items)

# Query the encrypted index
query_vector = [0.1, 0.2, 0.3, *([0.0] * 125)]  # 128 dimensions
results = await index.query(query_vector, 10)

# Print the results
for result in results.results:
    print(f"ID: {result.id}, Distance: {result.distance}")
```

**Advanced Usage**

**Batch Queries**

```python
# Search with multiple query vectors simultaneously
query_vectors = [
    [0.1, 0.2, 0.3, *([0.0] * 125)],
    [0.4, 0.5, 0.6, *([0.0] * 125)]
]

batch_results = await index.query(query_vectors, 5)
```

**Metadata Filtering**

```python
# Search with metadata filters
results = await index.query(
    query_vector,
    10,      # top_k
    1,       # n_probes
    False,   # greedy
    {'category': 'greeting', 'language': 'en'},  # filters
    ['distance', 'metadata', 'contents']         # include
)
```

**Index Training**

```python
# Train the index for better query performance (recommended for IVF indexes)
await index.train(2048, 100, 1e-6)
```

**Documentation**

For more detailed documentation, visit:
* [CyborgDB Documentation](https://docs.cyborg.co/)

**License**

The CyborgDB Python SDK is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

**About CyborgDB**

CyborgDB is dedicated to making AI safe and secure through confidential computing. We develop solutions that enable organizations to leverage AI while maintaining the confidentiality and privacy of their data.

[Visit our website](https://www.cyborg.co/) | [Contact Us](mailto:hello@cyborg.co)