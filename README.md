# CyborgDB Python SDK

![PyPI - Version](https://img.shields.io/pypi/v/cyborgdb)
![PyPI - License](https://img.shields.io/pypi/l/cyborgdb)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cyborgdb)

The **CyborgDB Python SDK** provides a comprehensive client library for interacting with [CyborgDB](https://docs.cyborg.co), the first Confidential Vector Database. This SDK enables you to perform encrypted vector operations including ingestion, search, and retrieval while maintaining end-to-end encryption of your vector embeddings. Built for Python applications, it offers seamless integration into modern Python applications and services.

This SDK provides an interface to [`cyborgdb-service`](https://pypi.org/project/cyborgdb-service/) which you will need to separately install and run in order to use the SDK. For more info, please see our [docs](https://docs.cyborg.co)

## Key Features

* **End-to-End Encryption**: All vector operations maintain encryption with client-side keys
- **Zero-Trust Design**: Novel architecture keeps confidential inference data secure
- **High Performance**: GPU-accelerated indexing and retrieval with CUDA support
- **Familiar API**: Easy integration with existing AI workflows
* **Flexible Indexing**: Support for multiple index types (IVFFlat, IVFPQ, etc.) with customizable parameters

## Getting Started

To get started in minutes, check out our [Quickstart Guide](https://docs.cyborg.co/quickstart).


### Installation

1. Install `cyborgdb-service`

```bash
# Install the CyborgDB Service
pip install cyborgdb-service

# Or via Docker
docker pull cyborginc/cyborgdb-service
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

## Documentation

For more information on CyborgDB, see the [Cyborg Docs](https://docs.cyborg.co).

## License

The CyborgDB Python SDK is licensed under the MIT License.
