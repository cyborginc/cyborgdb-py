# CyborgDB Python SDK

![PyPI - Version](https://img.shields.io/pypi/v/cyborgdb)
![PyPI - License](https://img.shields.io/pypi/l/cyborgdb)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cyborgdb)

The **CyborgDB Python SDK** provides a comprehensive client library for interacting with [CyborgDB](https://docs.cyborg.co), the first Confidential Vector Database. This SDK enables you to perform encrypted vector operations including ingestion, search, and retrieval while maintaining end-to-end encryption of your vector embeddings. Built for Python applications, it offers seamless integration into modern Python applications and services.

This SDK provides an interface to [`cyborgdb-service`](https://pypi.org/project/cyborgdb-service/) which you will need to separately install and run in order to use the SDK. For more info, please see our [docs](https://docs.cyborg.co).

## Key Features

- **End-to-End Encryption**: All vector operations maintain encryption with client-side keys
- **Zero-Trust Design**: Novel architecture keeps confidential inference data secure
- **High Performance**: GPU-accelerated indexing and retrieval with CUDA support
- **Familiar API**: Easy integration with existing AI workflows
- **Encrypted DiskIVF Indexing**: Disk-backed inverted-file index with customizable training parameters

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

### Usage

```python
from cyborgdb import Client

# Initialize the client
client = Client('https://localhost:8000', 'your-api-key')

# Generate a 32-byte encryption key
index_key = client.generate_key()

# Create an encrypted index
index = client.create_index(
    index_name='my-index', 
    index_key=index_key
)

# Add encrypted vector items
items = [
    {
        'id': 'doc1',
        'vector': [0.1] * 128,  # Replace with real embeddings
        'contents': 'Hello world!',
        'metadata': {'category': 'greeting', 'language': 'en'}
    },
    {
        'id': 'doc2',
        'vector': [0.1] * 128,  # Replace with real embeddings
        'contents': 'Bonjour le monde!',
        'metadata': {'category': 'greeting', 'language': 'fr'}
    }
]

index.upsert(items)

# Query the encrypted index
query_vector = [0.2] * 128  # 128 dimensions
results = index.query(query_vectors=query_vector,top_k=5)

# Print the results
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}")
```

### Advanced Usage

#### Batch Queries
```python
# Search with multiple query vectors simultaneously
query_vectors = [
    [0.1] * 128,
    [0.2] * 128
]

batch_results = index.query(query_vectors=query_vectors, top_k=5)

# Print the results (batch queries return list of lists)
for i, query_results in enumerate(batch_results):
    print(f"\nResults for query {i}:")
    for result in query_results:
        print(f"  ID: {result['id']}, Distance: {result['distance']}")
```

#### Metadata Filtering
```python
# Search with metadata filters
query_vector = [0.1] * 128
results = index.query(
    query_vectors=query_vector,
    top_k=10,
    n_probes=1,
    greedy=False,
    filters={'category': 'greeting', 'language': 'en'},
    include=['distance', 'metadata', 'contents']
)

# Print the results
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}, Metadata: {result['metadata']}")
```

#### Bring Your Own Key (BYOK) via KMS

When the service is configured with a `kms.registry` entry, the SDK can
delegate key management entirely to the server-side KMS. The service
generates the data encryption key, wraps it under the named KMS slot, and
persists the envelope — the SDK never sees or holds the key.

```python
# Create a KMS-backed index — no index_key from the SDK side.
# 'vendor-kms-slot' must match an entry in the service's cyborgdb.yaml.
index = client.create_index(
    index_name='kms-backed-index',
    kms_name='vendor-kms-slot',
    dimension=128,
    metric='euclidean',
)

# Reopening the index later doesn't require a key either; the service
# resolves the data key from the index's stored KMS envelope.
loaded = client.load_index(index_name='kms-backed-index')
loaded.upsert(items)
```

For `provider: none` registry entries, the SDK still supplies the KEK on
every call — pass both `index_key` and `kms_name`:

```python
index = client.create_index(
    index_name='sdk-keyed-index',
    index_key=index_key,
    kms_name='plain',   # registry slot with provider: none
    dimension=128,
)
```

> **How slots are configured.** A `kms.registry` slot is added to the
> service's `cyborgdb.yaml` by your **cyborgdb-service operator** — not
> from the SDK. Each slot declares one provider (`aws-kms`, `aws`,
> or `none`) plus the AWS identifiers needed to wrap/unwrap data keys.
> For real-KMS slots (`aws-kms` / `aws`), set-up also requires IAM
> work on the customer's AWS account; see `BYOK.md` in the
> cyborgdb-service repo for the full operator + customer walkthrough.
> From the SDK side, you only need the slot name your operator
> provisioned.

## Documentation

For more information on CyborgDB, see the [Cyborg Docs](https://docs.cyborg.co).

## License

The CyborgDB Python SDK is licensed under the MIT License.
