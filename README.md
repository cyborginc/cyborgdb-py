<p align="center">
  <a href="https://www.cyborg.co">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cyborginc/cyborgdb-py/main/assets/cyborgdb-logo-dark.svg">
      <img src="https://raw.githubusercontent.com/cyborginc/cyborgdb-py/main/assets/cyborgdb-logo-light.svg" alt="CyborgDB" width="320">
    </picture>
  </a>
</p>

# CyborgDB Python SDK

![PyPI - Version](https://img.shields.io/pypi/v/cyborgdb)
![PyPI - License](https://img.shields.io/pypi/l/cyborgdb)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cyborgdb)

The **CyborgDB Python SDK** is the Python client for [CyborgDB](https://www.cyborg.co) — the vector database that stays encrypted even while it's searching. Run similarity search directly on encrypted data with client-side keys; only the result of a query is ever decrypted, never the index. Built for Python, it fits into existing AI and data workflows.

This SDK talks to [`cyborgdb-service`](https://hub.docker.com/r/cyborginc/cyborgdb-service), which you self-host in your own VPC or on-prem and run alongside your app. Install and start it separately. See our [docs](https://docs.cyborg.co) for more info.

## Key Features

- **Encryption-in-use**: Search runs directly on ciphertext; only the query result is decrypted, never the index or stored vectors
- **Encrypted ANN**: Disk-backed encrypted DiskIVF index with recall within 2% of a plaintext baseline ([read the benchmarks](https://www.cyborg.co/performance))
- **Filters on encrypted metadata**: Combine vector similarity with equality and range predicates in a single request
- **BYOK / HYOK**: Wrap per-index keys with AWS KMS or AWS Secrets Manager, or hold the key client-side — you control the key material
- **Per-tenant key isolation**: Per-index, per-user keys with cryptographic RBAC; revoke a user and their keys are erased
- **Pythonic API**: Familiar client/index interface that integrates with existing Python AI workflows

## Getting Started

To get started in minutes, check out our [Quickstart Guide](https://docs.cyborg.co/quickstart).


### Install the SDK

1. Install `cyborgdb-service`

```bash
# Pull the CyborgDB Service image
docker pull cyborginc/cyborgdb-service

# Or install via pip
pip install cyborgdb-service
```

2. Install `cyborgdb` SDK:

```bash
# Install the CyborgDB Python SDK
pip install cyborgdb
```

### Index and query vectors

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
results = index.query(query_vectors=query_vector, top_k=5)

# Print the results
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}")
# ID: doc1, Distance: 1.1314
# ID: doc2, Distance: 1.1314
```

### Run batch queries
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
# Results for query 0:
#   ID: doc1, Distance: 0.0000
#   ID: doc2, Distance: 0.0000
#
# Results for query 1:
#   ID: doc1, Distance: 1.1314
#   ID: doc2, Distance: 1.1314
```

### Filter results by metadata
```python
# Search with metadata filters
query_vector = [0.1] * 128
results = index.query(
    query_vectors=query_vector,
    top_k=10,
    n_probes=1,
    greedy=False,
    filters={'category': 'greeting', 'language': 'en'},
    include=['distance', 'metadata']
)

# Print the results
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}, Metadata: {result['metadata']}")
# ID: doc1, Distance: 0.0000, Metadata: {'category': 'greeting', 'language': 'en'}
```

### Bring Your Own Key (BYOK) via KMS

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

Alternatively, the SDK can supply the key itself — pass `index_key` and omit
`kms_name`. This is the no-KMS path, which the service records internally as
`provider: none`:

```python
index = client.create_index(
    index_name='sdk-keyed-index',
    index_key=index_key,
    dimension=128,
)
```

Supply **exactly one** of `index_key` / `kms_name` — passing both is rejected
by the service with a 400, since the named slot already determines the key
source.

### Control access with per-user keys

When the service runs with a root admin key (`CYBORGDB_SERVICE_ROOT_KEY`) set, RBAC
is enabled. The root can mint **per-user API keys** scoped to a single index,
each with a `read` / `write` permission set. Permissions are enforced
*cryptographically*: a user's wrapped data-encryption keys **are** their
permission set. A read-only user cannot decrypt for a write operation;
revoking a user erases their keys.

```python
# Admin (root) client: mint users on an existing index.
admin = Client(base_url, api_key=ROOT_API_KEY)
index = admin.load_index(index_name='kms-backed-index')   # KMS-backed (see BYOK)

reader = index.create_user(permissions=['read'])
writer = index.create_user(permissions=['read', 'write'])
# Each returns {'user_id': '<hex>', 'api_key': 'cdbk_...'} — the api_key is
# shown ONCE and never stored by the service. Hand it to the user securely.

index.list_users()                 # [{'user_id': ..., 'permissions': [...]}, ...]
index.delete_user(reader['user_id'])   # revoke; the key stops working immediately
```

A user authenticates with their `cdbk_` key and needs no index key of their own
— they load the index by name and the service resolves its key:

```python
user = Client(base_url, api_key=reader['api_key'])
idx = user.load_index(index_name='kms-backed-index')   # no index_key
idx.query(query_vectors=[...], top_k=5)                # allowed for 'read'
idx.upsert(items)                                      # raises ValueError for read-only users
```

> User keys resolve the index key server-side, so they work against
> **KMS-backed** indexes. SDK-supplied-key indexes (`provider: none`) have no
> server-side key for the service to resolve on a user's behalf. See the
> service's `rbac.md` for the full design.

## Documentation

For more information on CyborgDB, see the [Cyborg Docs](https://docs.cyborg.co).

## License

The CyborgDB Python SDK is licensed under the MIT License.
