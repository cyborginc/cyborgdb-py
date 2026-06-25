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
from cyborgdb import Client, load_sample_dataset

# Initialize the client
client = Client('https://localhost:8000', 'your-service-root-key')

# Generate a 32-byte encryption key
index_key = client.generate_key()

# Create an encrypted index
index = client.create_index(
    index_name='my-index', 
    index_key=index_key
)

# Load the hosted sample dataset (fetched from S3 on first use, cached locally)
dataset = load_sample_dataset()  # 75k 128-dim vectors with metadata

# Add the encrypted vector items
index.upsert(dataset.items)

# Query the encrypted index with a sample query vector
results = index.query(query_vectors=dataset.sample_queries[0], top_k=5)

# Print the results (guaranteed non-empty against the sample dataset)
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']}")
```

> **Sample dataset:** `load_sample_dataset()` pulls a small reference dataset
> from S3 on demand and caches it locally — it is not bundled into the SDK.
> Each item has an explicit `id`, a 128-dim `vector`, and `metadata` with both
> string (`string`) and numeric (`number`) fields, so the same dataset drives
> ANN similarity search, metadata filter queries, and numeric range queries. It
> also ships `sample_queries` (query vectors) and `example_filters` (curated,
> guaranteed-to-match filters).

> **Encryption model:** the index is encrypted at rest, but an encrypted DB
> does **not** mean vectors are auto-hidden from you. You must pass your index
> key on `load_index` / `get` / `query` to retrieve **decrypted** vectors and
> metadata — without the key, only encrypted ciphertext is ever readable.
> HYOK-level security is not implied unless you manage the key material
> yourself (see BYOK below).

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

### Filter results by metadata and range
```python
dataset = load_sample_dataset()
query_vector = dataset.sample_queries[0]

# Equality filter on a string field
filtered = index.query(
    query_vectors=query_vector,
    top_k=10,
    filters={'string': 'string_0'},
    include=['distance', 'metadata'],
)

# Numeric range query (bounded) — combine similarity with a range predicate
ranged = index.query(
    query_vectors=query_vector,
    top_k=10,
    filters={'number': {'$gte': 1250, '$lte': 2500}},
    include=['distance', 'metadata'],
)

# The dataset also ships curated, guaranteed-to-match filters:
for example in dataset.example_filters:
    res = index.query(
        query_vectors=query_vector, top_k=5, filters=example['filter']
    )
    print(f"{example['name']}: {len(res)} results")
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
admin = Client(base_url, api_key=SERVICE_ROOT_KEY)
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
