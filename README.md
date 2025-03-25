# CyborgDB Python Client

A Python client library for [CyborgDB](https://cyborgdb.io), the confidential vector database for secure AI applications.

## Features

- **End-to-End Encryption**: All vector data is encrypted using strong cryptographic standards
- **Similarity Search**: Fast and accurate similarity search on encrypted vectors
- **Automatic Embedding**: Seamlessly convert text to embeddings using SentenceTransformers
- **Flexible Storage**: Multiple backend options including in-memory, Redis, and PostgreSQL
- **Advanced Indexing**: Support for multiple ANN algorithms (IVF, IVFPQ, IVFFlat)
- **Metadata Filtering**: Filter search results based on metadata

## Installation

```bash
pip install cyborgdb
```

## Dependencies

- Python 3.8+
- cyborgdb_core (C++ extension module, installed automatically)
- numpy
- sentence-transformers (for automatic embedding generation)
- pydantic

## Quick Start

```python
from cyborgdb.client.client import Client, DBConfig, IndexIVFFlat, generate_key

# Initialize client with in-memory storage
client = Client(
    index_location=DBConfig("memory"),
    config_location=DBConfig("memory")
)

# Generate a secure encryption key
index_key = generate_key()

# Create an index configuration
index_config = IndexIVFFlat(
    dimension=384,  # Matches the embedding model dimension
    n_lists=100,
    metric="cosine"
)

# Create an encrypted index with an embedding model
index = client.create_index(
    index_name="my_documents",
    index_key=index_key,
    index_config=index_config,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents with automatic embedding
documents = [
    {
        "id": "doc1",
        "contents": "CyborgDB is a confidential vector database for secure AI applications.",
        "metadata": {"category": "database"}
    },
    {
        "id": "doc2",
        "contents": "Vector databases store and query high-dimensional vector embeddings.",
        "metadata": {"category": "database"}
    }
]

# Upsert documents
index.upsert(documents)

# Train the index for faster search
index.train()

# Search the index
results = index.query(
    query_contents="How do vector databases work?",
    top_k=3
)

# Print results
for i, result in enumerate(results):
    print(f"{i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    print(f"   Metadata: {result['metadata']}")
```

## Using Different Storage Backends

### Redis

```python
# Store index and configuration in Redis
client = Client(
    index_location=DBConfig("redis", connection_string="host:127.0.0.1,port:6379,db:0"),
    config_location=DBConfig("redis", connection_string="host:127.0.0.1,port:6379,db:0")
)
```

### PostgreSQL

```python
# Store index and configuration in PostgreSQL
client = Client(
    index_location=DBConfig(
        "postgres",
        table_name="cyborg_indexes",
        connection_string="postgresql://user:password@localhost/dbname"
    ),
    config_location=DBConfig(
        "postgres",
        table_name="cyborg_configs",
        connection_string="postgresql://user:password@localhost/dbname"
    )
)
```

## Creating Different Index Types

### IVF (Inverted File)

```python
index_config = IndexIVF(
    dimension=384,
    n_lists=100,
    metric="cosine"
)
```

### IVFPQ (Inverted File with Product Quantization)

```python
index_config = IndexIVFPQ(
    dimension=384,
    n_lists=100,
    pq_dim=32,  # Subvector dimensionality
    pq_bits=8,  # Bits per subquantizer
    metric="cosine"
)
```

### IVFFlat (Inverted File with Flat Storage)

```python
index_config = IndexIVFFlat(
    dimension=384,
    n_lists=100,
    metric="cosine"
)
```

## Querying with Filters

```python
# Query with metadata filters
results = index.query(
    query_contents="Machine learning applications",
    filters={"category": "database", "tags": ["vector-db"]},
    top_k=5
)
```

## Performance Tips

1. Choose the appropriate index type for your workload:
   - IVFFlat: Higher accuracy, more memory usage
   - IVFPQ: Less memory usage, slightly lower accuracy

2. Increase the number of lists (`n_lists`) for larger datasets

3. For IVFPQ, adjust `pq_dim` and `pq_bits` based on your accuracy requirements

4. Use `max_cache_size` to control memory usage when loading an index

## Documentation

For detailed documentation, see the [official documentation](https://docs.cyborgdb.io).

## License
TBD