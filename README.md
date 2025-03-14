# CyborgDB Python Client

A clean, user-friendly Python client for interacting with the CyborgDB vector database.

## Installation

```bash
pip install cyborgdb-client
```

Or install from source:

```bash
git clone https://github.com/yourusername/cyborgdb-client-python.git
cd cyborgdb-client-python
pip install -e .
```

## Quick Start

```python
from cyborgdb_client import CyborgDBClient

# Initialize the client
client = CyborgDBClient(host="http://localhost:8000")

# Create an index
client.create_index(
    vector_dimension=768,
    index_type="IVF",
    namespace="documents",
    metric_type="COSINE"
)

# Insert vectors
vectors = [
    {
        "id": "doc_1",
        "vector": [0.1, 0.2, ..., 0.7],  # 768-dimensional vector
        "metadata": {
            "title": "Document 1",
            "category": "Category A"
        }
    }
]
client.upsert(vectors, namespace="documents")

# Search vectors
results = client.query(
    vector=[0.2, 0.3, ..., 0.8],  # Query vector
    k=10,
    namespace="documents",
    filter={"category": "Category A"}
)

# Close the client when done
client.close()
```

## Features

- Simple, intuitive interface
- Comprehensive error handling
- Full support for all CyborgDB operations:
  - Vector search
  - Vector insertion and updates
  - Vector deletion
  - Index creation and management
  - Filtering and metadata management

## Documentation

For detailed API documentation, please see the [docs](docs/) directory.

## Examples

See the [examples](examples/) directory for more usage examples.

## Development

### Regenerating the Client

If the CyborgDB API changes, you can regenerate the underlying OpenAPI client:

```bash
# Install the OpenAPI generator
npm install @openapitools/openapi-generator-cli -g

# Generate the client
openapi-generator-cli generate -i openapi.json -g python -o .
```

Then update the wrapper if necessary.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.