"""
Sample dataset loader for CyborgDB.

Fetches a small reference dataset hosted on S3 on demand and caches it locally,
so quickstart and test code can populate an index without bundling data into
the SDK. Hosting the dataset out-of-band keeps the SDK lean and lets us iterate
the dataset without cutting an SDK release.

Example:
    >>> import cyborgdb
    >>> dataset = cyborgdb.load_sample_dataset()
    >>> index = client.create_index(index_name="demo", index_key=index_key)
    >>> index.upsert(dataset.items)
"""

import gzip
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Base URL for hosted sample datasets (public-read S3 bucket). Datasets live at
# versioned per-dataset paths (``<name>/v<n>/dataset.json.gz``), so the dataset
# can be iterated without an SDK release: re-upload under a new version path and
# bump the entry in ``_DATASETS``.
SAMPLE_DATASETS_BASE_URL = "https://cyborgdb-sample-datasets.s3.amazonaws.com"

# Default dataset returned by ``load_sample_dataset()`` with no arguments.
DEFAULT_SAMPLE_DATASET = "quickstart-75k"


@dataclass(frozen=True)
class _DatasetEntry:
    """Where a dataset lives and how to verify it.

    ``sha256`` is the hex SHA-256 of the decompressed JSON, pinned so a bucket
    compromise or a poisoned local cache file can't be trusted silently. The
    same digest is verified post-download and on cache read.
    """

    object_path: str
    sha256: str


# Catalog of available datasets -> their catalog entry.
_DATASETS: Dict[str, _DatasetEntry] = {
    "quickstart-75k": _DatasetEntry(
        object_path="quickstart-75k/v1/dataset.json.gz",
        sha256="6e2db96a0932f036698ebf5e25cf0871cc69b649f7fb352f9e3dddcf9af0540f",
    ),
}

# Number of leading ``queries`` exposed as ``sample_queries`` for quick demos.
_NUM_SAMPLE_QUERIES = 10

# Upper bound on the decompressed dataset size. Guards against a decompression
# bomb: a tiny gzip that expands to many GBs and OOMs the host. The largest
# shipped dataset is well under this generous cap.
_MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024


@dataclass
class SampleDataset:
    """A fully-loaded sample dataset, ready to upsert and query.

    Combines dataset metadata, loader-derived convenience fields
    (``items``, ``sample_queries``, ``example_filters``), the raw parallel
    arrays (``ids`` / ``vectors`` / ``metadata``), and the ground-truth fixture
    data (``queries``, ``*_neighbors``, ``*_recall``, ...) used to validate
    recall/accuracy. Arrays are aligned by index.
    """

    # ---- dataset metadata ----
    name: str
    version: int
    description: str
    dimension: int
    metric: str
    count: int

    # ---- convenience (built by the loader) ----
    items: List[Dict[str, Any]]
    sample_queries: List[List[float]]
    example_filters: List[Dict[str, Any]]

    # ---- raw parallel arrays (aligned by index) ----
    ids: List[str]
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]

    # ---- ground-truth fixture data (for recall / accuracy validation) ----
    queries: List[List[float]]
    metadata_queries: List[Dict[str, Any]]
    metadata_query_names: List[str]
    untrained_neighbors: List[List[int]]
    trained_neighbors: List[List[int]]
    untrained_metadata_matches: List[List[int]]
    trained_metadata_matches: List[List[int]]
    untrained_metadata_neighbors: List[List[List[int]]]
    trained_metadata_neighbors: List[List[List[int]]]
    untrained_recall: float
    trained_recall: float
    num_untrained_vectors: int
    num_trained_vectors: int


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "cyborgdb"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decompress_bounded(compressed: bytes, name: str) -> bytes:
    """Gunzip ``compressed`` with a hard size cap (anti decompression-bomb).

    Reads one byte past the cap so an over-limit payload is detected rather than
    silently truncated.
    """
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
        data = gz.read(_MAX_DECOMPRESSED_BYTES + 1)
    if len(data) > _MAX_DECOMPRESSED_BYTES:
        raise RuntimeError(
            f'Sample dataset "{name}" exceeds maximum decompressed size of '
            f"{_MAX_DECOMPRESSED_BYTES} bytes"
        )
    return data


def _hydrate(raw: Dict[str, Any]) -> SampleDataset:
    """Build the loader-derived convenience fields from the raw arrays.

    The hosted artifact and the local cache store only the raw arrays (no
    duplicated vectors), so ``items`` and ``sample_queries`` are reconstructed
    on every load.
    """
    ids = raw["ids"]
    vectors = raw["vectors"]
    metadata = raw["metadata"]
    items = [
        {"id": ids[i], "vector": vectors[i], "metadata": metadata[i]}
        for i in range(len(ids))
    ]
    return SampleDataset(
        name=raw["name"],
        version=raw["version"],
        description=raw["description"],
        dimension=raw["dimension"],
        metric=raw["metric"],
        count=raw["count"],
        items=items,
        sample_queries=raw["queries"][:_NUM_SAMPLE_QUERIES],
        example_filters=raw["exampleFilters"],
        ids=ids,
        vectors=vectors,
        metadata=metadata,
        queries=raw["queries"],
        metadata_queries=raw["metadata_queries"],
        metadata_query_names=raw["metadata_query_names"],
        untrained_neighbors=raw["untrained_neighbors"],
        trained_neighbors=raw["trained_neighbors"],
        untrained_metadata_matches=raw["untrained_metadata_matches"],
        trained_metadata_matches=raw["trained_metadata_matches"],
        untrained_metadata_neighbors=raw["untrained_metadata_neighbors"],
        trained_metadata_neighbors=raw["trained_metadata_neighbors"],
        untrained_recall=raw["untrained_recall"],
        trained_recall=raw["trained_recall"],
        num_untrained_vectors=raw["num_untrained_vectors"],
        num_trained_vectors=raw["num_trained_vectors"],
    )


def load_sample_dataset(
    name: str = DEFAULT_SAMPLE_DATASET,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> SampleDataset:
    """Load a hosted sample dataset, fetching from S3 on first use and caching
    the decompressed copy locally for subsequent calls.

    Args:
        name: Dataset name (default: ``"quickstart-75k"``).
        cache_dir: Directory to cache the decompressed dataset in. Defaults to
            ``$XDG_CACHE_HOME/cyborgdb`` or ``~/.cache/cyborgdb``.
        force_download: Re-download even if a cached copy exists.

    Returns:
        SampleDataset: The parsed dataset, ready to ``upsert`` and ``query``.

    Raises:
        ValueError: If the dataset name is unknown.
        RuntimeError: If the download fails.
    """
    if name not in _DATASETS:
        known = ", ".join(_DATASETS)
        raise ValueError(
            f'Unknown sample dataset "{name}". Available datasets: {known}.'
        )

    entry = _DATASETS[name]
    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir()
    # Cache key mirrors the versioned object path so a dataset bump never serves
    # a stale cached copy.
    cache_name = entry.object_path.replace("/", "_")
    if cache_name.endswith(".gz"):
        cache_name = cache_name[: -len(".gz")]
    cache_file = cache_root / cache_name

    if not force_download and cache_file.exists():
        try:
            cached = cache_file.read_bytes()
            # Verify the cached file against the pinned digest: a poisoned cache
            # must not be trusted. A mismatch falls through to re-download.
            if _sha256_hex(cached) == entry.sha256:
                return _hydrate(json.loads(cached.decode("utf-8")))
        except (ValueError, KeyError, OSError):
            # Corrupt cache -- fall through and re-download.
            pass

    url = f"{SAMPLE_DATASETS_BASE_URL}/{entry.object_path}"
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f'Failed to download sample dataset "{name}" from {url}: {exc}'
        ) from exc

    # The object is stored as an opaque gzip blob (no Content-Encoding: gzip),
    # so requests does not auto-decompress -- we own the gunzip step (with a
    # size cap against decompression bombs).
    data = _decompress_bounded(response.content, name)

    digest = _sha256_hex(data)
    if digest != entry.sha256:
        raise RuntimeError(
            f'Integrity check failed for sample dataset "{name}": '
            f"expected SHA-256 {entry.sha256}, got {digest}."
        )

    raw = json.loads(data.decode("utf-8"))

    # Best-effort local cache of the raw payload; a failed write must not break
    # the load. items/sample_queries are rebuilt by _hydrate() on read.
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
    except OSError:
        pass

    return _hydrate(raw)
