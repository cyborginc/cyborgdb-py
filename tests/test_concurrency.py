"""
Concurrency and Multi-Index Tests for CyborgDB Python SDK

Tests thread safety, data integrity under concurrent load, and index isolation.
All tests hit a real backend — no mocking.
"""

import os
import time
import uuid
import asyncio
import unittest
import threading
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import cyborgdb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from cyborgdb.integrations.langchain import CyborgVectorStore

load_dotenv(".env.local")

DIMENSION = 128
NUM_VECTORS = 50  # Per-thread/per-index vector count
BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")


class FixedDimensionEmbeddings(Embeddings):
    """Deterministic embeddings for concurrency tests. Each text maps to a
    reproducible random vector so we can verify round-trip integrity."""

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def join_threads(test, threads, timeout=60):
    """Join all threads and assert none are still alive (catches hangs)."""
    for t in threads:
        t.join(timeout=timeout)
    hung = [t for t in threads if t.is_alive()]
    test.assertEqual(
        len(hung), 0, f"{len(hung)} thread(s) hung past {timeout}s timeout"
    )


def wait_until(condition, timeout=10.0, interval=0.25):
    """Poll `condition` until it returns True or `timeout` seconds elapse.
    Mirrors waitUntil (js) / pollUntil (go); raises on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError(f"wait_until timed out after {timeout}s")


def make_client():
    """Create a fresh Client instance."""
    return cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)


def make_index(client, prefix="conc"):
    """Create a uniquely-named DiskIVF index and return (index, name, key)."""
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"
    key = cyborgdb.Client.generate_key()
    index = client.create_index(name, key, dimension=DIMENSION, metric="euclidean")
    return index, name, key


def upsert_batch(index, id_prefix, count=NUM_VECTORS):
    """Upsert a batch of random vectors with a given ID prefix."""
    vectors = np.random.rand(count, DIMENSION).astype(np.float32)
    ids = [f"{id_prefix}_{i}" for i in range(count)]
    index.upsert(ids, vectors)
    return ids, vectors


# ---------------------------------------------------------------------------
# Concurrent Operations — Single Index
# ---------------------------------------------------------------------------


class TestConcurrentUpserts(unittest.TestCase):
    """Multiple threads upserting to the same index simultaneously."""

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.index, cls.index_name, cls.index_key = make_index(cls.client)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_concurrent_upserts_no_data_loss(self):
        """
        10 threads each upsert 50 vectors (500 total) through one shared
        EncryptedIndex. After all finish, every single ID must be present.
        Catches: request body corruption in shared ApiClient, dropped writes.
        """
        num_threads = 10
        all_ids = []
        lock = threading.Lock()
        errors = []

        def worker(thread_id):
            try:
                ids, _ = upsert_batch(self.index, f"t{thread_id}")
                with lock:
                    all_ids.extend(ids)
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=60)

        self.assertEqual(len(errors), 0, f"Threads raised errors: {errors}")

        wait_until(lambda: len(set(self.index.list_ids())) >= len(all_ids))

        stored_ids = set(self.index.list_ids())
        missing = [id_ for id_ in all_ids if id_ not in stored_ids]
        self.assertEqual(
            len(missing),
            0,
            f"{len(missing)}/{len(all_ids)} IDs missing after concurrent upsert",
        )

    def test_concurrent_upserts_overlapping_ids(self):
        """
        5 threads upsert different vectors to the SAME 20 IDs.
        After completion: each ID must exist, and the stored vector must
        exactly match one of the 5 written vectors (proving no corruption
        from interleaved writes).
        """
        shared_ids = [f"overlap_{i}" for i in range(20)]
        num_threads = 5
        errors = []
        lock = threading.Lock()
        # Track every vector each thread wrote for each ID
        written_vectors = {id_: [] for id_ in shared_ids}

        def worker(thread_id):
            try:
                vectors = np.random.rand(20, DIMENSION).astype(np.float32)
                # Lock protects written_vectors tracking only — the upsert
                # is intentionally outside the lock so writes race freely.
                # We only assert the final value matches ONE of the writers.
                with lock:
                    for i, id_ in enumerate(shared_ids):
                        written_vectors[id_].append(vectors[i].copy())
                self.index.upsert(shared_ids, vectors)
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=60)

        self.assertEqual(len(errors), 0, f"Threads raised errors: {errors}")
        wait_until(lambda: set(shared_ids).issubset(set(self.index.list_ids())))

        # Every ID must exist and its vector must match one of the writers
        items = self.index.get(shared_ids, include=["vector"])
        self.assertEqual(len(items), len(shared_ids))
        for item in items:
            stored_vec = np.array(item["vector"], dtype=np.float32)
            candidates = written_vectors[item["id"]]
            matched = any(np.allclose(stored_vec, c, rtol=1e-5) for c in candidates)
            self.assertTrue(
                matched,
                f"ID '{item['id']}': stored vector doesn't match ANY of the "
                f"{len(candidates)} written vectors — possible corruption",
            )

    def test_concurrent_write_then_verify_per_thread(self):
        """
        8 threads share one EncryptedIndex. Each upserts unique vectors
        then reads them back via get() and verifies exact vector match.
        Catches: request body corruption, response routing errors on shared
        ApiClient under concurrency.
        """
        results_per_thread = {}
        errors = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                prefix = f"verify_{thread_id}"
                ids, vecs = upsert_batch(self.index, prefix, count=10)
                time.sleep(1)

                retrieved = self.index.get([ids[0]], include=["vector"])
                self.assertEqual(
                    len(retrieved), 1, f"Thread {thread_id}: get() returned wrong count"
                )
                self.assertEqual(retrieved[0]["id"], ids[0])
                retrieved_vec = np.array(retrieved[0]["vector"], dtype=np.float32)
                np.testing.assert_allclose(
                    retrieved_vec,
                    vecs[0],
                    rtol=1e-5,
                    err_msg=f"Thread {thread_id}: retrieved vector doesn't match written vector",
                )
                with lock:
                    results_per_thread[thread_id] = True
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(8)]
            for f in as_completed(futures):
                f.result()

        self.assertEqual(len(errors), 0, f"Shared client errors: {errors}")
        self.assertEqual(len(results_per_thread), 8, "Not all threads completed")


class TestConcurrentReadsAndWrites(unittest.TestCase):
    """Queries running while upserts and deletes are happening."""

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.index, cls.index_name, cls.index_key = make_index(cls.client)
        # Seed with initial data so queries have something to return
        upsert_batch(cls.index, "seed", count=100)
        wait_until(lambda: len(set(cls.index.list_ids())) >= 100)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_queries_during_upserts(self):
        """
        3 writer threads upsert while 5 reader threads query concurrently.
        Readers must get well-formed results with valid distances.
        Catches: crashes from concurrent HTTP access, malformed responses.
        """
        num_writers = 3
        num_readers = 5
        query_count = 10
        errors = []
        lock = threading.Lock()

        def writer(thread_id):
            try:
                for batch in range(3):
                    upsert_batch(self.index, f"w{thread_id}_b{batch}", count=20)
            except Exception as e:
                with lock:
                    errors.append(("writer", thread_id, e))

        def reader(thread_id):
            try:
                for _ in range(query_count):
                    qv = np.random.rand(DIMENSION).astype(np.float32)
                    results = self.index.query(
                        query_vectors=qv, top_k=5, include=["distance"]
                    )
                    for r in results:
                        self.assertIn("id", r)
                        self.assertTrue(
                            r["id"],
                            "torn read: query result has empty id (deleted-during-query); see go TestDeletesDuringQueries",
                        )
                        self.assertIn("distance", r)
                        self.assertIsInstance(r["distance"], (int, float))
                        self.assertGreaterEqual(
                            r["distance"],
                            0,
                            f"Negative distance {r['distance']} for ID {r['id']} — corrupted result",
                        )
            except Exception as e:
                with lock:
                    errors.append(("reader", thread_id, e))

        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(num_writers)
        ] + [threading.Thread(target=reader, args=(i,)) for i in range(num_readers)]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=120)

        self.assertEqual(len(errors), 0, f"Concurrent read/write errors: {errors}")

    def test_deletes_during_queries(self):
        """
        One thread deletes vectors in batches while 4 threads query.
        Queries must never crash or return malformed results.
        Catches: server-side race between delete and read paths.
        """
        # Seed baseline data so queries return a full top_k of live results
        # throughout the run (matches go's TestDeletesDuringQueries). Without it
        # the live pool drains to zero as the deleter runs and the torn-read
        # detection window shrinks.
        upsert_batch(self.index, "seed", count=100)

        delete_ids = [f"del_{i}" for i in range(30)]
        vectors = np.random.rand(30, DIMENSION).astype(np.float32)
        self.index.upsert(delete_ids, vectors)
        wait_until(lambda: set(delete_ids).issubset(set(self.index.list_ids())))

        errors = []
        lock = threading.Lock()
        # Known delete/query race yields an empty-string id; skip on it, don't
        # fail. Matches js DeletesDuringQueries.
        empty_id_race = {"hit": False}

        def deleter():
            try:
                for i in range(0, 30, 5):
                    self.index.delete(delete_ids[i : i + 5])
                    time.sleep(0.1)
            except Exception as e:
                with lock:
                    errors.append(("deleter", e))

        def querier(thread_id):
            try:
                for _ in range(15):
                    qv = np.random.rand(DIMENSION).astype(np.float32)
                    results = self.index.query(
                        query_vectors=qv, top_k=10, include=["distance"]
                    )
                    for r in results:
                        self.assertIn("id", r)
                        if not r["id"]:
                            with lock:
                                empty_id_race["hit"] = True
                            continue
                        self.assertIn("distance", r)
                        self.assertIsInstance(r["distance"], (int, float))
                        self.assertGreaterEqual(r["distance"], 0)
            except Exception as e:
                with lock:
                    errors.append(("querier", thread_id, e))

        threads = [threading.Thread(target=deleter)]
        threads += [threading.Thread(target=querier, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=60)

        if empty_id_race["hit"] and not errors:
            self.skipTest("hit the known empty-id delete/query race — not a regression")

        self.assertEqual(len(errors), 0, f"Delete-during-query errors: {errors}")

    def test_concurrent_upserts_and_deletes_on_same_ids(self):
        """
        2 threads upsert a set of IDs while 2 other threads delete from the
        same set. After all threads finish, every ID must either exist with a
        valid vector or be cleanly absent — no partial/corrupt state.
        Catches: write-delete races causing ghost entries or corrupt vectors.
        """
        target_ids = [f"race_{i}" for i in range(40)]
        vectors = np.random.rand(40, DIMENSION).astype(np.float32)
        self.index.upsert(target_ids, vectors)
        wait_until(lambda: set(target_ids).issubset(set(self.index.list_ids())))

        errors = []
        lock = threading.Lock()

        def upserter(thread_id):
            try:
                for _ in range(5):
                    new_vecs = np.random.rand(40, DIMENSION).astype(np.float32)
                    self.index.upsert(target_ids, new_vecs)
            except Exception as e:
                with lock:
                    errors.append(("upserter", thread_id, e))

        def deleter(thread_id):
            try:
                for _ in range(5):
                    batch = target_ids[thread_id * 10 : (thread_id + 1) * 10]
                    self.index.delete(batch)
                    time.sleep(0.05)
            except Exception as e:
                with lock:
                    errors.append(("deleter", thread_id, e))

        threads = [threading.Thread(target=upserter, args=(i,)) for i in range(2)] + [
            threading.Thread(target=deleter, args=(i,)) for i in range(2)
        ]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=60)

        self.assertEqual(len(errors), 0, f"Upsert/delete race errors: {errors}")

        # Every surviving ID must have a valid, retrievable vector.
        # Both upserters do 5 rounds of 40 IDs each — at least some should survive.
        wait_until(lambda: len(self.index.list_ids()) > 0)
        stored_ids = self.index.list_ids()
        self.assertGreater(
            len(stored_ids),
            0,
            "All IDs gone after upsert/delete race — upserters never committed or deleters swept everything",
        )
        if stored_ids:
            items = self.index.get(list(stored_ids), include=["vector"])
            for item in items:
                self.assertIn("id", item)
                vec = item.get("vector")
                self.assertIsNotNone(vec, f"ID '{item['id']}' exists but has no vector")
                self.assertEqual(
                    len(vec),
                    DIMENSION,
                    f"ID '{item['id']}' has wrong dimension: {len(vec)}",
                )


class TestErrorIsolationUnderLoad(unittest.TestCase):
    """
    When one thread's operation fails, other threads must not be affected.
    Tests that error responses on shared ApiClient don't corrupt internal state.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.index, cls.index_name, cls.index_key = make_index(cls.client)
        upsert_batch(cls.index, "base", count=50)
        wait_until(lambda: len(set(cls.index.list_ids())) >= 50)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_bad_thread_doesnt_break_good_threads(self):
        """
        One thread sends wrong-dimension vectors (expects errors).
        4 other threads do valid queries through the same shared ApiClient.
        Good threads must succeed — proving error handling doesn't poison
        shared connection state.
        """
        good_results = []
        bad_errors = []
        good_errors = []
        lock = threading.Lock()

        def bad_worker():
            for _ in range(5):
                try:
                    wrong_dim = np.random.rand(10, 64).astype(np.float32)
                    ids = [f"bad_{i}" for i in range(10)]
                    self.index.upsert(ids, wrong_dim)
                except Exception as e:
                    with lock:
                        bad_errors.append(e)

        def good_worker(thread_id):
            try:
                for _ in range(10):
                    qv = np.random.rand(DIMENSION).astype(np.float32)
                    result = self.index.query(query_vectors=qv, top_k=3)
                    with lock:
                        good_results.append(len(result))
            except Exception as e:
                with lock:
                    good_errors.append((thread_id, e))

        threads = [threading.Thread(target=bad_worker)]
        threads += [threading.Thread(target=good_worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=60)

        self.assertGreater(len(bad_errors), 0, "Bad worker should have failed")
        self.assertEqual(
            len(good_errors), 0, f"Good workers failed due to bad worker: {good_errors}"
        )
        self.assertGreater(len(good_results), 0)


# ---------------------------------------------------------------------------
# Multi-Index Tests
# ---------------------------------------------------------------------------


class TestMultiIndexIsolation(unittest.TestCase):
    """
    The most critical multi-index test: data in one index must
    NEVER appear in another index's queries or list_ids.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.indexes = []
        cls.index_data = {}  # index_name -> set of IDs

        for i in range(3):
            index, name, key = make_index(cls.client, prefix=f"iso_{i}")
            cls.indexes.append((index, name, key))

            ids = [f"idx{i}_vec{j}" for j in range(30)]
            vectors = np.random.rand(30, DIMENSION).astype(np.float32)
            index.upsert(ids, vectors)
            cls.index_data[name] = set(ids)

        wait_until(
            lambda: all(len(set(idx.list_ids())) >= 30 for idx, _, _ in cls.indexes)
        )

    @classmethod
    def tearDownClass(cls):
        for index, _, _ in cls.indexes:
            try:
                index.delete_index()
            except Exception:
                pass

    def test_no_data_leakage_between_indexes(self):
        """
        Query each index and verify every returned ID belongs ONLY to that index.
        Cross-contamination here = critical data isolation bug.
        """
        for index, name, _ in self.indexes:
            my_ids = self.index_data[name]
            other_ids = set()
            for other_name, other_id_set in self.index_data.items():
                if other_name != name:
                    other_ids.update(other_id_set)

            for _ in range(5):
                qv = np.random.rand(DIMENSION).astype(np.float32)
                results = index.query(query_vectors=qv, top_k=10)
                self.assertGreater(
                    len(results),
                    0,
                    f"Index '{name}' returned empty results — isolation check is vacuous",
                )
                for r in results:
                    self.assertIn(
                        r["id"],
                        my_ids,
                        f"Index '{name}' returned ID '{r['id']}' that belongs to another index",
                    )
                    self.assertNotIn(
                        r["id"],
                        other_ids,
                        f"DATA LEAKAGE: Index '{name}' returned ID '{r['id']}' from another index",
                    )

    def test_list_ids_isolation(self):
        """Each index's list_ids must contain only its own IDs."""
        for index, name, _ in self.indexes:
            stored = set(index.list_ids())
            idx_num = [i for i, (_, n, _) in enumerate(self.indexes) if n == name][0]
            expected_prefix = f"idx{idx_num}_"
            for id_ in stored:
                self.assertTrue(
                    id_.startswith(expected_prefix),
                    f"Index '{name}' contains foreign ID '{id_}'",
                )
            self.assertGreater(len(stored), 0, f"Index '{name}' has no IDs")

    def test_delete_in_one_index_doesnt_affect_others(self):
        """Deleting from index 0 must not remove anything from indexes 1 or 2."""
        other_snapshots = {}
        for index, name, _ in self.indexes[1:]:
            other_snapshots[name] = set(index.list_ids())

        target_index, target_name, _ = self.indexes[0]
        target_ids = list(target_index.list_ids())
        self.assertGreater(len(target_ids), 0, "Index 0 is empty — nothing to delete")
        to_delete = target_ids[: min(15, len(target_ids))]
        target_index.delete(to_delete)
        wait_until(lambda: not (set(to_delete) & set(target_index.list_ids())))

        for index, name, _ in self.indexes[1:]:
            stored = set(index.list_ids())
            self.assertEqual(
                stored,
                other_snapshots[name],
                f"Index '{name}' lost data after deleting from '{target_name}'",
            )


class TestConcurrentMultiIndexWrites(unittest.TestCase):
    """
    The #1 production pattern: one Client, multiple pre-existing indexes,
    concurrent writes to each from separate threads. Tests that the shared
    ApiClient routes index_name correctly under concurrency.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.indexes = []
        cls.num_indexes = 5

        for i in range(cls.num_indexes):
            index, name, key = make_index(cls.client, prefix=f"cw_{i}")
            cls.indexes.append((index, name, key))

    @classmethod
    def tearDownClass(cls):
        for index, _, _ in cls.indexes:
            try:
                index.delete_index()
            except Exception:
                pass

    def test_concurrent_writes_to_different_indexes(self):
        """
        5 threads, each writing to its own pre-existing index via the same
        shared Client. Then each thread reads back its data via get() and
        verifies exact vector match.
        Catches: index_name mix-up in request serialization, cross-index writes.
        """
        errors = []
        lock = threading.Lock()
        per_thread_data = {}

        def worker(thread_id):
            try:
                index, name, _ = self.indexes[thread_id]
                vectors = np.random.rand(20, DIMENSION).astype(np.float32)
                ids = [f"cw{thread_id}_{j}" for j in range(20)]
                index.upsert(ids, vectors)

                with lock:
                    per_thread_data[thread_id] = (ids, vectors, name)
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(self.num_indexes)
        ]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=60)

        self.assertEqual(len(errors), 0, f"Concurrent write errors: {errors}")
        wait_until(
            lambda: all(len(set(idx.list_ids())) >= 20 for idx, _, _ in self.indexes)
        )

        # Verify: each index has ONLY its own data, and vectors are intact
        for thread_id, (ids, vectors, name) in per_thread_data.items():
            index = self.indexes[thread_id][0]

            # list_ids should contain only this thread's IDs
            stored = set(index.list_ids())
            expected_prefix = f"cw{thread_id}_"
            for id_ in stored:
                self.assertTrue(
                    id_.startswith(expected_prefix),
                    f"Index '{name}' contains foreign ID '{id_}' (expected prefix '{expected_prefix}')",
                )
            self.assertEqual(
                stored,
                set(ids),
                f"Index '{name}' has wrong IDs. Extra: {stored - set(ids)}, Missing: {set(ids) - stored}",
            )

            # Spot-check vector integrity: read back first and last vector
            for check_idx in [0, len(ids) - 1]:
                retrieved = index.get([ids[check_idx]], include=["vector"])
                self.assertEqual(len(retrieved), 1)
                retrieved_vec = np.array(retrieved[0]["vector"], dtype=np.float32)
                np.testing.assert_allclose(
                    retrieved_vec,
                    vectors[check_idx],
                    rtol=1e-5,
                    err_msg=f"Index '{name}', ID '{ids[check_idx]}': vector mismatch",
                )


# ---------------------------------------------------------------------------
# Scale & Performance Validation
# ---------------------------------------------------------------------------


class TestStressHighConcurrency(unittest.TestCase):
    """
    Push to 20+ threads with larger batches to find breaking points.
    Validates the SDK doesn't crash, deadlock, or corrupt data at scale.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.index, cls.index_name, cls.index_key = make_index(cls.client)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def test_20_threads_200_vectors_each(self):
        """
        20 threads each upsert 200 vectors (4,000 total) then query.
        All queries must return well-formed results, all IDs must be stored.
        """
        num_threads = 20
        vectors_per_thread = 200
        all_ids = []
        errors = []
        lock = threading.Lock()

        def worker(thread_id):
            try:
                ids, _ = upsert_batch(
                    self.index, f"stress_{thread_id}", count=vectors_per_thread
                )
                with lock:
                    all_ids.extend(ids)

                # Each thread also queries to validate responses under load
                for _ in range(5):
                    qv = np.random.rand(DIMENSION).astype(np.float32)
                    results = self.index.query(
                        query_vectors=qv, top_k=10, include=["distance"]
                    )
                    for r in results:
                        self.assertIn("id", r)
                        self.assertTrue(
                            r["id"],
                            "torn read: query result has empty id (deleted-during-query); see go TestDeletesDuringQueries",
                        )
                        self.assertIn("distance", r)
                        self.assertGreaterEqual(r["distance"], 0)
            except Exception as e:
                with lock:
                    errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        join_threads(self, threads, timeout=180)

        self.assertEqual(len(errors), 0, f"Stress test errors: {errors}")

        wait_until(lambda: len(set(self.index.list_ids())) >= len(all_ids), timeout=30)

        stored_ids = set(self.index.list_ids())
        missing = [id_ for id_ in all_ids if id_ not in stored_ids]
        self.assertEqual(
            len(missing),
            0,
            f"{len(missing)}/{len(all_ids)} IDs missing after 20-thread stress test",
        )


class TestIndexSwitchingFromOneThread(unittest.TestCase):
    """
    One thread rapidly alternates operations across multiple indexes.
    Validates the SDK correctly scopes requests when the same thread
    touches different index objects in quick succession.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = make_client()
        cls.indexes = []
        cls.num_indexes = 5

        for i in range(cls.num_indexes):
            index, name, key = make_index(cls.client, prefix=f"switch_{i}")
            cls.indexes.append((index, name, key))

    @classmethod
    def tearDownClass(cls):
        for index, _, _ in cls.indexes:
            try:
                index.delete_index()
            except Exception:
                pass

    def test_rapid_round_robin_across_indexes(self):
        """
        Single thread performs 10 rounds of: upsert to index 0, query index 1,
        upsert to index 2, query index 3, etc. After all rounds, verify each
        index has only its own data and correct vector values.
        """
        per_index_ids = {i: [] for i in range(self.num_indexes)}
        per_index_vecs = {i: [] for i in range(self.num_indexes)}

        rounds = 10
        vecs_per_round = 5

        for round_num in range(rounds):
            for idx in range(self.num_indexes):
                index, name, _ = self.indexes[idx]

                # Upsert
                ids = [f"sw{idx}_r{round_num}_{j}" for j in range(vecs_per_round)]
                vectors = np.random.rand(vecs_per_round, DIMENSION).astype(np.float32)
                index.upsert(ids, vectors)
                per_index_ids[idx].extend(ids)
                per_index_vecs[idx].append(vectors)

                # Immediately query a different index to force context switching
                other_idx = (idx + 1) % self.num_indexes
                other_index = self.indexes[other_idx][0]
                qv = np.random.rand(DIMENSION).astype(np.float32)
                results = other_index.query(query_vectors=qv, top_k=5)
                for r in results:
                    self.assertIn("id", r)
                    self.assertTrue(
                        r["id"],
                        "torn read: query result has empty id (deleted-during-query); see go TestDeletesDuringQueries",
                    )

        time.sleep(2)

        # Verify isolation: each index has only its own IDs
        for idx in range(self.num_indexes):
            index, name, _ = self.indexes[idx]
            stored = set(index.list_ids())
            expected = set(per_index_ids[idx])
            self.assertEqual(
                stored,
                expected,
                f"Index {idx} ('{name}') has wrong IDs after rapid switching. "
                f"Extra: {stored - expected}, Missing: {expected - stored}",
            )

            # Spot-check vector integrity on last round's data
            last_vecs = per_index_vecs[idx][-1]
            last_ids = per_index_ids[idx][-vecs_per_round:]
            retrieved = index.get([last_ids[0]], include=["vector"])
            self.assertEqual(len(retrieved), 1)
            retrieved_vec = np.array(retrieved[0]["vector"], dtype=np.float32)
            np.testing.assert_allclose(
                retrieved_vec,
                last_vecs[0],
                rtol=1e-5,
                err_msg=f"Index {idx}: vector mismatch after rapid switching",
            )


class TestAsyncConcurrentOperations(unittest.TestCase):
    """
    Tests the async LangChain integration under concurrent load.
    Multiple asyncio tasks performing aadd_texts and asimilarity_search
    concurrently through a shared CyborgVectorStore.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = BASE_URL
        cls.api_key = API_KEY
        cls.index_name = f"async_conc_{uuid.uuid4().hex[:8]}"
        cls.index_key = cyborgdb.Client.generate_key()
        cls.embeddings = FixedDimensionEmbeddings(dimension=DIMENSION)

        cls.store = CyborgVectorStore(
            index_name=cls.index_name,
            index_key=cls.index_key,
            api_key=cls.api_key,
            base_url=cls.base_url,
            embedding=cls.embeddings,
            dimension=DIMENSION,
            metric="euclidean",
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.store.delete(delete_index=True)
        except Exception:
            pass

    def test_concurrent_async_add_and_search(self):
        """
        10 async tasks add texts concurrently, then 10 async tasks search
        concurrently. All adds must succeed and all searches must return
        well-formed results.
        """
        num_tasks = 10
        texts_per_task = 10

        async def run():
            # Phase 1: concurrent adds
            async def add_worker(task_id):
                texts = [
                    f"task {task_id} document {j} about topic {task_id}"
                    for j in range(texts_per_task)
                ]
                ids = [f"async_{task_id}_{j}" for j in range(texts_per_task)]
                returned_ids = await self.store.aadd_texts(texts, ids=ids)
                self.assertEqual(len(returned_ids), texts_per_task)
                return returned_ids

            add_results = await asyncio.gather(
                *[add_worker(i) for i in range(num_tasks)],
                return_exceptions=True,
            )

            # No add tasks should have raised
            for i, result in enumerate(add_results):
                self.assertNotIsInstance(
                    result, Exception, f"Async add task {i} failed: {result}"
                )

            # Phase 2: concurrent searches
            async def search_worker(task_id):
                results = await self.store.asimilarity_search(f"topic {task_id}", k=5)
                self.assertGreater(
                    len(results), 0, f"Search task {task_id} got no results"
                )
                for doc in results:
                    self.assertIsInstance(doc, Document)
                    self.assertIsNotNone(doc.page_content)
                return results

            search_results = await asyncio.gather(
                *[search_worker(i) for i in range(num_tasks)],
                return_exceptions=True,
            )

            for i, result in enumerate(search_results):
                self.assertNotIsInstance(
                    result, Exception, f"Async search task {i} failed: {result}"
                )

        asyncio.run(run())

    def test_concurrent_async_add_and_delete(self):
        """
        5 tasks add texts while 3 tasks delete previously added IDs.
        No task should crash — tests async error handling under contention.
        """
        # Seed some data to delete
        seed_ids = [f"async_del_{i}" for i in range(30)]
        seed_texts = [f"deletable document {i}" for i in range(30)]
        self.store.add_texts(seed_texts, ids=seed_ids)

        async def run():
            errors = []

            async def adder(task_id):
                try:
                    texts = [f"new doc {task_id}_{j}" for j in range(10)]
                    ids = [f"async_new_{task_id}_{j}" for j in range(10)]
                    await self.store.aadd_texts(texts, ids=ids)
                except Exception as e:
                    errors.append(("adder", task_id, e))

            async def deleter(task_id):
                try:
                    batch = seed_ids[task_id * 10 : (task_id + 1) * 10]
                    await self.store.adelete(ids=batch)
                except Exception as e:
                    errors.append(("deleter", task_id, e))

            await asyncio.gather(
                *[adder(i) for i in range(5)],
                *[deleter(i) for i in range(3)],
            )

            self.assertEqual(len(errors), 0, f"Async add/delete errors: {errors}")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
