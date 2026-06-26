"""Query rerank_mult knob (0.17.0 API). Mirrors go rerank_test.go and
js rerank.test.ts: rerank_mult is the stage-1 retrieval multiplier for
reranking indexes — optional, with a server-side default when unset. This
verifies the SDK threads the value into the request and the server accepts it
on a standard query.
"""

import os
import time
import unittest
import uuid

import numpy as np
from dotenv import load_dotenv

import cyborgdb

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")
DIM = 8


class TestQueryWithRerankMult(unittest.TestCase):
    def setUp(self):
        self.client = cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)
        self.index = self.client.create_index(
            f"rerank_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            dimension=DIM,
            metric="euclidean",
        )
        vectors = np.random.rand(20, DIM).astype(np.float32)
        ids = [f"rerank_{i}" for i in range(20)]
        self.index.upsert(ids, vectors)
        time.sleep(2)

    def tearDown(self):
        try:
            self.index.delete_index()
        except Exception:
            pass

    def test_query_with_rerank_mult(self):
        qv = np.random.rand(DIM).astype(np.float32)
        results = self.index.query(
            query_vectors=qv, top_k=5, rerank_mult=4, include=["distance"]
        )
        self.assertGreaterEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
