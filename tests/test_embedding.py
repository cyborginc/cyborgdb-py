"""Built-in text embedding (cyborgdb-embed).

cyborgdb-core now embeds in C++ via cyborgdb-embed rather than calling
sentence-transformers from Python (cyborgdb-core#2422). cyborgdb-service#271 is
the service half: it drops the sentence-transformers gate, adds
`GET /v1/embedding-models`, and maps core's EmbeddingModelUnavailable to 503
instead of letting it fall through as a 500.

These fail until cyborgdb-service#271 merges. They assert the contract that
PR's own tests assert, so they double as the SDK-side check that it landed.
"""

import os
import unittest
import uuid

import requests
from dotenv import load_dotenv

import cyborgdb
from helpers import wait_for_ids

load_dotenv(".env.local")

BASE_URL = os.getenv("CYBORGDB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("CYBORGDB_API_KEY", "")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIM = 384

# Three distinct topics, so a semantic hit is unambiguous. Ported from
# cyborgdb-core tests/embedding_test.cpp.
CORPUS = {
    "fox": "The quick brown fox jumps over the lazy dog.",
    "revenue": "Quarterly revenue grew eleven percent year over year.",
    "weather": "Heavy rain and strong winds are expected tomorrow.",
}


def _client():
    return cyborgdb.Client(base_url=BASE_URL, api_key=API_KEY)


class TestEmbeddingModelCatalog(unittest.TestCase):
    """`GET /v1/embedding-models`, called directly — the SDK has no wrapper
    for it yet."""

    def test_lists_the_supported_models(self):
        response = requests.get(
            f"{BASE_URL}/v1/embedding-models",
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"expected the catalog endpoint, got {response.status_code}: {response.text}",
        )
        models = {m["name"]: m for m in response.json()["models"]}
        self.assertIn(MODEL, models)
        self.assertEqual(models[MODEL]["dimension"], MODEL_DIM)
        for name, model in models.items():
            with self.subTest(model=name):
                self.assertIsInstance(model["dimension"], int)
                self.assertGreater(model["dimension"], 0)
                self.assertGreater(model["max_seq_length"], 0)


class TestEmbeddingModelValidation(unittest.TestCase):
    def setUp(self):
        self.client = _client()
        self.created = []

    def tearDown(self):
        for index in self.created:
            try:
                index.delete_index()
            except Exception:
                pass

    def _create(self, **kwargs):
        index = self.client.create_index(
            f"embed_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            **kwargs,
        )
        self.created.append(index)
        return index

    def test_bare_mixed_case_name_is_accepted(self):
        # cyborgdb-embed accepts the name with or without the org prefix.
        index = self._create(embedding_model="ALL-MINILM-L6-V2")
        self.assertEqual(index.dimension, MODEL_DIM)

    def test_full_name_is_accepted(self):
        index = self._create(embedding_model=MODEL)
        self.assertEqual(index.dimension, MODEL_DIM)

    def test_unknown_model_is_a_client_error(self):
        # Before #271 this was a 500, which tells a caller to retry something
        # that can never succeed.
        with self.assertRaises(cyborgdb.ValidationError):
            self._create(embedding_model="not-a-real-model")

    def test_openai_style_name_is_rejected(self):
        with self.assertRaises(cyborgdb.ValidationError):
            self._create(embedding_model="text-embedding-3-small")

    def test_dimension_contradicting_the_model_is_rejected(self):
        with self.assertRaises(cyborgdb.ValidationError):
            self._create(embedding_model=MODEL, dimension=MODEL_DIM + 1)

    def test_dimension_matching_the_model_is_accepted(self):
        # Anchors the test above: the rejection is about the contradiction,
        # not about passing a dimension at all.
        index = self._create(embedding_model=MODEL, dimension=MODEL_DIM)
        self.assertEqual(index.dimension, MODEL_DIM)


class TestEmbeddingRoundTrip(unittest.TestCase):
    """Text in, semantically-related text finds it again."""

    @classmethod
    def setUpClass(cls):
        cls.client = _client()
        cls.index = cls.client.create_index(
            f"embed_rt_{uuid.uuid4().hex[:8]}",
            cyborgdb.Client.generate_key(),
            embedding_model=MODEL,
        )
        cls.index.upsert(
            [
                {"id": doc_id, "contents": text}
                for doc_id, text in CORPUS.items()
            ]
        )
        wait_for_ids(cls.index, list(CORPUS))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _top_hit(self, text):
        results = self.index.query(query_contents=text, top_k=1)
        return results[0]["id"] if results else None

    def test_paraphrase_finds_the_right_document(self):
        # Neither query shares a distinctive word with its target, so a match
        # has to come from the embedding rather than lexical overlap.
        self.assertEqual(self._top_hit("a fox leaping over a sleepy dog"), "fox")
        self.assertEqual(self._top_hit("company earnings this quarter"), "revenue")
        self.assertEqual(self._top_hit("a storm is coming"), "weather")

    def test_contents_are_stored_alongside_the_vector(self):
        row = self.index.get(["fox"], include=["contents"])[0]
        self.assertEqual(row["contents"], CORPUS["fox"])

    def test_embedded_vectors_have_the_models_dimension(self):
        row = self.index.get(["fox"], include=["vector"])[0]
        self.assertEqual(len(row["vector"]), MODEL_DIM)


if __name__ == "__main__":
    unittest.main()
