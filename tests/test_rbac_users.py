"""RBAC user-management integration tests for the CyborgDB Python SDK.

These exercise the user-key lifecycle the service exposes when it runs with
``CYBORGDB_ROOT_API_KEY`` set (RBAC enabled, see the service's ``rbac.md``):

  * the **root** client mints per-user API keys with
    ``EncryptedIndex.create_user(permissions=[...])``;
  * a **user** client authenticates with the returned ``cdbk_`` key and is
    confined to that one index with ``read`` / ``write`` permissions enforced
    *cryptographically* by the service — the wrapped data-encryption keys that
    exist for the user ARE the permission set, so a read-only user simply
    cannot decrypt for a write op;
  * ``list_users`` / ``delete_user`` let the root enumerate and revoke; after a
    delete the user's key stops working immediately.

User keys resolve the index KEK server-side, so they only work against
**KMS-backed** indexes. The suite is therefore gated on both the root key and
a KMS registry slot:

  - CYBORGDB_ROOT_API_KEY — the service's admin key (RBAC must be enabled).
  - CYBORGDB_KMS_NAME     — a kms.registry slot the service can use to wrap the
                            per-index KEK (e.g. the same value used by the KMS
                            BYOK suite).

Run a service with both configured, point CYBORGDB_BASE_URL at it, and these
run live; otherwise they skip.
"""

import os
import unittest
import uuid

import numpy as np
from dotenv import load_dotenv

import cyborgdb


load_dotenv(".env.local")

# The e2e nightly sets CYBORGDB_URL; the KMS BYOK suite uses CYBORGDB_BASE_URL.
# Accept either so this runs unchanged in both places.
BASE_URL = (
    os.getenv("CYBORGDB_URL") or os.getenv("CYBORGDB_BASE_URL") or "http://localhost:8000"
)
ROOT_API_KEY = os.getenv("CYBORGDB_ROOT_API_KEY")
KMS_NAME = os.getenv("CYBORGDB_KMS_NAME")

DIMENSION = 4


def _seed():
    return [
        {"id": "a", "vector": [0.1, 0.2, 0.3, 0.4]},
        {"id": "b", "vector": [0.9, 0.8, 0.7, 0.6]},
    ]


@unittest.skipUnless(
    ROOT_API_KEY and KMS_NAME,
    "set CYBORGDB_ROOT_API_KEY and CYBORGDB_KMS_NAME against an RBAC-enabled service",
)
class RBACUserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = cyborgdb.Client(BASE_URL, api_key=ROOT_API_KEY)
        cls.index_name = f"rbac_users_test_{uuid.uuid4().hex[:8]}"
        # KMS-backed so user keys can resolve the index KEK server-side.
        cls.index = cls.root.create_index(
            index_name=cls.index_name, kms_name=KMS_NAME, dimension=DIMENSION
        )
        cls.index.upsert(_seed())

    @classmethod
    def tearDownClass(cls):
        try:
            cls.index.delete_index()
        except Exception:
            pass

    def _user_index(self, api_key):
        """Load this index as a user (no index_key — service resolves it)."""
        return cyborgdb.Client(BASE_URL, api_key=api_key).load_index(self.index_name)

    def test_create_returns_key_and_id(self):
        out = self.index.create_user(permissions=["read"])
        self.assertIn("api_key", out)
        self.assertIn("user_id", out)
        self.assertTrue(out["api_key"].startswith("cdbk_"))
        # Cleanup so list assertions elsewhere stay deterministic.
        self.index.delete_user(out["user_id"])

    def test_read_only_user_can_query_but_not_write(self):
        out = self.index.create_user(permissions=["read"])
        try:
            reader = self._user_index(out["api_key"])
            # read op succeeds
            results = reader.query(query_vectors=[0.1, 0.2, 0.3, 0.4], top_k=1)
            self.assertTrue(len(results) >= 1)
            # write op is cryptographically denied
            with self.assertRaises(ValueError):
                reader.upsert([{"id": "z", "vector": [0.0, 0.0, 0.0, 1.0]}])
        finally:
            self.index.delete_user(out["user_id"])

    def test_read_write_user_can_do_both(self):
        out = self.index.create_user(permissions=["read", "write"])
        try:
            writer = self._user_index(out["api_key"])
            writer.upsert([{"id": "w", "vector": [0.0, 0.0, 0.0, 1.0]}])
            results = writer.query(query_vectors=[0.0, 0.0, 0.0, 1.0], top_k=1)
            self.assertTrue(len(results) >= 1)
        finally:
            self.index.delete_user(out["user_id"])

    def test_list_then_revoke(self):
        out = self.index.create_user(permissions=["read", "write"])
        users = self.index.list_users()
        self.assertIn(out["user_id"], {u["user_id"] for u in users})
        listed = next(u for u in users if u["user_id"] == out["user_id"])
        self.assertEqual(sorted(listed["permissions"]), ["read", "write"])

        # Revoke; the key must stop working on the next request.
        self.index.delete_user(out["user_id"])
        self.assertNotIn(
            out["user_id"], {u["user_id"] for u in self.index.list_users()}
        )
        revoked = self._user_index(out["api_key"])
        with self.assertRaises(ValueError):
            revoked.query(query_vectors=[0.1, 0.2, 0.3, 0.4], top_k=1)


if __name__ == "__main__":
    unittest.main()
