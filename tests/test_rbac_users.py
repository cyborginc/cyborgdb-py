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
                            BYOK suite). CYBORGDB_KMS_NAME_REAL is accepted as a
                            fallback, matching the e2e nightly's RBAC step.

Run a service with both configured, point CYBORGDB_BASE_URL at it, and these
run live; otherwise they skip.
"""

import os
import unittest
import uuid

from dotenv import load_dotenv

import cyborgdb


load_dotenv(".env.local")

# The e2e nightly sets CYBORGDB_URL; the KMS BYOK suite uses CYBORGDB_BASE_URL.
# Accept either so this runs unchanged in both places.
BASE_URL = (
    os.getenv("CYBORGDB_URL")
    or os.getenv("CYBORGDB_BASE_URL")
    or "http://localhost:8000"
)
ROOT_API_KEY = os.getenv("CYBORGDB_ROOT_API_KEY")
# The e2e nightly's RBAC step exports the slot as CYBORGDB_KMS_NAME_REAL; accept
# the plain CYBORGDB_KMS_NAME too so this runs unchanged in either setup.
KMS_NAME = os.getenv("CYBORGDB_KMS_NAME") or os.getenv("CYBORGDB_KMS_NAME_REAL")

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

    def test_read_only_user_lists_with_read_permission(self):
        out = self.index.create_user(permissions=["read"])
        try:
            users = self.index.list_users()
            listed = next(u for u in users if u["user_id"] == out["user_id"])
            self.assertEqual(listed["permissions"], ["read"])
        finally:
            self.index.delete_user(out["user_id"])

    def test_write_only_user_can_write_but_not_query(self):
        out = self.index.create_user(permissions=["write"])
        try:
            writer = self._user_index(out["api_key"])
            # write op succeeds
            writer.upsert([{"id": "wo", "vector": [0.0, 0.0, 1.0, 0.0]}])
            # read op is cryptographically denied — no read DEK for this user
            with self.assertRaises(ValueError):
                writer.query(query_vectors=[0.0, 0.0, 1.0, 0.0], top_k=1)
        finally:
            self.index.delete_user(out["user_id"])

    def test_invalid_permissions_rejected(self):
        # The grant must be a non-empty subset of {"read", "write"}; the
        # service rejects an empty set and unknown permission names alike.
        with self.assertRaises(ValueError):
            self.index.create_user(permissions=[])
        with self.assertRaises(ValueError):
            self.index.create_user(permissions=["admin"])

    def test_non_root_user_cannot_manage_users(self):
        out = self.index.create_user(permissions=["read", "write"])
        try:
            user_index = self._user_index(out["api_key"])
            # Minting, listing, and revoking users are root-only operations;
            # a user key is rejected on each.
            with self.assertRaises(ValueError):
                user_index.create_user(permissions=["read"])
            with self.assertRaises(ValueError):
                user_index.list_users()
            with self.assertRaises(ValueError):
                user_index.delete_user(out["user_id"])
        finally:
            self.index.delete_user(out["user_id"])

    def test_revoking_one_user_leaves_another_working(self):
        first = self.index.create_user(permissions=["read"])
        second = self.index.create_user(permissions=["read"])
        try:
            # Revoking one user drops only that user's wrapped keys; the
            # other user's key keeps resolving the index.
            self.index.delete_user(first["user_id"])
            survivor = self._user_index(second["api_key"])
            results = survivor.query(query_vectors=[0.1, 0.2, 0.3, 0.4], top_k=1)
            self.assertTrue(len(results) >= 1)
        finally:
            for u in (first, second):
                try:
                    self.index.delete_user(u["user_id"])
                except Exception:
                    pass

    def test_list_then_revoke(self):
        out = self.index.create_user(permissions=["read", "write"])
        users = self.index.list_users()
        self.assertIn(out["user_id"], {u["user_id"] for u in users})
        listed = next(u for u in users if u["user_id"] == out["user_id"])
        self.assertEqual(sorted(listed["permissions"]), ["read", "write"])

        # Revoke; the key must stop working immediately. The service drops the
        # user's wrapped DEK, so even loading the index (which describes it,
        # gated by the user-wrap check) is denied — hence load or query may
        # raise.
        self.index.delete_user(out["user_id"])
        self.assertNotIn(
            out["user_id"], {u["user_id"] for u in self.index.list_users()}
        )
        with self.assertRaises(ValueError):
            revoked = self._user_index(out["api_key"])
            revoked.query(query_vectors=[0.1, 0.2, 0.3, 0.4], top_k=1)


if __name__ == "__main__":
    unittest.main()
