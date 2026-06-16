"""RBAC user-management integration tests for the CyborgDB Python SDK."""

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
# KMS slot names mirror test_kms_byok.py. CYBORGDB_KMS_NAME is the legacy single
# var the nightly used to set — keep it as a fallback for the real-provider slot.
KMS_NAME_REAL = os.getenv("CYBORGDB_KMS_NAME_REAL") or os.getenv("CYBORGDB_KMS_NAME")
KMS_NAME_SM = os.getenv("CYBORGDB_KMS_NAME_SM")

DIMENSION = 4


def _seed():
    return [
        {"id": "a", "vector": [0.1, 0.2, 0.3, 0.4]},
        {"id": "b", "vector": [0.9, 0.8, 0.7, 0.6]},
    ]


class _RBACUserSuite:
    """User-key lifecycle assertions shared across index-key encodings.

    Concrete subclasses implement ``_create_index`` to build the index under
    test (KMS-backed or SDK-supplied key); everything downstream is identical.
    This is a plain mixin (no ``TestCase`` base) so it isn't collected on its
    own.
    """

    @classmethod
    def _create_index(cls):
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        cls.root = cyborgdb.Client(BASE_URL, api_key=ROOT_API_KEY)
        cls.index_name = f"rbac_users_test_{uuid.uuid4().hex[:8]}"
        cls.index = cls._create_index()
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


@unittest.skipUnless(
    ROOT_API_KEY,
    "set CYBORGDB_ROOT_API_KEY against an RBAC-enabled service",
)
class RBACUserSDKKeyTests(_RBACUserSuite, unittest.TestCase):
    """RBAC on an SDK-supplied-key index (no KMS). The root supplies the KEK on
    create and the SDK forwards it to the user-management calls so the DEK can
    be re-wrapped under each user's KEK."""

    @classmethod
    def _create_index(cls):
        return cls.root.create_index(
            index_name=cls.index_name,
            index_key=cyborgdb.Client.generate_key(),
            dimension=DIMENSION,
        )


@unittest.skipUnless(
    ROOT_API_KEY and KMS_NAME_REAL,
    "set CYBORGDB_ROOT_API_KEY and CYBORGDB_KMS_NAME_REAL (aws-kms) against an "
    "RBAC-enabled service",
)
class RBACUserKMSRealTests(_RBACUserSuite, unittest.TestCase):
    """RBAC on an aws-kms (HSM) KMS-backed index; the service resolves the KEK
    server-side on every request."""

    @classmethod
    def _create_index(cls):
        return cls.root.create_index(
            index_name=cls.index_name, kms_name=KMS_NAME_REAL, dimension=DIMENSION
        )


@unittest.skipUnless(
    ROOT_API_KEY and KMS_NAME_SM,
    "set CYBORGDB_ROOT_API_KEY and CYBORGDB_KMS_NAME_SM (aws Secrets Manager) "
    "against an RBAC-enabled service",
)
class RBACUserKMSSecretsTests(_RBACUserSuite, unittest.TestCase):
    """RBAC on an aws (Secrets Manager) KMS-backed index."""

    @classmethod
    def _create_index(cls):
        return cls.root.create_index(
            index_name=cls.index_name, kms_name=KMS_NAME_SM, dimension=DIMENSION
        )


if __name__ == "__main__":
    unittest.main()
