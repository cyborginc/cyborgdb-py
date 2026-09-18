"""TLS certificate verification on `Client`.

Mirrors js src/__tests__/ssl-verification.test.ts, which had 15 tests against
Python's two incidental mentions.

`verify_ssl` resolves through a decision tree rather than a simple assignment:
an `http://` URL forces it off, an unset value is auto-detected from the host,
and an explicit value otherwise wins. These pin each branch, and the boundary
between "local development convenience" and "a host that merely looks local" —
which is where cyborgdb-core#2399 lives.

Offline: constructing a Client performs no I/O, so none of this needs a service.
"""

import logging
import unittest

import cyborgdb

LOCAL_HOSTS = ["https://localhost:8000", "https://127.0.0.1:8000"]

# Hosts that contain a local-looking substring but are not local. Every one is
# a domain an attacker could register, or a legitimate production URL.
LOOKALIKE_HOSTS = [
    "https://localhost.evil.com",
    "https://127.0.0.1.evil.com",
    "https://notlocalhost.example.com",
    "https://my-localhost-proxy.example.com",
    "https://api.example.com/?region=localhost",
]


def verify_ssl_for(base_url, **kwargs):
    return cyborgdb.Client(base_url=base_url, api_key="k", **kwargs).config.verify_ssl


class TestSSLAutoDetection(unittest.TestCase):
    """`verify_ssl` unset — the default path, resolved from the URL."""

    def test_remote_https_verifies(self):
        self.assertTrue(verify_ssl_for("https://api.example.com"))

    def test_localhost_skips_verification(self):
        for url in LOCAL_HOSTS:
            with self.subTest(url=url):
                self.assertFalse(verify_ssl_for(url))

    def test_http_never_verifies(self):
        # No TLS to verify on a plaintext URL.
        self.assertFalse(verify_ssl_for("http://api.example.com"))
        self.assertFalse(verify_ssl_for("http://localhost:8000"))

    def test_lookalike_hosts_still_verify(self):
        # SECURITY BUG — fails today. cyborgdb-core#2399: the host check is a
        # substring match over the whole URL, so any of these silently connects
        # without verifying the server certificate.
        for url in LOOKALIKE_HOSTS:
            with self.subTest(url=url):
                self.assertTrue(
                    verify_ssl_for(url),
                    f"{url} is not a local host; TLS must still be verified",
                )


class TestSSLExplicitConfiguration(unittest.TestCase):
    """`verify_ssl` passed explicitly."""

    def test_explicit_true_on_remote_host(self):
        self.assertTrue(verify_ssl_for("https://api.example.com", verify_ssl=True))

    def test_explicit_false_on_remote_host(self):
        self.assertFalse(verify_ssl_for("https://api.example.com", verify_ssl=False))

    def test_explicit_true_overrides_local_auto_detection(self):
        # Auto-detection is a convenience, not a ceiling: asking for
        # verification against a local host must be honoured.
        for url in LOCAL_HOSTS:
            with self.subTest(url=url):
                self.assertTrue(verify_ssl_for(url, verify_ssl=True))

    def test_explicit_false_on_local_host(self):
        for url in LOCAL_HOSTS:
            with self.subTest(url=url):
                self.assertFalse(verify_ssl_for(url, verify_ssl=False))

    def test_http_overrides_an_explicit_true(self):
        # Current behaviour, pinned: there is no TLS on a plaintext URL, so the
        # explicit request is discarded. cyborgdb-core#2399 asks whether this
        # should warn rather than pass silently.
        self.assertFalse(verify_ssl_for("http://api.example.com", verify_ssl=True))


class TestSSLWarnings(unittest.TestCase):
    """Disabling a security control must be visible in the log."""

    def test_explicit_disable_warns(self):
        with self.assertLogs("cyborgdb", level=logging.WARNING) as captured:
            verify_ssl_for("https://api.example.com", verify_ssl=False)
        self.assertTrue(
            any("SSL verification is disabled" in line for line in captured.output),
            captured.output,
        )

    def test_auto_disable_is_announced(self):
        # Auto-detected localhost currently logs at INFO; explicit disable logs
        # at WARNING. Either level is defensible — silence is not.
        with self.assertLogs("cyborgdb", level=logging.INFO) as captured:
            verify_ssl_for("https://localhost:8000")
        self.assertTrue(
            any("SSL verification disabled" in line for line in captured.output),
            captured.output,
        )


class TestSSLDoesNotDisturbTheRestOfTheClient(unittest.TestCase):
    """The SSL branch runs before auth and host setup; neither may be affected."""

    def test_api_key_survives_each_ssl_branch(self):
        for url, kwargs in [
            ("https://api.example.com", {}),
            ("https://localhost:8000", {}),
            ("http://api.example.com", {}),
            ("https://api.example.com", {"verify_ssl": False}),
        ]:
            with self.subTest(url=url, kwargs=kwargs):
                client = cyborgdb.Client(base_url=url, api_key="secret", **kwargs)
                self.assertEqual(client.config.api_key, {"X-API-Key": "secret"})

    def test_base_url_is_preserved_verbatim(self):
        for url in [
            "https://api.example.com",
            "https://localhost:8000/",
            "http://x.io",
        ]:
            with self.subTest(url=url):
                self.assertEqual(
                    cyborgdb.Client(base_url=url, api_key="k").config.host, url
                )

    def test_client_without_api_key_sets_no_auth_header(self):
        client = cyborgdb.Client(base_url="https://api.example.com")
        self.assertFalse(getattr(client.config, "api_key", None))


if __name__ == "__main__":
    unittest.main()
