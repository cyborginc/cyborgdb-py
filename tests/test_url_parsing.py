"""Unit tests for URL-parsing and SSL-verification logic in Client.__init__."""

import logging
import unittest
from unittest.mock import MagicMock, patch


def _make_client(base_url, **kwargs):
    with patch(
        "cyborgdb.openapi_client.api_client.ApiClient"
    ) as mock_api_client, patch("cyborgdb.openapi_client.api.default_api.DefaultApi"):
        mock_api_client.return_value = MagicMock()
        from cyborgdb.client.client import Client

        return Client(base_url, **kwargs)


class TestUrlParsing(unittest.TestCase):
    # Case 1: non-loopback HTTPS URL
    def test_non_loopback_https(self):
        client = _make_client("https://api.example.com", api_key="k")
        self.assertIs(client.config.verify_ssl, True)

    # Case 2: loopback by hostname
    def test_loopback_localhost(self):
        client = _make_client("https://localhost:8000", api_key="k")
        self.assertIs(client.config.verify_ssl, False)

    # Case 3: loopback by IPv4
    def test_loopback_ipv4(self):
        client = _make_client("http://127.0.0.1:8080", api_key="k")
        self.assertIs(client.config.verify_ssl, False)

    # Case 4: loopback by IPv6 via http (takes http:// short-circuit)
    def test_loopback_ipv6(self):
        client = _make_client("http://[::1]:8080", api_key="k")
        self.assertIs(client.config.verify_ssl, False)

    # Case 4b: loopback by IPv6 via https — exercises the "::1" auto-detect branch
    def test_loopback_ipv6_https_autodetect(self):
        with self.assertLogs("cyborgdb.client.client", level=logging.WARNING):
            client = _make_client("https://[::1]:8080", api_key="k")
        self.assertIs(client.config.verify_ssl, False)

    # Case 5: hostname suffix includes "localhost" — primary regression test
    def test_localhost_suffix_not_loopback(self):
        client = _make_client("https://localhost.evil.com", api_key="k")
        self.assertIs(client.config.verify_ssl, True)

    # Case 6: hostname prefix containing "localhost"
    def test_localhost_prefix_not_loopback(self):
        client = _make_client("https://notlocalhost.example.com", api_key="k")
        self.assertIs(client.config.verify_ssl, True)

    # Case 7: "127.0.0.1" in query string only
    def test_ipv4_in_query_string(self):
        client = _make_client("https://myhost.com/?x=127.0.0.1", api_key="k")
        self.assertIs(client.config.verify_ssl, True)

    # Case 8: "127.0.0.1" as a hostname suffix
    def test_ipv4_suffix_not_loopback(self):
        client = _make_client("https://127.0.0.1.evil.com", api_key="k")
        self.assertIs(client.config.verify_ssl, True)

    # Case 9: explicit verify_ssl=True overrides auto-detect
    def test_explicit_verify_ssl_true_overrides_autodetect(self):
        client = _make_client("https://localhost:8000", api_key="k", verify_ssl=True)
        self.assertIs(client.config.verify_ssl, True)

    # Case 10: explicit verify_ssl=False
    def test_explicit_verify_ssl_false(self):
        client = _make_client("https://api.example.com", api_key="k", verify_ssl=False)
        self.assertIs(client.config.verify_ssl, False)

    # Case 11: http:// with default verify_ssl — no warning about "no effect"
    def test_http_default_verify_ssl_no_warning(self):
        with self.assertLogs("cyborgdb.client.client", level=logging.WARNING) as log:
            client = _make_client("http://localhost:8000", api_key="k")
        # http:// with verify_ssl=None emits "SSL verification is disabled" (explicit-False
        # branch), not the "no effect" warning which only fires for explicit verify_ssl=True.
        no_effect_msgs = [m for m in log.output if "no effect" in m]
        self.assertEqual(no_effect_msgs, [])
        self.assertIs(client.config.verify_ssl, False)

    # Case 12: http:// with explicit verify_ssl=True — warning emitted
    def test_http_explicit_verify_ssl_true_warns(self):
        with self.assertLogs("cyborgdb.client.client", level=logging.WARNING) as log:
            client = _make_client("http://localhost:8000", api_key="k", verify_ssl=True)
        self.assertIs(client.config.verify_ssl, False)
        # Both "no effect" and "SSL verification is disabled" warnings fire for this URL;
        # filter to the one under test.
        matching = [m for m in log.output if "no effect" in m and "http://" in m]
        self.assertTrue(
            matching, f"Expected 'no effect'+'http://' warning; got: {log.output}"
        )

    # Case 13: auto-detect emits logger.warning (not logger.info)
    def test_autodetect_uses_warning_level(self):
        # Use INFO level so a regression to INFO would still be captured and caught below.
        with self.assertLogs("cyborgdb.client.client", level=logging.INFO) as log:
            _make_client("https://localhost:8000", api_key="k")
        loopback_msgs = [m for m in log.output if "auto-disabled" in m]
        self.assertEqual(
            len(loopback_msgs),
            1,
            f"Expected exactly one auto-detect log; got: {log.output}",
        )
        self.assertTrue(
            loopback_msgs[0].startswith("WARNING:"),
            f"Expected WARNING level; got: {loopback_msgs[0]}",
        )

    # Case 14: auto-detect warning includes the matched hostname
    def test_autodetect_warning_includes_hostname(self):
        with self.assertLogs("cyborgdb.client.client", level=logging.WARNING) as log:
            _make_client("https://localhost:8000", api_key="k")
        combined = " ".join(log.output)
        self.assertIn("localhost", combined)

    # Case 15: unparseable URL — no crash, verify_ssl=True
    def test_unparseable_url_no_crash(self):
        client = _make_client("not-a-url", api_key="k")
        self.assertIs(client.config.verify_ssl, True)


if __name__ == "__main__":
    unittest.main()
