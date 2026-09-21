"""
    CyborgDB Service — Async transport configuration

    Thin re-export of the sync Configuration; the same fields
    (host, api_key, verify_ssl, connection_pool_maxsize) are used by
    AsyncRESTClientObject to configure the httpx.Limits pool.

    Do not edit the class manually.
"""  # noqa: E501

from cyborgdb.openapi_client.configuration import Configuration  # noqa: F401
