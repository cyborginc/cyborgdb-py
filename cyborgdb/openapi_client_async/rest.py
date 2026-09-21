"""
    CyborgDB Service — Async httpx transport layer

    REST API for CyborgDB: The Confidential Vector Database

    Generated via: openapi-generator-cli generate -i openapi.json -g python
      -o . --package-name cyborgdb.openapi_client_async
      --additional-properties=generateSourceCodeOnly=true,library=httpx
    (generator version 7.22.0; see README.md for the full command)

    Do not edit the class manually.
"""  # noqa: E501

import json

import httpx

from cyborgdb.openapi_client.exceptions import ApiException, ApiValueError  # noqa: F401


class AsyncRESTResponse:
    """Wraps an httpx.Response so it matches the RESTResponse interface
    expected by ApiClient.response_deserialize."""

    def __init__(self, httpx_response: httpx.Response) -> None:
        self.status: int = httpx_response.status_code
        self.data: bytes = httpx_response.content
        self.headers = httpx_response.headers
        self.reason: str = httpx_response.reason_phrase


class AsyncRESTClientObject:
    """httpx-backed async REST transport."""

    def __init__(self, configuration) -> None:
        verify = getattr(configuration, "verify_ssl", True)
        maxsize = getattr(configuration, "connection_pool_maxsize", None) or 100
        limits = httpx.Limits(
            max_connections=maxsize,
            max_keepalive_connections=max(5, maxsize // 4),
        )
        self._default_timeout = httpx.Timeout(connect=5.0, read=60.0)
        self.pool = httpx.AsyncClient(limits=limits, verify=verify)

    async def request(
        self,
        method,
        url,
        headers=None,
        body=None,
        post_params=None,
        _request_timeout=None,
    ) -> AsyncRESTResponse:
        headers = dict(headers) if headers else {}
        content: bytes | None = None

        if body is not None:
            content_type = headers.get("Content-Type", "")
            if not content_type or "json" in content_type.lower():
                content = json.dumps(body).encode("utf-8")
                if not content_type:
                    headers["Content-Type"] = "application/json"
            elif isinstance(body, bytes):
                content = body
            elif isinstance(body, str):
                content = body.encode("utf-8")

        timeout: httpx.Timeout = self._default_timeout
        if _request_timeout:
            if isinstance(_request_timeout, (int, float)):
                timeout = httpx.Timeout(_request_timeout)
            elif isinstance(_request_timeout, tuple) and len(_request_timeout) == 2:
                timeout = httpx.Timeout(
                    connect=_request_timeout[0], read=_request_timeout[1]
                )

        response = await self.pool.request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            timeout=timeout,
        )
        return AsyncRESTResponse(response)

    async def close(self) -> None:
        await self.pool.aclose()
