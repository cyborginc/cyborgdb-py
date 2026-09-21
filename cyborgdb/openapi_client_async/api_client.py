"""
    CyborgDB Service — Async API client wrapper

    REST API for CyborgDB: The Confidential Vector Database

    Generated via: openapi-generator-cli generate -i openapi.json -g python
      -o . --package-name cyborgdb.openapi_client_async
      --additional-properties=generateSourceCodeOnly=true,library=httpx
    (generator version 7.22.0; see README.md for the full command)

    Do not edit the class manually.
"""  # noqa: E501

from cyborgdb.openapi_client.api_client import ApiClient
from cyborgdb.openapi_client.configuration import Configuration
from cyborgdb.openapi_client_async.rest import AsyncRESTClientObject, AsyncRESTResponse  # noqa: F401


class AsyncApiClient(ApiClient):
    """Async wrapper around the generated ApiClient.

    Inherits all serialisation helpers (param_serialize,
    sanitize_for_serialization, response_deserialize, …) from the sync
    ApiClient.  The only override is call_api, which becomes async and
    uses httpx via AsyncRESTClientObject instead of urllib3.

    Models are shared with cyborgdb.openapi_client — the async transport
    never duplicates them.
    """

    def __init__(
        self,
        configuration=None,
        header_name=None,
        header_value=None,
        cookie=None,
    ) -> None:
        if configuration is None:
            configuration = Configuration.get_default()
        # Set the attributes that param_serialize / response_deserialize rely
        # on, but do NOT call super().__init__() to avoid creating the urllib3
        # PoolManager that is unused here.
        self.configuration = configuration
        self.default_headers: dict = {}
        if header_name is not None:
            self.default_headers[header_name] = header_value
        self.cookie = cookie
        # The user_agent property setter stores into default_headers.
        self.user_agent = "CyborgDB-Python-SDK-Async/1.0.0"
        self.client_side_validation = getattr(
            configuration, "client_side_validation", True
        )
        # Async httpx transport
        self.rest_client = AsyncRESTClientObject(configuration)

    async def call_api(
        self,
        method,
        url,
        header_params=None,
        body=None,
        post_params=None,
        _request_timeout=None,
    ) -> AsyncRESTResponse:
        return await self.rest_client.request(
            method,
            url,
            headers=header_params,
            body=body,
            post_params=post_params,
            _request_timeout=_request_timeout,
        )

    async def close(self) -> None:
        await self.rest_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
