"""
    CyborgDB Service — Async DefaultApi

    REST API for CyborgDB: The Confidential Vector Database

    Generated via: openapi-generator-cli generate -i openapi.json -g python
      -o . --package-name cyborgdb.openapi_client_async
      --additional-properties=generateSourceCodeOnly=true,library=httpx
    (generator version 7.22.0; see README.md for the full command)

    Do not edit the class manually.
"""  # noqa: E501

from typing import Any, Dict, List, Optional

from cyborgdb.openapi_client.models.binary_query_request import BinaryQueryRequest
from cyborgdb.openapi_client.models.binary_upsert_request import BinaryUpsertRequest
from cyborgdb.openapi_client.models.create_index_request import CreateIndexRequest
from cyborgdb.openapi_client.models.create_user_request import CreateUserRequest
from cyborgdb.openapi_client.models.create_user_response import CreateUserResponse
from cyborgdb.openapi_client.models.delete_request import DeleteRequest
from cyborgdb.openapi_client.models.get_request import GetRequest
from cyborgdb.openapi_client.models.get_response_model import GetResponseModel
from cyborgdb.openapi_client.models.index_info_response_model import (
    IndexInfoResponseModel,
)
from cyborgdb.openapi_client.models.index_list_response_model import (
    IndexListResponseModel,
)
from cyborgdb.openapi_client.models.index_operation_request import (
    IndexOperationRequest,
)
from cyborgdb.openapi_client.models.index_training_status_response_model import (
    IndexTrainingStatusResponseModel,
)
from cyborgdb.openapi_client.models.list_ids_request import ListIDsRequest
from cyborgdb.openapi_client.models.list_ids_response import ListIDsResponse
from cyborgdb.openapi_client.models.list_users_response import ListUsersResponse
from cyborgdb.openapi_client.models.query_metadata_request import QueryMetadataRequest
from cyborgdb.openapi_client.models.query_metadata_response import (
    QueryMetadataResponse,
)
from cyborgdb.openapi_client.models.query_response import QueryResponse
from cyborgdb.openapi_client.models.request import Request
from cyborgdb.openapi_client.models.train_request import TrainRequest
from cyborgdb.openapi_client.models.upsert_request import UpsertRequest

from cyborgdb.openapi_client_async.api_client import AsyncApiClient
from cyborgdb.openapi_client_async.rest import AsyncRESTResponse


class DefaultApi:
    """Async version of the generated DefaultApi.

    Every public method is async and uses the AsyncApiClient (httpx-backed).
    Models are imported from cyborgdb.openapi_client to avoid duplication.
    """

    def __init__(self, api_client: "AsyncApiClient") -> None:
        self.api_client = api_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _serialize(
        self,
        method: str,
        resource_path: str,
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[List] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Any = None,
        auth_settings: Optional[List[str]] = None,
    ):
        return self.api_client.param_serialize(
            method=method,
            resource_path=resource_path,
            path_params=path_params or {},
            query_params=query_params or [],
            header_params=header_params or {},
            body=body,
            post_params=[],
            files={},
            auth_settings=auth_settings or ["APIKeyHeader"],
            collection_formats={},
        )

    async def _call(
        self,
        method: str,
        resource_path: str,
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[List] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Any = None,
        auth_settings: Optional[List[str]] = None,
        _request_timeout=None,
    ) -> AsyncRESTResponse:
        m, url, hdrs, bd, pp = self._serialize(
            method,
            resource_path,
            path_params=path_params,
            query_params=query_params,
            header_params=header_params,
            body=body,
            auth_settings=auth_settings,
        )
        return await self.api_client.call_api(m, url, hdrs, bd, pp, _request_timeout)

    async def _call_and_deserialize(
        self,
        method: str,
        resource_path: str,
        response_types_map: Dict[str, Optional[str]],
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[List] = None,
        header_params: Optional[Dict[str, Any]] = None,
        body: Any = None,
        auth_settings: Optional[List[str]] = None,
        _request_timeout=None,
    ):
        response_data = await self._call(
            method,
            resource_path,
            path_params=path_params,
            query_params=query_params,
            header_params=header_params,
            body=body,
            auth_settings=auth_settings,
            _request_timeout=_request_timeout,
        )
        return self.api_client.response_deserialize(
            response_data=response_data,
            response_types_map=response_types_map,
        ).data

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def list_indexes_v1_indexes_list_get(
        self,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> IndexListResponseModel:
        return await self._call_and_deserialize(
            "GET",
            "/v1/indexes/list",
            {"200": "IndexListResponseModel", "422": "HTTPValidationError"},
            header_params=_headers,
            _request_timeout=_request_timeout,
        )

    async def create_index_v1_indexes_create_post(
        self,
        create_index_request: CreateIndexRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ):
        return await self._call_and_deserialize(
            "POST",
            "/v1/indexes/create",
            {
                "200": "CyborgdbServiceApiSchemasIndexSuccessResponseModel",
                "401": "ErrorResponseModel",
                "409": "ErrorResponseModel",
                "422": "HTTPValidationError",
                "500": "ErrorResponseModel",
            },
            header_params=_headers,
            body=create_index_request,
            _request_timeout=_request_timeout,
        )

    async def get_index_info_v1_indexes_describe_post(
        self,
        index_operation_request: IndexOperationRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> IndexInfoResponseModel:
        return await self._call_and_deserialize(
            "POST",
            "/v1/indexes/describe",
            {
                "200": "IndexInfoResponseModel",
                "401": "ErrorResponseModel",
                "404": "ErrorResponseModel",
                "422": "HTTPValidationError",
            },
            header_params=_headers,
            body=index_operation_request,
            _request_timeout=_request_timeout,
        )

    async def delete_index_v1_indexes_delete_post(
        self,
        index_operation_request: IndexOperationRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> None:
        await self._call_and_deserialize(
            "POST",
            "/v1/indexes/delete",
            {"200": "CyborgdbServiceApiSchemasIndexSuccessResponseModel"},
            header_params=_headers,
            body=index_operation_request,
            _request_timeout=_request_timeout,
        )

    async def train_index_v1_indexes_train_post(
        self,
        train_request: TrainRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> None:
        await self._call_and_deserialize(
            "POST",
            "/v1/indexes/train",
            {"200": "CyborgdbServiceApiSchemasIndexSuccessResponseModel"},
            header_params=_headers,
            body=train_request,
            _request_timeout=_request_timeout,
        )

    async def get_training_status_v1_indexes_training_status_get(
        self,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> IndexTrainingStatusResponseModel:
        return await self._call_and_deserialize(
            "GET",
            "/v1/indexes/training-status",
            {"200": "IndexTrainingStatusResponseModel"},
            header_params=_headers,
            _request_timeout=_request_timeout,
        )

    async def health_check_v1_health_get(
        self,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ):
        return await self._call_and_deserialize(
            "GET",
            "/v1/health",
            {"200": "object"},
            header_params=_headers,
            _request_timeout=_request_timeout,
        )

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    async def upsert_vectors_v1_vectors_upsert_post(
        self,
        upsert_request: UpsertRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> None:
        await self._call_and_deserialize(
            "POST",
            "/v1/vectors/upsert",
            {"200": "CyborgdbServiceApiSchemasVectorsSuccessResponseModel"},
            header_params=_headers,
            body=upsert_request,
            _request_timeout=_request_timeout,
        )

    async def upsert_vectors_binary_v1_vectors_upsert_binary_post(
        self,
        binary_upsert_request: BinaryUpsertRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> None:
        await self._call_and_deserialize(
            "POST",
            "/v1/vectors/upsert_binary",
            {"200": "CyborgdbServiceApiSchemasVectorsSuccessResponseModel"},
            header_params=_headers,
            body=binary_upsert_request,
            _request_timeout=_request_timeout,
        )

    async def delete_vectors_v1_vectors_delete_post(
        self,
        delete_request: DeleteRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> None:
        await self._call_and_deserialize(
            "POST",
            "/v1/vectors/delete",
            {"200": "CyborgdbServiceApiSchemasVectorsSuccessResponseModel"},
            header_params=_headers,
            body=delete_request,
            _request_timeout=_request_timeout,
        )

    async def get_vectors_v1_vectors_get_post(
        self,
        get_request: GetRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> GetResponseModel:
        return await self._call_and_deserialize(
            "POST",
            "/v1/vectors/get",
            {"200": "GetResponseModel", "422": "HTTPValidationError"},
            header_params=_headers,
            body=get_request,
            _request_timeout=_request_timeout,
        )

    async def query_vectors_v1_vectors_query_post(
        self,
        request: Request,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> AsyncRESTResponse:
        """Returns raw AsyncRESTResponse so callers can inspect status and
        parse the body themselves (mirrors the sync _without_preload_content
        variant used by EncryptedIndex.query)."""
        return await self._call(
            "POST",
            "/v1/vectors/query",
            header_params=_headers,
            body=request,
            _request_timeout=_request_timeout,
        )

    async def query_vectors_binary_v1_vectors_query_binary_post(
        self,
        binary_query_request: BinaryQueryRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> QueryResponse:
        return await self._call_and_deserialize(
            "POST",
            "/v1/vectors/query_binary",
            {"200": "QueryResponse", "422": "HTTPValidationError"},
            header_params=_headers,
            body=binary_query_request,
            _request_timeout=_request_timeout,
        )

    async def query_metadata_v1_vectors_query_metadata_post(
        self,
        query_metadata_request: QueryMetadataRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> QueryMetadataResponse:
        return await self._call_and_deserialize(
            "POST",
            "/v1/vectors/query_metadata",
            {"200": "QueryMetadataResponse", "422": "HTTPValidationError"},
            header_params=_headers,
            body=query_metadata_request,
            _request_timeout=_request_timeout,
        )

    async def list_ids_v1_vectors_list_ids_post(
        self,
        list_ids_request: ListIDsRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> ListIDsResponse:
        return await self._call_and_deserialize(
            "POST",
            "/v1/vectors/list_ids",
            {"200": "ListIDsResponse"},
            header_params=_headers,
            body=list_ids_request,
            _request_timeout=_request_timeout,
        )

    # ------------------------------------------------------------------
    # RBAC / user management
    # ------------------------------------------------------------------

    async def create_user_v1_indexes_index_name_users_post(
        self,
        index_name: str,
        create_user_request: CreateUserRequest,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> CreateUserResponse:
        return await self._call_and_deserialize(
            "POST",
            "/v1/indexes/{index_name}/users",
            {"200": "CreateUserResponse", "422": "HTTPValidationError"},
            path_params={"index_name": index_name},
            header_params=_headers,
            body=create_user_request,
            _request_timeout=_request_timeout,
        )

    async def list_users_v1_indexes_index_name_users_get(
        self,
        index_name: str,
        x_index_key: Optional[str] = None,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> ListUsersResponse:
        hdrs = dict(_headers) if _headers else {}
        if x_index_key is not None:
            hdrs["X-Index-Key"] = x_index_key
        return await self._call_and_deserialize(
            "GET",
            "/v1/indexes/{index_name}/users",
            {"200": "ListUsersResponse"},
            path_params={"index_name": index_name},
            header_params=hdrs,
            _request_timeout=_request_timeout,
        )

    async def delete_user_v1_indexes_index_name_users_user_id_delete(
        self,
        index_name: str,
        user_id: str,
        x_index_key: Optional[str] = None,
        _headers: Optional[Dict[str, Any]] = None,
        _request_timeout=None,
    ) -> None:
        hdrs = dict(_headers) if _headers else {}
        if x_index_key is not None:
            hdrs["X-Index-Key"] = x_index_key
        await self._call_and_deserialize(
            "DELETE",
            "/v1/indexes/{index_name}/users/{user_id}",
            {"200": "object"},
            path_params={"index_name": index_name, "user_id": user_id},
            header_params=hdrs,
            _request_timeout=_request_timeout,
        )
