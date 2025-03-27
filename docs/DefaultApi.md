# cyborgdb.openapi_client.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_index_v1_indexes_create_post**](DefaultApi.md#create_index_v1_indexes_create_post) | **POST** /v1/indexes/create | Create Encrypted Index
[**delete_index_v1_indexes_delete_post**](DefaultApi.md#delete_index_v1_indexes_delete_post) | **POST** /v1/indexes/delete | Delete Encrypted Index
[**delete_vectors_v1_vectors_delete_post**](DefaultApi.md#delete_vectors_v1_vectors_delete_post) | **POST** /v1/vectors/delete | Delete Items from Encrypted Index
[**get_index_info_v1_indexes_describe_post**](DefaultApi.md#get_index_info_v1_indexes_describe_post) | **POST** /v1/indexes/describe | Describe Encrypted Index
[**get_vectors_v1_vectors_get_post**](DefaultApi.md#get_vectors_v1_vectors_get_post) | **POST** /v1/vectors/get | Get Items from Encrypted Index
[**health_check_v1_health_get**](DefaultApi.md#health_check_v1_health_get) | **GET** /v1/health | Health check endpoint
[**list_indexes_v1_indexes_list_get**](DefaultApi.md#list_indexes_v1_indexes_list_get) | **GET** /v1/indexes/list | List Encrypted Indexes
[**query_vectors_v1_vectors_query_post**](DefaultApi.md#query_vectors_v1_vectors_query_post) | **POST** /v1/vectors/query | Query Encrypted Index
[**train_index_v1_indexes_train_post**](DefaultApi.md#train_index_v1_indexes_train_post) | **POST** /v1/indexes/train | Train Encrypted index
[**upsert_vectors_v1_vectors_upsert_post**](DefaultApi.md#upsert_vectors_v1_vectors_upsert_post) | **POST** /v1/vectors/upsert | Add Items to Encrypted Index


# **create_index_v1_indexes_create_post**
> CyborgdbServiceApiSchemasIndexSuccessResponseModel create_index_v1_indexes_create_post(create_index_request)

Create Encrypted Index

Create a new encrypted index with the provided configuration.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.create_index_request import CreateIndexRequest
from cyborgdb.openapi_client.models.cyborgdb_service_api_schemas_index_success_response_model import CyborgdbServiceApiSchemasIndexSuccessResponseModel
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    create_index_request = cyborgdb.openapi_client.CreateIndexRequest() # CreateIndexRequest | 

    try:
        # Create Encrypted Index
        api_response = api_instance.create_index_v1_indexes_create_post(create_index_request)
        print("The response of DefaultApi->create_index_v1_indexes_create_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->create_index_v1_indexes_create_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_index_request** | [**CreateIndexRequest**](CreateIndexRequest.md)|  | 

### Return type

[**CyborgdbServiceApiSchemasIndexSuccessResponseModel**](CyborgdbServiceApiSchemasIndexSuccessResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**409** | Conflict for index name |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_index_v1_indexes_delete_post**
> CyborgdbServiceApiSchemasIndexSuccessResponseModel delete_index_v1_indexes_delete_post(index_operation_request)

Delete Encrypted Index

Delete a specific index.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.cyborgdb_service_api_schemas_index_success_response_model import CyborgdbServiceApiSchemasIndexSuccessResponseModel
from cyborgdb.openapi_client.models.index_operation_request import IndexOperationRequest
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    index_operation_request = cyborgdb.openapi_client.IndexOperationRequest() # IndexOperationRequest | 

    try:
        # Delete Encrypted Index
        api_response = api_instance.delete_index_v1_indexes_delete_post(index_operation_request)
        print("The response of DefaultApi->delete_index_v1_indexes_delete_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->delete_index_v1_indexes_delete_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **index_operation_request** | [**IndexOperationRequest**](IndexOperationRequest.md)|  | 

### Return type

[**CyborgdbServiceApiSchemasIndexSuccessResponseModel**](CyborgdbServiceApiSchemasIndexSuccessResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**404** | Not able to find index |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_vectors_v1_vectors_delete_post**
> CyborgdbServiceApiSchemasVectorsSuccessResponseModel delete_vectors_v1_vectors_delete_post(delete_request)

Delete Items from Encrypted Index

Delete vectors by their IDs.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.cyborgdb_service_api_schemas_vectors_success_response_model import CyborgdbServiceApiSchemasVectorsSuccessResponseModel
from cyborgdb.openapi_client.models.delete_request import DeleteRequest
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    delete_request = cyborgdb.openapi_client.DeleteRequest() # DeleteRequest | 

    try:
        # Delete Items from Encrypted Index
        api_response = api_instance.delete_vectors_v1_vectors_delete_post(delete_request)
        print("The response of DefaultApi->delete_vectors_v1_vectors_delete_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->delete_vectors_v1_vectors_delete_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **delete_request** | [**DeleteRequest**](DeleteRequest.md)|  | 

### Return type

[**CyborgdbServiceApiSchemasVectorsSuccessResponseModel**](CyborgdbServiceApiSchemasVectorsSuccessResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Unable to find item to delete |  -  |
**500** | Unexpected server error |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_index_info_v1_indexes_describe_post**
> IndexInfoResponseModel get_index_info_v1_indexes_describe_post(index_operation_request)

Describe Encrypted Index

Get information about a specific index.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.index_info_response_model import IndexInfoResponseModel
from cyborgdb.openapi_client.models.index_operation_request import IndexOperationRequest
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    index_operation_request = cyborgdb.openapi_client.IndexOperationRequest() # IndexOperationRequest | 

    try:
        # Describe Encrypted Index
        api_response = api_instance.get_index_info_v1_indexes_describe_post(index_operation_request)
        print("The response of DefaultApi->get_index_info_v1_indexes_describe_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_index_info_v1_indexes_describe_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **index_operation_request** | [**IndexOperationRequest**](IndexOperationRequest.md)|  | 

### Return type

[**IndexInfoResponseModel**](IndexInfoResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**404** | Not able to find index |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_vectors_v1_vectors_get_post**
> GetResponseModel get_vectors_v1_vectors_get_post(get_request)

Get Items from Encrypted Index

Retrieve vectors by their IDs.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.get_request import GetRequest
from cyborgdb.openapi_client.models.get_response_model import GetResponseModel
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    get_request = cyborgdb.openapi_client.GetRequest() # GetRequest | 

    try:
        # Get Items from Encrypted Index
        api_response = api_instance.get_vectors_v1_vectors_get_post(get_request)
        print("The response of DefaultApi->get_vectors_v1_vectors_get_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_vectors_v1_vectors_get_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **get_request** | [**GetRequest**](GetRequest.md)|  | 

### Return type

[**GetResponseModel**](GetResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **health_check_v1_health_get**
> Dict[str, str] health_check_v1_health_get()

Health check endpoint

Check if the API is running.

### Example


```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)

    try:
        # Health check endpoint
        api_response = api_instance.health_check_v1_health_get()
        print("The response of DefaultApi->health_check_v1_health_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->health_check_v1_health_get: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

**Dict[str, str]**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_indexes_v1_indexes_list_get**
> IndexListResponseModel list_indexes_v1_indexes_list_get()

List Encrypted Indexes

List all available indexes.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.index_list_response_model import IndexListResponseModel
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)

    try:
        # List Encrypted Indexes
        api_response = api_instance.list_indexes_v1_indexes_list_get()
        print("The response of DefaultApi->list_indexes_v1_indexes_list_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->list_indexes_v1_indexes_list_get: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

[**IndexListResponseModel**](IndexListResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **query_vectors_v1_vectors_query_post**
> QueryResponse query_vectors_v1_vectors_query_post(request)

Query Encrypted Index

Search for nearest neighbors in the index.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.query_response import QueryResponse
from cyborgdb.openapi_client.models.request import Request
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    request = cyborgdb.openapi_client.Request() # Request | 

    try:
        # Query Encrypted Index
        api_response = api_instance.query_vectors_v1_vectors_query_post(request)
        print("The response of DefaultApi->query_vectors_v1_vectors_query_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->query_vectors_v1_vectors_query_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **request** | [**Request**](Request.md)|  | 

### Return type

[**QueryResponse**](QueryResponse.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **train_index_v1_indexes_train_post**
> CyborgdbServiceApiSchemasIndexSuccessResponseModel train_index_v1_indexes_train_post(train_request)

Train Encrypted index

Train the index for efficient querying.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.cyborgdb_service_api_schemas_index_success_response_model import CyborgdbServiceApiSchemasIndexSuccessResponseModel
from cyborgdb.openapi_client.models.train_request import TrainRequest
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    train_request = cyborgdb.openapi_client.TrainRequest() # TrainRequest | 

    try:
        # Train Encrypted index
        api_response = api_instance.train_index_v1_indexes_train_post(train_request)
        print("The response of DefaultApi->train_index_v1_indexes_train_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->train_index_v1_indexes_train_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **train_request** | [**TrainRequest**](TrainRequest.md)|  | 

### Return type

[**CyborgdbServiceApiSchemasIndexSuccessResponseModel**](CyborgdbServiceApiSchemasIndexSuccessResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **upsert_vectors_v1_vectors_upsert_post**
> CyborgdbServiceApiSchemasVectorsSuccessResponseModel upsert_vectors_v1_vectors_upsert_post(upsert_request)

Add Items to Encrypted Index

Add or update vectors in the index.

### Example

* Api Key Authentication (APIKeyHeader):

```python
import cyborgdb.openapi_client
from cyborgdb.openapi_client.models.cyborgdb_service_api_schemas_vectors_success_response_model import CyborgdbServiceApiSchemasVectorsSuccessResponseModel
from cyborgdb.openapi_client.models.upsert_request import UpsertRequest
from cyborgdb.openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = cyborgdb.openapi_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: APIKeyHeader
configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

# Enter a context with an instance of the API client
with cyborgdb.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = cyborgdb.openapi_client.DefaultApi(api_client)
    upsert_request = cyborgdb.openapi_client.UpsertRequest() # UpsertRequest | 

    try:
        # Add Items to Encrypted Index
        api_response = api_instance.upsert_vectors_v1_vectors_upsert_post(upsert_request)
        print("The response of DefaultApi->upsert_vectors_v1_vectors_upsert_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->upsert_vectors_v1_vectors_upsert_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **upsert_request** | [**UpsertRequest**](UpsertRequest.md)|  | 

### Return type

[**CyborgdbServiceApiSchemasVectorsSuccessResponseModel**](CyborgdbServiceApiSchemasVectorsSuccessResponseModel.md)

### Authorization

[APIKeyHeader](../README.md#APIKeyHeader)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response |  -  |
**401** | Permission denied from license issue |  -  |
**500** | Unexpected server error |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

