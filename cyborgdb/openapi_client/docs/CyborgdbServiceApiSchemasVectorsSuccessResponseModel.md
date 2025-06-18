# CyborgdbServiceApiSchemasVectorsSuccessResponseModel

Standard success response model for operations like upsert and delete.  Attributes:     status (str): Operation status. Defaults to `\"success\"`.     message (str): Descriptive success message.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**status** | **str** |  | [optional] [default to 'success']
**message** | **str** |  | 

## Example

```python
from cyborgdb.openapi_client.models.cyborgdb_service_api_schemas_vectors_success_response_model import CyborgdbServiceApiSchemasVectorsSuccessResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of CyborgdbServiceApiSchemasVectorsSuccessResponseModel from a JSON string
cyborgdb_service_api_schemas_vectors_success_response_model_instance = CyborgdbServiceApiSchemasVectorsSuccessResponseModel.from_json(json)
# print the JSON string representation of the object
print(CyborgdbServiceApiSchemasVectorsSuccessResponseModel.to_json())

# convert the object into a dict
cyborgdb_service_api_schemas_vectors_success_response_model_dict = cyborgdb_service_api_schemas_vectors_success_response_model_instance.to_dict()
# create an instance of CyborgdbServiceApiSchemasVectorsSuccessResponseModel from a dict
cyborgdb_service_api_schemas_vectors_success_response_model_from_dict = CyborgdbServiceApiSchemasVectorsSuccessResponseModel.from_dict(cyborgdb_service_api_schemas_vectors_success_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


