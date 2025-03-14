# CyborgdbServiceApiSchemasIndexSuccessResponseModel

Standard success response model.  Attributes:     status (str): The status of the response. Defaults to \"success\".     message (str): A success message.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**status** | **str** |  | [optional] [default to 'success']
**message** | **str** |  | 

## Example

```python
from openapi_client.models.cyborgdb_service_api_schemas_index_success_response_model import CyborgdbServiceApiSchemasIndexSuccessResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of CyborgdbServiceApiSchemasIndexSuccessResponseModel from a JSON string
cyborgdb_service_api_schemas_index_success_response_model_instance = CyborgdbServiceApiSchemasIndexSuccessResponseModel.from_json(json)
# print the JSON string representation of the object
print(CyborgdbServiceApiSchemasIndexSuccessResponseModel.to_json())

# convert the object into a dict
cyborgdb_service_api_schemas_index_success_response_model_dict = cyborgdb_service_api_schemas_index_success_response_model_instance.to_dict()
# create an instance of CyborgdbServiceApiSchemasIndexSuccessResponseModel from a dict
cyborgdb_service_api_schemas_index_success_response_model_from_dict = CyborgdbServiceApiSchemasIndexSuccessResponseModel.from_dict(cyborgdb_service_api_schemas_index_success_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


