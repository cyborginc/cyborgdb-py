# IndexInfoResponseModel

Response model for retrieving information about an index.  Attributes:     index_name (str): The name of the index.     index_type (str): The type of index (e.g., IVF, IVFFlat, IVFPQ).     is_trained (bool): Indicates whether the index has been trained.     index_config (Dict[str, Any]): The full configuration details of the index.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_name** | **str** |  | 
**index_type** | **str** |  | 
**is_trained** | **bool** |  | 
**index_config** | **object** |  | 

## Example

```python
from openapi_client.models.index_info_response_model import IndexInfoResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of IndexInfoResponseModel from a JSON string
index_info_response_model_instance = IndexInfoResponseModel.from_json(json)
# print the JSON string representation of the object
print(IndexInfoResponseModel.to_json())

# convert the object into a dict
index_info_response_model_dict = index_info_response_model_instance.to_dict()
# create an instance of IndexInfoResponseModel from a dict
index_info_response_model_from_dict = IndexInfoResponseModel.from_dict(index_info_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


