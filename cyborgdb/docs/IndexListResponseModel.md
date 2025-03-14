# IndexListResponseModel

Response model for listing all indexes.  Attributes:     indexes (List[str]): List of available index names.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**indexes** | **List[str]** |  | 

## Example

```python
from openapi_client.models.index_list_response_model import IndexListResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of IndexListResponseModel from a JSON string
index_list_response_model_instance = IndexListResponseModel.from_json(json)
# print the JSON string representation of the object
print(IndexListResponseModel.to_json())

# convert the object into a dict
index_list_response_model_dict = index_list_response_model_instance.to_dict()
# create an instance of IndexListResponseModel from a dict
index_list_response_model_from_dict = IndexListResponseModel.from_dict(index_list_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


