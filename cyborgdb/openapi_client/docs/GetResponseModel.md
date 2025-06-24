# GetResponseModel

Response model for retrieving multiple encrypted index items.  Attributes:     results (List[GetResultItem]): A list of retrieved items with requested fields.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[GetResultItemModel]**](GetResultItemModel.md) |  | 

## Example

```python
from cyborgdb.openapi_client.models.get_response_model import GetResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of GetResponseModel from a JSON string
get_response_model_instance = GetResponseModel.from_json(json)
# print the JSON string representation of the object
print(GetResponseModel.to_json())

# convert the object into a dict
get_response_model_dict = get_response_model_instance.to_dict()
# create an instance of GetResponseModel from a dict
get_response_model_from_dict = GetResponseModel.from_dict(get_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


