# GetResultItemModel

Represents an individual item retrieved from the encrypted index.  Attributes:     id (str): The unique identifier of the item.     metadata (Optional[Dict[str, Any]]): Additional metadata associated with the item.     contents (Optional[bytes]): The raw byte contents of the item.     vector (Optional[List[float]]): The vector representation of the item.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** |  | 
**metadata** | **Dict[str, object]** |  | [optional] 
**contents** | **bytearray** |  | [optional] 
**vector** | **List[float]** |  | [optional] 

## Example

```python
from cyborgdb.openapi_client.models.get_result_item_model import GetResultItemModel

# TODO update the JSON string below
json = "{}"
# create an instance of GetResultItemModel from a JSON string
get_result_item_model_instance = GetResultItemModel.from_json(json)
# print the JSON string representation of the object
print(GetResultItemModel.to_json())

# convert the object into a dict
get_result_item_model_dict = get_result_item_model_instance.to_dict()
# create an instance of GetResultItemModel from a dict
get_result_item_model_from_dict = GetResultItemModel.from_dict(get_result_item_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


