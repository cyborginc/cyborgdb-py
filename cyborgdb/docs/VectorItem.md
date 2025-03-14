# VectorItem

Represents a vectorized item for storage in the encrypted index.  Attributes:     id (str): Unique identifier for the vector item.     vector (Optional[List[float]]): The vector representation of the item.     contents (Optional[str]): The original text or associated content.     metadata (Optional[Dict[str, Any]]): Additional metadata associated with the item.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** |  | 
**vector** | **List[float]** |  | [optional] 
**contents** | **str** |  | [optional] 
**metadata** | **object** |  | [optional] 

## Example

```python
from openapi_client.models.vector_item import VectorItem

# TODO update the JSON string below
json = "{}"
# create an instance of VectorItem from a JSON string
vector_item_instance = VectorItem.from_json(json)
# print the JSON string representation of the object
print(VectorItem.to_json())

# convert the object into a dict
vector_item_dict = vector_item_instance.to_dict()
# create an instance of VectorItem from a dict
vector_item_from_dict = VectorItem.from_dict(vector_item_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


