# ListIDsResponse

Response model for listing all IDs in the index.  Attributes:     ids (List[str]): List of all item IDs in the index.     count (int): Total number of IDs in the index.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ids** | **List[str]** |  | 
**count** | **int** |  | 

## Example

```python
from cyborgdb.openapi_client.models.list_ids_response import ListIDsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListIDsResponse from a JSON string
list_ids_response_instance = ListIDsResponse.from_json(json)
# print the JSON string representation of the object
print(ListIDsResponse.to_json())

# convert the object into a dict
list_ids_response_dict = list_ids_response_instance.to_dict()
# create an instance of ListIDsResponse from a dict
list_ids_response_from_dict = ListIDsResponse.from_dict(list_ids_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


