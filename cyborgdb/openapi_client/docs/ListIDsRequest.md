# ListIDsRequest

Request model for listing all IDs in the index.  Inherits:     IndexOperationRequest: Includes `index_name` and `index_key`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 

## Example

```python
from cyborgdb.openapi_client.models.list_ids_request import ListIDsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ListIDsRequest from a JSON string
list_ids_request_instance = ListIDsRequest.from_json(json)
# print the JSON string representation of the object
print(ListIDsRequest.to_json())

# convert the object into a dict
list_ids_request_dict = list_ids_request_instance.to_dict()
# create an instance of ListIDsRequest from a dict
list_ids_request_from_dict = ListIDsRequest.from_dict(list_ids_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


