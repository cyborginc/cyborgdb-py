# IndexOperationRequest

Request model for performing operations on an existing index (e.g., delete, describe).  Attributes:     index_key (str): A 32-byte encryption key as a hex string.     index_name (str): The name/identifier of the index.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 

## Example

```python
from openapi_client.models.index_operation_request import IndexOperationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of IndexOperationRequest from a JSON string
index_operation_request_instance = IndexOperationRequest.from_json(json)
# print the JSON string representation of the object
print(IndexOperationRequest.to_json())

# convert the object into a dict
index_operation_request_dict = index_operation_request_instance.to_dict()
# create an instance of IndexOperationRequest from a dict
index_operation_request_from_dict = IndexOperationRequest.from_dict(index_operation_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


