# UpsertRequest

Request model for adding or updating vectors in an encrypted index.  Inherits:     IndexOperationRequest: Includes `index_name` and `index_key`.  Attributes:     items (List[VectorItem]): List of vector items to be inserted or updated.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**items** | [**List[VectorItem]**](VectorItem.md) |  | 

## Example

```python
from openapi_client.models.upsert_request import UpsertRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpsertRequest from a JSON string
upsert_request_instance = UpsertRequest.from_json(json)
# print the JSON string representation of the object
print(UpsertRequest.to_json())

# convert the object into a dict
upsert_request_dict = upsert_request_instance.to_dict()
# create an instance of UpsertRequest from a dict
upsert_request_from_dict = UpsertRequest.from_dict(upsert_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


