# GetRequest

Request model for retrieving specific vectors from the index.  Inherits:     IndexOperationRequest: Includes `index_name` and `index_key`.  Attributes:     ids (List[str]): List of vector item IDs to retrieve.     include (List[str]): List of fields to include in the response.          Defaults to `[\"vector\", \"contents\", \"metadata\"]`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**ids** | **List[str]** |  | 
**include** | **List[str]** |  | [optional] [default to [vector, contents, metadata]]

## Example

```python
from cyborgdb.openapi_client.models.get_request import GetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of GetRequest from a JSON string
get_request_instance = GetRequest.from_json(json)
# print the JSON string representation of the object
print(GetRequest.to_json())

# convert the object into a dict
get_request_dict = get_request_instance.to_dict()
# create an instance of GetRequest from a dict
get_request_from_dict = GetRequest.from_dict(get_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


