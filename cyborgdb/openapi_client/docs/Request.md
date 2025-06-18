# Request


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**query_vectors** | **List[List[float]]** |  | 
**query_contents** | **str** |  | [optional] 
**top_k** | **int** |  | [optional] [default to 100]
**n_probes** | **int** |  | [optional] [default to 1]
**greedy** | **bool** |  | [optional] [default to False]
**filters** | **Dict[str, object]** |  | [optional] 
**include** | **List[str]** |  | [optional] [default to [distance, metadata]]

## Example

```python
from cyborgdb.openapi_client.models.request import Request

# TODO update the JSON string below
json = "{}"
# create an instance of Request from a JSON string
request_instance = Request.from_json(json)
# print the JSON string representation of the object
print(Request.to_json())

# convert the object into a dict
request_dict = request_instance.to_dict()
# create an instance of Request from a dict
request_from_dict = Request.from_dict(request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


