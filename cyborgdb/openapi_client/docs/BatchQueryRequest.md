# BatchQueryRequest

Request model for batch similarity search.  Inherits:     IndexOperationRequest: Includes `index_name` and `index_key`.  Attributes:     query_vectors (List[List[float]]): List of vectors to search for in batch mode.     top_k (int): Number of nearest neighbors to return for each query. Defaults to 100.     n_probes (int): Number of lists to probe during the query. Defaults to 1.     greedy (bool): Whether to use greedy search. Defaults to False.     filters (Optional[Dict[str, Any]]): JSON-like dictionary specifying metadata filters. Defaults to {}.     include (List[str]): List of additional fields to include in the response. Defaults to `[\"distance\", \"metadata\"]`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**query_vectors** | **List[List[float]]** |  | 
**top_k** | **int** |  | [optional] [default to 100]
**n_probes** | **int** |  | [optional] [default to 1]
**greedy** | **bool** |  | [optional] [default to False]
**filters** | **Dict[str, object]** |  | [optional] 
**include** | **List[str]** |  | [optional] [default to [distance, metadata]]

## Example

```python
from cyborgdb.openapi_client.models.batch_query_request import BatchQueryRequest

# TODO update the JSON string below
json = "{}"
# create an instance of BatchQueryRequest from a JSON string
batch_query_request_instance = BatchQueryRequest.from_json(json)
# print the JSON string representation of the object
print(BatchQueryRequest.to_json())

# convert the object into a dict
batch_query_request_dict = batch_query_request_instance.to_dict()
# create an instance of BatchQueryRequest from a dict
batch_query_request_from_dict = BatchQueryRequest.from_dict(batch_query_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


