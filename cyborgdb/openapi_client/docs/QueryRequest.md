# QueryRequest

Request model for performing a similarity search in the encrypted index.  Inherits:     IndexOperationRequest: Includes `index_name` and `index_key`.  Attributes:     query_vectors (Optional[List[float]]): The vector used for the similarity search.     query_contents (Optional[str]): Text-based content used for semantic search.     top_k (int): Number of nearest neighbors to return for each query. Defaults to 100.     n_probes (int): Number of lists to probe during the query. Defaults to 1.     greedy (bool): Whether to use greedy search. Defaults to False.     filters (Optional[Dict[str, Any]]): JSON-like dictionary specifying metadata filters. Defaults to {}.     include (List[str]): List of additional fields to include in the response. Defaults to `[\"distance\", \"metadata\"]`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**query_vectors** | **List[float]** |  | [optional] 
**query_contents** | **str** |  | [optional] 
**top_k** | **int** |  | [optional] [default to 100]
**n_probes** | **int** |  | [optional] [default to 1]
**greedy** | **bool** |  | [optional] [default to False]
**filters** | **Dict[str, object]** |  | [optional] 
**include** | **List[str]** |  | [optional] [default to [distance, metadata]]

## Example

```python
from cyborgdb.openapi_client.models.query_request import QueryRequest

# TODO update the JSON string below
json = "{}"
# create an instance of QueryRequest from a JSON string
query_request_instance = QueryRequest.from_json(json)
# print the JSON string representation of the object
print(QueryRequest.to_json())

# convert the object into a dict
query_request_dict = query_request_instance.to_dict()
# create an instance of QueryRequest from a dict
query_request_from_dict = QueryRequest.from_dict(query_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


