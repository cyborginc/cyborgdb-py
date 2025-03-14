# QueryResultItem

Represents a single result from a similarity search.  Attributes:     id (str): The identifier of the retrieved item.     distance (Optional[float]): Distance from the query vector (smaller = more similar).     metadata (Optional[Dict[str, Any]]): Additional metadata for the result.     vector (Optional[List[float]]): The retrieved vector (if included in response).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** |  | 
**distance** | **float** |  | [optional] 
**metadata** | **object** |  | [optional] 
**vector** | **List[float]** |  | [optional] 

## Example

```python
from openapi_client.models.query_result_item import QueryResultItem

# TODO update the JSON string below
json = "{}"
# create an instance of QueryResultItem from a JSON string
query_result_item_instance = QueryResultItem.from_json(json)
# print the JSON string representation of the object
print(QueryResultItem.to_json())

# convert the object into a dict
query_result_item_dict = query_result_item_instance.to_dict()
# create an instance of QueryResultItem from a dict
query_result_item_from_dict = QueryResultItem.from_dict(query_result_item_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


