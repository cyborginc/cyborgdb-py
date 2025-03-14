# CreateIndexRequest

Request model for creating a new encrypted index.  Attributes:     index_config (Union[IndexIVFModel, IndexIVFPQModel, IndexIVFFlatModel]):          The configuration model for the index.     index_key (str): A 32-byte encryption key as a hex string.     index_name (str): The name/identifier of the index.     embedding_model (Optional[str]): Optional embedding model name.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_config** | [**IndexConfig**](IndexConfig.md) |  | 
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**embedding_model** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.create_index_request import CreateIndexRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateIndexRequest from a JSON string
create_index_request_instance = CreateIndexRequest.from_json(json)
# print the JSON string representation of the object
print(CreateIndexRequest.to_json())

# convert the object into a dict
create_index_request_dict = create_index_request_instance.to_dict()
# create an instance of CreateIndexRequest from a dict
create_index_request_from_dict = CreateIndexRequest.from_dict(create_index_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


