# IndexConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | **int** |  | [optional] 
**type** | **str** |  | [optional] [default to 'ivfflat']
**pq_dim** | **int** |  | 
**pq_bits** | **int** |  | 

## Example

```python
from cyborgdb.openapi_client.models.index_config import IndexConfig

# TODO update the JSON string below
json = "{}"
# create an instance of IndexConfig from a JSON string
index_config_instance = IndexConfig.from_json(json)
# print the JSON string representation of the object
print(IndexConfig.to_json())

# convert the object into a dict
index_config_dict = index_config_instance.to_dict()
# create an instance of IndexConfig from a dict
index_config_from_dict = IndexConfig.from_dict(index_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


