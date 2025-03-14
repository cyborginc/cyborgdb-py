# IndexIVFModel

Model for configuring an IVF (Inverted File) index.  Attributes:     type (str): Index type identifier. Defaults to \"ivf\".

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | **int** |  | [optional] 
**n_lists** | **int** |  | 
**metric** | **str** |  | [optional] 
**type** | **str** |  | [optional] [default to 'ivf']

## Example

```python
from openapi_client.models.index_ivf_model import IndexIVFModel

# TODO update the JSON string below
json = "{}"
# create an instance of IndexIVFModel from a JSON string
index_ivf_model_instance = IndexIVFModel.from_json(json)
# print the JSON string representation of the object
print(IndexIVFModel.to_json())

# convert the object into a dict
index_ivf_model_dict = index_ivf_model_instance.to_dict()
# create an instance of IndexIVFModel from a dict
index_ivf_model_from_dict = IndexIVFModel.from_dict(index_ivf_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


