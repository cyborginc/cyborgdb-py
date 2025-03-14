# IndexIVFFlatModel

Model for configuring an IVFFlat (Inverted File with Flat quantization) index.  Attributes:     type (str): Index type identifier. Defaults to \"ivfflat\".

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | **int** |  | [optional] 
**n_lists** | **int** |  | 
**metric** | **str** |  | [optional] 
**type** | **str** |  | [optional] [default to 'ivfflat']

## Example

```python
from openapi_client.models.index_ivf_flat_model import IndexIVFFlatModel

# TODO update the JSON string below
json = "{}"
# create an instance of IndexIVFFlatModel from a JSON string
index_ivf_flat_model_instance = IndexIVFFlatModel.from_json(json)
# print the JSON string representation of the object
print(IndexIVFFlatModel.to_json())

# convert the object into a dict
index_ivf_flat_model_dict = index_ivf_flat_model_instance.to_dict()
# create an instance of IndexIVFFlatModel from a dict
index_ivf_flat_model_from_dict = IndexIVFFlatModel.from_dict(index_ivf_flat_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


