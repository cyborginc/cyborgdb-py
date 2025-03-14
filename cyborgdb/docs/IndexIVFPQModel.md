# IndexIVFPQModel

Model for configuring an IVFPQ (Inverted File with Product Quantization) index.  Attributes:     type (str): Index type identifier. Defaults to \"ivfpq\".     pq_dim (int): Dimensionality of PQ codes.     pq_bits (int): Number of bits per quantizer.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | **int** |  | [optional] 
**n_lists** | **int** |  | 
**metric** | **str** |  | [optional] 
**type** | **str** |  | [optional] [default to 'ivfpq']
**pq_dim** | **int** |  | 
**pq_bits** | **int** |  | 

## Example

```python
from openapi_client.models.index_ivfpq_model import IndexIVFPQModel

# TODO update the JSON string below
json = "{}"
# create an instance of IndexIVFPQModel from a JSON string
index_ivfpq_model_instance = IndexIVFPQModel.from_json(json)
# print the JSON string representation of the object
print(IndexIVFPQModel.to_json())

# convert the object into a dict
index_ivfpq_model_dict = index_ivfpq_model_instance.to_dict()
# create an instance of IndexIVFPQModel from a dict
index_ivfpq_model_from_dict = IndexIVFPQModel.from_dict(index_ivfpq_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


