# TrainRequest

Request model for training an index.  Attributes:     batch_size (int): Size of each batch for training. Default is 2048.     max_iters (int): Maximum iterations for training. Default is 100.     tolerance (float): Convergence tolerance for training. Default is 1e-6.     max_memory (int): Maximum memory (MB) usage during training. Default is 0 (no limit).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**index_key** | **str** | 32-byte encryption key as hex string | 
**index_name** | **str** | ID name | 
**batch_size** | **int** |  | [optional] [default to 2048]
**max_iters** | **int** |  | [optional] [default to 100]
**tolerance** | **float** |  | [optional] [default to 1.0E-6]
**max_memory** | **int** |  | [optional] [default to 0]

## Example

```python
from cyborgdb.openapi_client.models.train_request import TrainRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TrainRequest from a JSON string
train_request_instance = TrainRequest.from_json(json)
# print the JSON string representation of the object
print(TrainRequest.to_json())

# convert the object into a dict
train_request_dict = train_request_instance.to_dict()
# create an instance of TrainRequest from a dict
train_request_from_dict = TrainRequest.from_dict(train_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


