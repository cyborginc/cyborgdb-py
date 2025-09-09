# IndexTrainingStatusResponseModel

Response model for retrieving the training status of indexes.  Attributes:     training_indexes (List[str]): List of index names currently being trained.     retrain_threshold (int): The multiplier used for the retraining threshold.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**training_indexes** | **List[str]** |  | 
**retrain_threshold** | **int** |  | 
**worker_pid** | **int** |  | 
**global_training** | **Dict[str, object]** |  | 

## Example

```python
from cyborgdb.openapi_client.models.index_training_status_response_model import IndexTrainingStatusResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of IndexTrainingStatusResponseModel from a JSON string
index_training_status_response_model_instance = IndexTrainingStatusResponseModel.from_json(json)
# print the JSON string representation of the object
print(IndexTrainingStatusResponseModel.to_json())

# convert the object into a dict
index_training_status_response_model_dict = index_training_status_response_model_instance.to_dict()
# create an instance of IndexTrainingStatusResponseModel from a dict
index_training_status_response_model_from_dict = IndexTrainingStatusResponseModel.from_dict(index_training_status_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


