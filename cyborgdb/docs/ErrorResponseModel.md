# ErrorResponseModel

Standard error response model.  Attributes:     status_code (int): HTTP status code of the error.     detail (str): A detailed message describing the error.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**status_code** | **int** |  | 
**detail** | **str** |  | 

## Example

```python
from openapi_client.models.error_response_model import ErrorResponseModel

# TODO update the JSON string below
json = "{}"
# create an instance of ErrorResponseModel from a JSON string
error_response_model_instance = ErrorResponseModel.from_json(json)
# print the JSON string representation of the object
print(ErrorResponseModel.to_json())

# convert the object into a dict
error_response_model_dict = error_response_model_instance.to_dict()
# create an instance of ErrorResponseModel from a dict
error_response_model_from_dict = ErrorResponseModel.from_dict(error_response_model_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


