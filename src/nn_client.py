from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

hostport = 'localhost:9000'


def do_prediction(hostport):

  # Create connection
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  # Initialize a request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'nn'
  request.model_spec.signature_name = 'prediction'

  # Use evenly-spaced points for test data
  tests = temp_data = np.array([range(-1, 6, 1)]).transpose().astype(
    np.float32)

  # Set the tests as the input for prediction
  request.inputs['input'].CopyFrom(
    tf.contrib.util.make_tensor_proto(tests, shape=tests.shape))

  # Get prediction from server
  result = stub.Predict(request, 5.0) # 5 second timeout

  # Compare to noise-free actual values
  actual = np.sum(0.5 * temp_data + 2.5, 1)

  return result, actual


prediction, actual = do_prediction(hostport)
print('Prediction is: ', prediction)
print('Noise-free value is: ', actual)
