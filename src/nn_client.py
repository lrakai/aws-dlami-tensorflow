from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

hostport = 'localhost:9000'


def do_prediction(hostport):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the Prediction Service.
    Returns:
    predicted value, noise-free value
    """
    # create connection
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # initialize a request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'nn_model'
    request.model_spec.signature_name = 'prediction'

    # Randomly generate some test data
    temp_data = np.random.randn(60, 1).astype(np.float32)
    data, actual = temp_data, np.sum(0.5 * temp_data + 2.5, 1)
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=data.shape))

    # predict
    result = stub.Predict(request, 5.0)  # 5 second timeout
    return result, actual


def main(_):
    prediction, actual = do_prediction(hostport)
    print('Prediction is: ', prediction)
    print('Noise-free value is: ', actual)


if __name__ == '__main__':
    tf.app.run()
