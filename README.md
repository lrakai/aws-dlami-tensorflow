# aws-dlami-tensorflow

This demo illustrates working with TensorFlow using an Amazon Deep Learning AMI (DLAMI). It includes:

1. `src/basics.py` - Understand basic operations in TensorFlow
1. `src/nn.py` - Create a small neural network regression model in TensorFlow
1. `src/nn_with_summaries.py` - Show how to augment code with TensorFlow summaries to visualize the graph and learning process in TensorBoard
1. `src/nn_export.py` - Illustrates how to save a TensorFlow model to disk so that it can be served by TensorFlow Serving
1. `src/nn_client.py` - Example of how to consume the model served by TensorFlow Serving

![Lab environment](https://user-images.githubusercontent.com/3911650/36124280-b53215ca-100c-11e8-86b5-7cdac414b7e9.png)

## Getting Started

Deploy the CloudFormation stack in the template in `infrastructure/`. The template creates a user with the following credentials and minimal required permisisons to complete the Lab:

- Username: _student_
- Password: _password_

## Instructions

- Connect to the instance using the SSH username: _ubuntu_. 
- Run the Jupyter notebook server that comes pre-installed on the [Amazon Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB): `jupyter notebook` 
- SSH tunnel to the notebook server running on port 8888
- Open a browser to the notebook server on localhost. Get the URL with token from the command `jupyter notebook list`
- Create a new python 2.7 and TensorFlow environment notebook for each file in the `src/` directory
- Paste the code in from each script in the `src/` directory into a cell
- Run the notebooks
  - To view the summaries of `src/nn_with_summaries.py` in TensorBoard, run the command: `tensorboard --logdir /tmp/tensorflow/nn`
  - To serve the model saved by `src/nn_export.py` with TensorFlow Serving, run the command: `tensorflow_model_server --port=9000 --model_name=nn --model_base_path=/tmp/nn`

## Cleaning Up

Delete the CloudFormation stack to remove all the resources. No resources are created outside of those created by the template.
