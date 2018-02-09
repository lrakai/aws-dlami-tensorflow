from __future__ import print_function

import tensorflow as tf

# Explicitly create a computation graph
graph = tf.Graph()
with graph.as_default():
    # Declare one-dimensional tensors (vectors)
    input1 = tf.constant([1.0, 2.0])
    input2 = tf.constant([3.0, 4.0])
    # Add the two tensors
    output = tf.add(input1, input2)

# print the operations stored by the graph
print(graph.get_operations())

# Evaluate the graph in a session
with tf.Session(graph=graph):
    result = output.eval()
    print("result: ", result)


# Evaluate using the default graph
with tf.Session():
    input1 = tf.constant([1.0, 2.0])
    input2 = tf.constant([3.0, 4.0])
    output = tf.add(input1, input2)
    # Show the operations in the default graph
    print(tf.get_default_graph().get_operations())
    result = output.eval()
    print("result: ", result)


# Evaluate a matrix-vector multiplication
matmul_graph = tf.Graph()
with matmul_graph.as_default():
    # Declare a 2x2 matrix and a 2x1 vector
    matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    vector = tf.constant([[1.0], [2.0]])
    # Matrix multiply (matmul) the two tensors
    output = tf.matmul(matrix, vector)

with tf.Session(graph=matmul_graph):
    result = output.eval()
    print(result)


