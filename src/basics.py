from __future__ import print_function

import tensorflow as tf

# Explicitly create a computation graph
graph=tf.Graph()
with graph.as_default():
    # Declare one-dimensional tensors (vectors)
    input1 = tf.constant([1.0, 1.0, 1.0, 1.0])
    input2 = tf.constant([2.0, 2.0, 2.0, 2.0])
    # Add the two tensors
    output = tf.add(input1, input2)

# print the operations stored by the graph
print(graph.get_operations())

# Evaluate the graph in a session
with tf.Session(graph = graph):
    result = output.eval()
    print("result: ", result)