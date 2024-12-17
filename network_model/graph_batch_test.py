
#Sanity check for batch_update implementation for SequentialNode

import numpy as np
import basic_nodes
from composite_nodes import GraphNode

#Helper function for constructing a dense neural network layer
def layer(inputs, outputs):
    return SequentialNode([ \
        basic_nodes.LinearTransformNode(inputs, outputs), \
        basic_nodes.BiasNode(outputs), \
        basic_nodes.SigmoidNode(outputs) \
    ])

#Learning XOR using batch gradient descent

#nodes = {
#    "w1" : basic_nodes.LinearTransformNode(2, 2),
#    "b1" : basic_nodes.BiasNode(2),
#    "s1" : basic_nodes.SigmoidNode(2),
#    "w2" : basic_nodes.LinearTransformNode(2, 1),
#    "b2" : basic_nodes.BiasNode(1),
#    "s2" : basic_nodes.SigmoidNode(1),
#}
#
#edges = {
#    ("in", 0) : ("w1", 0),
#    ("w1", 0) : ("b1", 0),
#    ("b1", 0) : ("s1", 0),
#    ("s1", 0) : ("w2", 0),
#    ("w2", 0) : ("b2", 0),
#    ("b2", 0) : ("s2", 0),
#    ("s2", 0) : ("out", 0),
#}
#
#nn = GraphNode((2,), (1,), nodes, edges)
#
#input_list = [ \
#    (np.array([0, 0]),), \
#    (np.array([0, 1]),), \
#    (np.array([1, 0]),), \
#    (np.array([1, 1]),), \
#]
#
#print("Before training: ")
#for sample in input_list:
#    print(sample[0], ": ", nn(sample))
#
#for i in range(2000):
#    (r00,), (r01,), (r10,), (r11,) = [nn(sample) for sample in input_list]
#    grad_list = [(-r00,), (1 - r01,), (1 - r10,), (-r11,)]
#    nn = nn.update_batch(input_list, grad_list, 5)
#    if i % 100 == 0:
#        print("After iteration ", i)
#        for sample in input_list:
#            print(sample[0], ": ", nn(sample))

#MNIST dataset with neural network 784 -> 16 -> 16 -> 10

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 128 - 0.5
x_test = x_test / 128 - 0.5

nodes = {
    "w1" : basic_nodes.LinearTransformNode(784, 16),
    "b1" : basic_nodes.BiasNode(16),
    "s1" : basic_nodes.SigmoidNode(16),
    "w2" : basic_nodes.LinearTransformNode(16, 16),
    "b2" : basic_nodes.BiasNode(16),
    "s2" : basic_nodes.SigmoidNode(16),
    "w3" : basic_nodes.LinearTransformNode(16, 10),
    "b3" : basic_nodes.BiasNode(10),
    "s3" : basic_nodes.SigmoidNode(10)
}

edges = {
    ("in", 0) : ("w1", 0),
    ("w1", 0) : ("b1", 0),
    ("b1", 0) : ("s1", 0),
    ("s1", 0) : ("w2", 0),
    ("w2", 0) : ("b2", 0),
    ("b2", 0) : ("s2", 0),
    ("s2", 0) : ("w3", 0),
    ("w3", 0) : ("b3", 0),
    ("b3", 0) : ("s3", 0),
    ("s3", 0) : ("out", 0),
}

nn = GraphNode((784,), (10,), nodes, edges)

def test_set_accuracy(network):
    correct = 0
    count = 0
    print(network((x_test[0].flatten(),)))
    for x, y in zip(x_test, y_test):
        (res,) = network((x.flatten(),))
        if res.tolist().index(max(res)) + 1 == y: correct += 1
        count += 1
    return correct / count

expected = [[int(i + 1 == y) for i in range(10)] for y in range(11)]
for i in range(99):
    print("Accuracy after ", i, " epochs of training: ", test_set_accuracy(nn))
    for i in range(0, len(x_train), 10):
        inputs = [(x.flatten(),) for x in x_train[i:i + 10]]
        res = [nn(x)[0] for x in inputs]
        grad = [(expected[y] - r,) for y, r in zip(y_train[i:i + 10], res)]
        nn = nn.update_batch(inputs, grad, 1)

