
#Sanity checks for Node implementations
from basic_nodes import LinearTransformNode, BiasNode, SigmoidNode
from composite_nodes import GraphNode, SequentialNode
import numpy as np
import random

#Learning the xor function with a neural network 2 -> 2 -> 1
# But using the GraphNode class instead of SequentialNode, mainly as a sanity
# check for GraphNode.
#nodes = {
#    "w1" : LinearTransformNode(2, 2),
#    "b1" : BiasNode(2),
#    "s1" : SigmoidNode(2),
#    "w2" : LinearTransformNode(2, 1),
#    "b2" : BiasNode(1),
#    "s2" : SigmoidNode(1),
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
#samples = [ \
#    np.array([0, 0]), \
#    np.array([0, 1]), \
#    np.array([1, 0]), \
#    np.array([1, 1]) \
#]
#
#print("Before training: ")
#for sample in samples:
#    print(sample, ": ", nn((sample,)))
#
#for i in range(3000):
#    sample = samples[random.randint(0, 3)]
#    (result,) = nn((sample,))
#    expected = int(sample[0] != sample[1])
#    gradient = -1 / (1 - expected - result)
#    nn = nn.update((sample,), (gradient,), 1)
#    if i % 100 == 0:
#        print("After iteration ", i)
#        for sample in samples:
#            print(sample, ": ", nn((sample,)))
#
#print("After training: ")
#for sample in samples:
#    print(sample, ": ", nn((sample,)))
#
#MNIST dataset
#
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 128 - 0.5
x_test = x_test / 128 - 0.5

nodes = {
    "w1" : LinearTransformNode(784, 16),
    "b1" : BiasNode(16),
    "s1" : SigmoidNode(16),
    "w2" : LinearTransformNode(16, 16),
    "b2" : BiasNode(16),
    "s2" : SigmoidNode(16),
    "w3" : LinearTransformNode(16, 10),
    "b3" : BiasNode(10),
    "s3" : SigmoidNode(10)
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
#nn = SequentialNode([
#    LinearTransformNode(784, 16), 
#    BiasNode(16), 
#    SigmoidNode(16), 
#    LinearTransformNode(16, 16), 
#    BiasNode(16), 
#    SigmoidNode(16),
#    LinearTransformNode(16, 10),
#    BiasNode(10),
#    SigmoidNode(10)
#])

#Returns the test set accuracy for the callable [network].
def test_accuracy(network):
    total = 0
    correct = 0
    for x, y in zip(x_test, y_test):
        total += 1
        (predict,) = network((x.flatten(),))
        category = 1
        confidence = 0
        for i, p in enumerate(predict):
            if p > confidence:
                category = i + 1
                confidence = p
        if category == y: correct += 1
    return correct / total

print("Before training accuracy: ", test_accuracy(nn))
for i in range(9):
    print("Accuracy before epoch ", i, ": ", test_accuracy(nn))
    for x, y in zip(x_train, y_train):
        x = x.flatten()
        (result,) = nn((x,))
        gradient = []
        for i, r in enumerate(result):
            if i + 1 == y:
                #then expect a 1, so gradient should be 1/x
                gradient.append(1 - r)
            else:
                #expect a 0, so gradient should be 1/(x - 1)
                gradient.append(-r)
        nn = nn.update((x,), (np.array(gradient),), 1)

print("After training accuracy: ", test_accuracy(nn))
