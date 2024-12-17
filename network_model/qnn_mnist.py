
#Using QNN to classify MNIST images

from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

sample_3_train = []
sample_6_train = []
sample_3_test = []
sample_6_test = []

def process(x):
    small = np.array([[sum(x[7 * i + l][7 * j + m] for l in range(7) \
        for m in range(7)) for j in range(4)] for i in range(4)])
    scaled = small.flatten() / 255.0 / 49.0
    output = np.array([scaled[1], scaled[2], scaled[5], scaled[6], scaled[9],\
        scaled[10], scaled[13], scaled[14]])
    return output

for x, y in zip(x_train, y_train):
    if y == 3: sample_3_train.append(process(x))
    elif y == 6: sample_6_train.append(process(x))

for x, y in zip(x_test, y_test):
    if y == 3: sample_3_test.append(process(x))
    elif y == 6: sample_6_test.append(process(x))

print("3 train: ", len(sample_3_train))
print("6 train: ", len(sample_6_train))

import qiskit as qk
from qiskit.circuit import Parameter
from qnn import Gate, QNNNode
input_params = [Parameter("i" + str(i)) for i in range(8)]
layer1x = [Parameter("x1" + str(i)) for i in range(8)]
layer1z = [Parameter("z1" + str(i)) for i in range(8)]
layer2x = [Parameter("x2" + str(i)) for i in range(8)]
layer2z = [Parameter("z2" + str(i)) for i in range(8)]
trainable_params = []
trainable_params.extend(layer1x)
trainable_params.extend(layer1z)
trainable_params.extend(layer2x)
trainable_params.extend(layer2z)
gates = []
gates.extend(Gate(Gate.RX, i, param=p) for i, p in enumerate(input_params))
gates.extend(Gate(Gate.CZ, i, i + 1) for i in range(7))
gates.extend(Gate(Gate.RX, i, param=p) for i, p in enumerate(input_params))
gates.extend(Gate(Gate.CZ, i, i + 1) for i in range(7))
gates.extend(Gate(Gate.RX, i, param=p) for i, p in enumerate(layer1x))
gates.extend(Gate(Gate.RZ, i, param=p) for i, p in enumerate(layer1z))
gates.extend(Gate(Gate.CX, i, i + 1) for i in range(7))
gates.extend(Gate(Gate.RX, i, param=p) for i, p in enumerate(layer2x))
gates.extend(Gate(Gate.RZ, i, param=p) for i, p in enumerate(layer2z))
gates.extend(Gate(Gate.CX, i, i + 1) for i in range(7))

qnn = QNNNode(8, input_params, trainable_params, gates)

import basic_nodes
from composite_nodes import SequentialNode
from node import Node

class FixedLinearCombination(Node):
    #require [w] is a numpy array
    def __init__(self, w):
        super().__init__((len(w),), (1,))
        self._w = w
    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (np.array([inputs[0].dot(self._w)]),)
    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        return (self._w * output_gradient[0],)
    def update(self, inputs, output_gradient, step_size):
        return self
    def update_batch(self, inputs, output_gradient, step_size):
        return self

nn = SequentialNode([qnn, FixedLinearCombination(np.array([0, 0, 0, 0, 0, 0, 0, 1]))])
#nn = SequentialNode([basic_nodes.LinearTransformNode(8, 4), basic_nodes.BiasNode(4), basic_nodes.SigmoidNode(4), basic_nodes.LinearTransformNode(4, 2), basic_nodes.SigmoidNode(2), FixedLinearCombination(np.array([-10, 10])), basic_nodes.SigmoidNode(1)])

def test_accuracy(network):
    count = 0
    correct = 0
    for x in sample_3_test:
        print("Testing sample ", count)
        (output,) = nn((x,))
        if output[0] < 0.5: correct += 1
        count += 1
    for x in sample_6_test:
        print("Testing sample ", count)
        (output,) = nn((x,))
        if output[0] > 0.5: correct += 1
        count += 1
    return correct / count

pre_train_accuracy = test_accuracy(nn)
for k in range(5):
    iterations = 0
    for x_3, x_6 in zip(sample_3_test, sample_6_test):
        print("Epoch ", k, ", Training iteration ", iterations)
        (output,) = nn((x_3,))
        grad = (np.array([-output[0]]),)
        nn = nn.update((x_3,), grad, 0.1)
        (output,) = nn((x_6,))
        grad = (np.array([1 - output[0]]),)
        nn = nn.update((x_3,), grad, 0.1)
        iterations += 1

post_train_accuracy = test_accuracy(nn)
print("pre-training accuracy: ", pre_train_accuracy)
print("post-training accuracy: ", post_train_accuracy)
