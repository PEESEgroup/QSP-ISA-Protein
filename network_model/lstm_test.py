
from rnn import LSTMNode
from composite_nodes import SequentialNode
from basic_nodes import LinearTransformNode, BiasNode, SigmoidNode
from node import Node
import math
import numpy as np
import random

#Training an LSTM to recognize sin(x/10), step size 1

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

nn = SequentialNode([LSTMNode(1, 5, 4), FixedLinearCombination(np.array([0.2] * 5))])

data = [math.sin(x / 10) for x in range(1000)]

#the first 400 data points are for training, the next 400 for testing

def prep_data(i):
    return (tuple(np.array([x]) for x in data[i:i + 4]), data[i + 4])

def l1_error(model):
    output = 0
    for i in range(400, 800):
        x, y = prep_data(i)
        (res,) = nn(x)
        output += abs(res - y)
    return output

print("Before training error: ", l1_error(nn))

seq = [i for i in range(400)]
for _ in range(10):
    for i in seq:
        print("Training sample: ", i)
        print("current error: ", l1_error(nn))
        x, y = prep_data(i)
        (res,) = nn(x)
        grad = (y - res,)
        nn = nn.update(x, grad, 1)

print("After training error: ", l1_error(nn))
