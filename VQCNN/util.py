import math
from random import random
#Useful functions for neural network construction

def sigmoid(x):
    if x < -700: return 0
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s - s * s

#Returns the gradient of mean square error for produced output [x], expected
# output [y]
def dmse(x, y):
    return tuple([x[i] - y[i] for i in range(len(x))])

#Returns the total squared error for produced [x], expected [y].
def mse(x, y):
    output = 0
    for i in range(len(x)): output += (x[i] - y[i]) ** 2
    return output

def relu(x):
    return (x > 0) * x

def drelu(x):
    return int(x > 0)

def lrelu(x, alpha=0.3):
    if x > 0: return x
    return alpha * x

def dlrelu(x, alpha=0.3):
    if x > 0: return 1
    return alpha

#Methods for generating random matrices and vectors
def rand_mat(row, col):
    output = []
    for p in range(row):
        r = []
        for q in range(col):
            r.append(2 * random() - 1)
        output.append(r)
    return tuple(output)

def rand_vec(dims):
    output = []
    for d in range(dims):
        output.append(2 * random() - 1)
    return tuple(output)
