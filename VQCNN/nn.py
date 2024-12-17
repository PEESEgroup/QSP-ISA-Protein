import layer
import layers
import util
import datagen
import time

#Feed-forward, multi-layer neural network.
#Implementation note: each neural network is a tuple of several layers.

#Constructs a neural network given a list of normal layers [l]. 
# Required that the number of outputs of
# each layer is the number of inputs of the next layer.
def from_layers(l):
    return tuple(l)

#Constructs a classical neural network using the dimension list [d], using
# sigmoid activation function. If [d] is a list of integers [d1], [d2], .., [dn]
# then the output is a neural network with [n - 1] layers. The first layer maps
# from [d1] inputs to [d2] outputs, the second layer maps [d2] inputs to [d3] 
# outputs, and so on. 
# Parameters are initialized randomly.
def classical(d):
    output = []
    for i in range(len(d) - 1):
        output.append(layers.classical_layer(d[i], d[i + 1]))
        output.append(layers.normal_layer(d[i + 1], util.sigmoid, 
            util.dsigmoid))
    return from_layers(output)

#Takes neural network [nn], computes its output on input [v]. 
def compute(nn, v):
    output = v
    for l in nn:
        output = layer.compute(l, output)
    return output

#Takes neural network [nn], input vector [v], expected output [w], training step
# [a], and error function gradient [g], and returns a new neural network [n'] 
# defined as follows. Let [u] be the output of [nn] on [v]. Then for small [a],
# [n'] on [v] returns [u'] if [u' - u] is approximately [-a * g(u, w)]. 
#Implementation note: first, compute the input to each layer. Next, cycle
# backwards through the layers, compute the gradient for that layer, train that
# layer on the gradient, and compute the gradient for the next layer.
def train(nn, v, w, g, a):
    inputs = []
    for l in nn:
        inputs.append(v)
        v = layer.compute(l, v)
    #So now v is the output of the computation and inputs[i] is the input to the
    # index [i] layer.
    assert v == compute(nn, inputs[0])
    e = g(v, w)
    for i in reversed(range(len(nn))):
        layer.train(nn[i], inputs[i], e, a)
        e = layer.grad_ext(nn[i], inputs[i], e)

#Train function but for the average gradient of a batch of inputs and outputs
# [vs] and [ws].
def train_batch(nn, vs, ws, g, a):
    inputs = [[] for _ in range(len(nn))]
    for i, l in enumerate(nn):
        for j, v in enumerate(vs):
            inputs[i].append(v)
            vs[j] = layer.compute(l, v)
    for j, v in enumerate(vs):
        assert v == compute(nn, inputs[0][j])
    iter = zip(vs, ws)
    es = [g(v, w) for v, w in iter]
    for i in reversed(range(len(nn))):
        layer.train_batch(nn[i], inputs[i], es, a)
        for j, e in enumerate(es):
            es[j] = layer.grad_ext(nn[i], inputs[i][j], e)
#Evaluates the neural network [nn] on the data from the generator [gen]. Returns
# (c, t), where [c] is the number of correct predictions and [t] is the total
# number of test samples.
def evaluate_gen(nn, gen):
    correct = 0
    total = 0
    while datagen.has_next(gen):
        (d, t) = datagen.next(gen)
        p = compute(nn, d)
        tt = p.index(max(p))
        if tt == t: 
            correct += 1
        total += 1
    return (correct, total)
