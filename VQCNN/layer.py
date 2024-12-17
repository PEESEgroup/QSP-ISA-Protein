from random import random
from math import exp

#Support for single layers of neural networks. Generally, this module assumes
# input vectors are normalized, but does not normalize output vectors.

#Asks layer [l] to compute the output on vector [v].
def compute(l, v):
    return l[0](l[1], v)

#Trains layer [l] on vector [v] with gradient vector [e], with training step 
# size [a]. Modifies [l] to [ll] such that [compute(ll, v) - compute(l, v)]
# is approximately [-a * e] (for small [a]).
def train(l, v, e, a):
    (_, t, g, _, u) = l
    gd = g(t, v, e)
    u(t, gd, -a)

#Trains layer [l] on a batch of data points. The list [vs] represents the list
# of inputs, and the list [es] represents the list of output gradients on each
# input. [a] is the training step.
def train_batch(l, vs, es, a):
    (_, t, g, _, u) = l
    bs = len(vs)
    a /= -bs
    assert bs == len(es)
    gds = []
    for v, e in zip(vs, es):
        gds.append(g(t, v, e))
    for gd in gds: 
        u(t, gd, a)

#For a layer [l], vector [v], and gradient vector [e], computes vector [vv] such
# that [compute(l, v + a * vv) - compute(l, v)] is approximately [-a * e]
# for small [a].
def grad_ext(l, v, e):
    (_, t, _, h, _) = l
    return h(t, v, e)

#Implementation note: A layer will be a 6-tuple (c, t, g, h, u) where [c] 
# is the computation function, [t] is the layer parameters object, [g] is the 
# parameters gradient function, [h] is the input gradient function, and [u] is
# the update function. 
# If the layer is fed input vector [v], then it will output vector [c(t, v)].
# If the layer is fed input vector [v], and it outputs [w], while the expected
# output is [w'], then the mean squared error loss is [e = |w - w'|^2]. Then,
# [g(t, v, w - w') = de/dt] and [h(t, v, w - w') = de/dv].
# If [gg] is [g(t, v, w - w')] for some parameter set [t], input [v], and
# desired gradient [w - w'], then calling [u(t, gg, a)] modifies [t] into a
# parameter set [tt] such that [c(tt, v) - c(t, v)] is approximately [a(w - w')]
# for small [a].

