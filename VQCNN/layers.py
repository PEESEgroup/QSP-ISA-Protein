#Module for implementing different types of neural network layers.
import layer
import util
import random
import math

#Constructs a classical neural network layer. [i] is the number of inputs and 
# [j] is the number of outputs. If [v] is the [i]-dimensional input, then the
# output will be [w = Mv + b], where [M] is an [j] by [i] parameter matrix, and 
# [b] is a [j] dimensional parameter vector. [M] and [b] are optional arguments
# and will be initialized randomly if they are not specified.
# Note that the only functionality of [M] required is for [M[p][q]] to be the
# element at the [p]th row and [q]th column of [M]. Required that [M] have [j]
# rows and [i] columns, and that [b] have [j] entries, and that [b[r]] is the
# [r]th entry in [b]. Indices start at zero.
#Implementation note: parameter vector [t] is a tuple where [M_pq] is 
# [t[i * p + q]] and [b[r]] is t[i * j + r]. Indices start at zero, and [M_pq]
# represents the element at the [p]th row and [q]th column of [M]
def classical_layer(i, j, M=None, b=None):
    if M == None: M = util.rand_mat(j, i)
    if b == None: b = util.rand_vec(j)
    t = []
    for p in range(j):
        for q in range(i):
            t.append(M[p][q])
    for r in range(j):
        t.append(b[r])
    return _classical_from_params(i, j, t)
    
#Returns a classical neural network layer with [i] inputs, [j] outputs, and
# parameters [t]
def _classical_from_params(i, j, t):
    def c(tt, v):
        output = []
        for s in range(j):
            o = sum(tt[i * s + u] * v[u] for u in range(i)) + tt[i * j + s]
            output.append(o)
        return output
    def g(tt, v, e):
        output = []
        for p in range(j):
            for q in range(i):
                output.append(e[p] * v[q])
        for r in range(j):
            output.append(e[r])
        return output
    def h(tt, v, e):
        output = [sum(e[k] * tt[i * k + s] for k in range(j))
            for s in range(i)]
        return output
    def u(t, g, a):
        for i in range(len(t)):
            t[i] += a * g[i]
    return (c, t, g, h, u)

#The internal gradient function for an activation layer
def _g_a(t, v, e):
    return tuple()

#Returns a neural network layer that applies a function [a] to all of its [i]
# inputs and returns them as outputs.
def normal_layer(a, da):
    def h(tt, v, e):
        return tuple(e[i] * da(v[i]) for i in range(len(e)))
    def c(tt, v):
        return tuple(map(a, v))
    def u(t, g, a):
        return
    return (c, tuple(), _g_a, h, u)

import vqc_model as vm
import vqc_train as vt
#Converts a VQC model and a parameter set into a layer. The parameter object
# will be a 2-tuple (model, param).
def vqc_to_layer(model, param=None):
    if param == None: param = vm.rand_params(model)
    def c(tt, v):
        (m, p) = tt
        return vm.compute(m, v, p)
    def g(tt, v, e):
        (m, p) = tt
        return vt.grad_param(m, p, v, e)
    def h(tt, v, e):
        (m, p) = tt
        return vt.grad_prop(m, p, v, e)
    def u(tt, g, a):
        (m, p) = tt
        vt.update(p, g, a)
    return (c, (model, param), g, h, u)

def l2_norm_g(params):
    output = 0
    for k in params:
        output += params[k]**2
    return output**0.5

def l1_norm_g(params):
    output = 0
    for k in params:
        if params[k] > output: output = params[k]
        if -params[k] > output: output = -params[k]
    return output

def prunable_vqc_to_layer(model, param=None, s=0.5, h=0.05, mp=0.2):
    if param == None:
        param = {}
        for i in range(vm.layers(model)):
            c = list(vm.layer_params(model, i))
            random.shuffle(c)
            for j in range(int(s * len(c))):
                param[c[j]] = math.pi * (2 * random.random() - 1)
    def c(tt, v):
        (m, p, _) = tt
        return vm.compute(m, v, p)
    def g(tt, v, e):
        (m, p, _) = tt
        return vt.grad_param(m, p, v, e)
    def hh(tt, v, e):
        (m, p, _) = tt
        return vt.grad_prop(m, p, v, e)
    def u(tt, g, a):
        (m, p, hhh) = tt
        vt.update(p, g, a)
        norm = l1_norm_g(g)
        if norm < 0.007: 
            vt.prune(m, p, s, hhh, mp)
        print("Gradient norm: ", norm)
    return (c, (model, param, [h]), g, hh, u)

