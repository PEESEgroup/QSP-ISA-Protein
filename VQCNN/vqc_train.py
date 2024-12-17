import vqc_model as vm
import math
import random

#Training methods for VQC

def _reproduce(init, children, step, discrete, mono):
    output = []
    if mono: output.extend(init)
    for p in init:
        for _ in range(children):
            output.append(vm.child(p, step, discrete))
    return output

def _errors(candidates, model, batch, loss_fn):
    output = [0 for _ in range(len(candidates))]
    for (d, t) in batch:
        results = compute_sweep(model, d, candidates)
        for i, r in enumerate(results):
            output[i] += loss_fn(r, t)
    return output

#Returns the result of a single round of genetic algorithm.
# [model] is the model to train
# [init] is a list of initial parameter set candidates.
# [step] is the training step size.
# [batch] is the batch of data to train over. [batch] is a list of data points,
#   and each data point is a 2-tuple (d, t), where [d] is the model input and
#   [t] is the expected model output.
# [loss_fn] is a function that takes two arguments [e, a], and returns the loss
#   if [e] is the model prediction and [a] is the desired prediction.
# [children] is the number of children that each parameter set in [init]
#   produces.
# [survivors] is the number of parameter sets to return at the end. If the
#   number of candidates generated is less than [survivors], then every 
#   candidate is returned.
# [mono] is True if it should be possible to return an element of [init] as one
#   of the surviving candidates. This guarantees that the output list of
#   parameter sets is "at least as good" as the input list, but may be prone to
#   getting stuck in local minima
# [discrete] is True if the parameter adjustments should be in discrete
#   intervals of [step], otherwise, the adjustments are taken from a uniform
#   random distribution [-step, step]
def genetic_train(model, init, step, batch, loss_fn,
    children=8, survivors=4, mono=False, discrete=False):
    candidates = _reproduce(init, children, step, discrete, mono)
    if len(candidates) <= survivors: return candidates
    errors = _compute_errors(candidates, model, batch, loss_fn)
    indices = [i for i in range(len(candidates))]
    indices.sort(key=lambda x : errors[x])
    return tuple(candidates[i] for i in indices[:survivors])

def _cedist(p1, p2):
    def cyclic_square_dist(v1, v2):
        diff = (v2 - v1) % (2 * math.pi)
        if diff > math.pi: diff -= 2 * math.pi
        return diff * diff
    return sum(diff(p1[k1], p2[k2]) for k1, k2 in zip(p1, p2))

#Performs genetic algorithm also approximately maximizing the variety of the
# output pool. Variety is measured by computing dist_fn for all pairs of param
# sets in the output. If no distance function is specified, the euclidean
# distance is used.
def variety_train(model, init, step, batch, loss_fn, dist_fn=_cedist,
    variety_weight=1, children=4, survivors=8, mono=False, discrete=False):
    variety_weight /= len(init[0])
    candidates = _reproduce(init, children, step, discrete, mono)
    n = len(candidates)
    if n <= survivors: return candidates
    errors = _compute_errors(candidates, model, batch, loss_fn)
    m = float("inf")
    output = []
    while len(output) < survivors:
        i = index(min(errors))
        errors[i] = m
        c = candidates[i]
        output.append(c)
        for j in range(n):
            errors[j] -= variety_weight * dist_fn(candidates[j], c)
    return output

#Takes a model [model], a parameter set [params], and an error gradient with
# respect to the output [grad] on input [input], returns a dictionary [d]
# mapping each parameter [p] to d[err] / d[p]
def grad_param(model, params, input, grad):
    output = {}
    for param in params:
        #Compute the gradient, take the dot product, and update
        g = vm.grad_int(model, param, params, input)
        d = sum(a * b for a, b in zip(g, grad))
        output[param] = d
    return output

#Computes the error gradient with respect to the inputs given error gradient
# with respect to the outputs [grad].
def grad_prop(model, params, input, grad):
    output = []
    for i in range(len(input)):
        g = vm.grad_ext(model, i, input, params)
        output.append(sum(a * b for a, b in zip(g, grad)))
    return output

#Takes a parameter set params, and updates it in the direction of the gradient
# by an amount step. In gradient descent algorithms, [step] should be negative.
# This function modifies [params]
def update(params, grad, step):
    for p in params:
        params[p] += step * grad[p]
        while params[p] >= math.pi: params[p] -= 2 * math.pi
        if params[p] < -math.pi: params[p] += 2 * math.pi
        assert -math.pi <= params[p] < math.pi

def _layer_of(param, model):
    for i in range(vm.layers(model)):
        if param in vm.layer_params(model, i):
            return i
    print("Unrecognized param ", param)
    assert False

def pretty_print_params(params):
    keys = list(params.keys())
    str_keys = list(map(str, keys))
    sorted_keys = list(str_keys)
    sorted_keys.sort()
    print("{")
    for str_key in sorted_keys:
        key = keys[str_keys.index(str_key)]
        print(key, " : ", params[key])
    print("}")

_debug = True
def prune(model, params, s=0.5, h=None, mp=0.1):
    if _debug: print("Pruning!")
    if h == None: h = [0.05]
    if _debug: 
        print("Pre pruning: ")
        print("h is ", h[0])
        pretty_print_params(params)
    nl = vm.layers(model)
    pl = [[] for _ in range(nl)]
    pr = [[] for _ in range(nl)]
    for pp in params:
        layer = _layer_of(pp, model)
        pl[layer].append(pp)
        if -h[0] < params[pp] < h[0]: pr[layer].append(pp)
    for ll in pr:
        for pp in ll: params.pop(pp)
    n = sum(len(l) for l in pl)
    r = sum(len(l) for l in pr)
    if _debug: print("r is ", r)
    if r == 0: 
        h[0] += 0.1
        if h[0] > 0.5: h[0] -= 0.07
    if r > mp * n or h[0] > 1: h[0] /= 2
    rgr = [1 - len(pr[i]) / len(pl[i]) if len(pl[i]) > 0 else 0.1 
        for i in range(nl)]
    norm = sum(rgr)
    if norm == 0: return
    for i in range(len(rgr)): rgr[i] /= norm
    target_params = int(s * len(vm.param_set(model)))
    while len(params) < target_params:
        layer = random.random()
        i = 0
        while layer > rgr[i]: 
            layer -= rgr[i]
            if i + 1 >= len(rgr): break
            i += 1
        #Get a parameter from that layer, or if it's empty, skip
        c = set(vm.layer_params(model, i)).difference(set(params.keys()))
        if len(c) == 0:
            rgr[i] = 0
            norm = sum(rgr)
            if norm == 0: break
            for i in range(len(rgr)): rgr[i] /= norm
            assert -1e-4 < sum(rgr) - 1 < 1e-4
        else:
            params[c.pop()] = math.pi * (2 * random.random() - 1)
    if _debug: 
        print("Post pruning params: ")
        print("h is ", h[0])
        pretty_print_params(params)
