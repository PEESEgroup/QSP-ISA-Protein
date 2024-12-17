
#Support for data generators
# Implementation note: each generator is implemented as a 2-tuple 
# (seq, next), where seq is a list of indices and next is a function 
# that returns a data point for each index in [seq]. Each data point is in the
# format of a 2-tuple, (d, t), where [d] is the data and [t] is the
# classification. [t] is an integer between 0 and [n - 1], where [n] is the
# number of relevant classes.

def has_next(gen):
    return len(gen[0]) > 0

def next(gen):
    return gen[1](gen[0].pop())

#Returns two lists. The first is the list of the next [batch_size] samples in
# [gen] and the second is the list of their types
def next_batch(gen, batch_size):
    ds = []
    ts = []
    while has_next(gen) and len(ds) < batch_size:
        (d, t) = next(gen)
        ds.append(d)
        ts.append(t)
    assert len(ds) == len(ts)
    assert len(ds) == batch_size
    return (ds, ts)

import random

def cstr_type_to_str(t):
    if t == 0: return "Norm"
    if t == 1: return "F1"
    if t == 2: return "F2"
    if t == 3: return "F3"
    assert False

def cstr_path(t, j, train):
    if train: return "CSTR/Train/" + cstr_type_to_str(t) + "/" + str(j)
    else: return "CSTR/Test/" + cstr_type_to_str(t) + "/" + str(j)

def cstr_get(path):
    with open(path, "r") as f:
        output = []
        line = f.readline()
        while True:
            try:
                output.append(float(line))
                line = f.readline()
            except ValueError:
                break
    return tuple(output)
def cstr_next(i, batch_size, train):
    (t, j) = i
    first = j - batch_size + 1
    return (tuple(cstr_get(cstr_path(t, first + k, train)) 
        for k in range(batch_size)), t)

def cstr_gen(batch_size, train, shuffle=True):
    #Require batch_size <= 201, and then just take the points at indices i to
    # i + batch_size. 
    # indices [i] of the format [t, j] where [t] is the type and [j] is the
    # starting index.
    assert batch_size <= 201
    #batches are indexed by the index of the last data point
    seq = []
    tot = 1201 if train else 601
    for t in range(4):
        for j in range(200, tot):
            seq.append((t, j))
    if shuffle: random.shuffle(seq)
    def next(i):
        return cstr_next(i, batch_size, train)
    return (seq, next)

means = []
stdev = []
with open("CSTR/Train/meta.txt", "r") as f:
    line = f.readline()
    while len(line) > 0 and not line.isspace():
        (m, s) = line.split(",")
        means.append(float(m))
        stdev.append(float(s))
        line = f.readline()
means = tuple(means)
stdev = tuple(stdev)
def cstr_means():
    return means

def cstr_stdev():
    return stdev

def cstr_next_normal(i, batch_size, train):
    (sample, type) = cstr_next(i, batch_size, train)
    output = []
    for point in sample:
        output.append(tuple((point[i] - means[i]) / stdev[i]
            for i in range(len(point))))
    return (tuple(output), type)

def cstr_gen_normal(batch_size, train, shuffle=True):
    (seq, next) = cstr_gen(batch_size, train, shuffle)
    def nnext(i):
        return cstr_next_normal(i, batch_size, train)
    return (seq, nnext)

def cstr_gen_normal_weighted(batch_size, train, weights):
    seq = []
    tot = 1201 if train else 601
    for t in range(4):
        lt = []
        for j in range(200, tot):
            lt.append((t, j))
        random.shuffle(lt)
        lt = lt[:int(weights[t] * len(lt))]
        seq.extend(lt)
    random.shuffle(seq)
    def next(i):
        return cstr_next_normal(i, batch_size, train)
    return (seq, next)
