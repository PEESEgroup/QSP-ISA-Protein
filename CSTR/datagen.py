
import data
import random

norm = 0
f1 = 1
f2 = 2
f3 = 3

types = (norm, f1, f2, f3)
#train_counts = tuple(data.num_samples(True, t) for t in types)
#test_counts = tuple(data.num_samples(False, t) for t in types)
train_counts = tuple(1201 for t in types)
test_counts = tuple(601 for t in types)
train_tot = sum(train_counts)
test_tot = sum(test_counts)

#Returns (d, t) where [d] is the [n]th training sample and [t] is its type
def train_sample(n):
    type = 0
    while n >= train_counts[type]:
        n -= train_counts[type]
        type += 1
    if n < 200: type = 0
    return (data.get_sample(True, type, n), type)

#Same as train_sample, but for testing data
def test_sample(n):
    type = 0
    while n >= test_counts[type]:
        type += 1
        n -= test_counts[type]
    return (data.get_sample(False, type, n), type)

#Returns a new generator. If [train], then the iterator iterates over training
# data, otherwise, it iterates over testing data.
def get_gen(train):
    count = train_tot if train else test_tot
    sequence = [i for i in range(count)]
    random.shuffle(sequence)
    return (train, sequence)

#Returns a new generator over training data, but with data from each dataset 
# weighted according to [weights]. 
# [weights] is (a, b, c, d) with a, b, c, d all between 0 and 1
# inclusive, and the data sampled will include only [a] of the normal data, 
# [b] of the fault type 1 data, [c] of the fault type 2 data, and [d] of the
# fault type 3 data.
def get_gen_weighted(weights):
    output = []
    assert len(weights) == len(types)
    buf = 0
    for i, p in enumerate(weights):
        l = [j for j in range(train_counts[i])]
        random.shuffle(l)
        l = l[:int(p * train_counts[i])]
        output.extend(map(lambda x : buf + x, l))
        buf += train_counts[i]
    random.shuffle(output)
    return (True, output)
    
#Checks whether the data generator [gen] has more elements to return
def has_next(gen):
    return len(gen[1]) > 0

#Returns the next element in [gen]. Throws exception if there are no more
# elements in [gen].
def next(gen):
    (t, s) = gen
    n = s.pop()
    if t:
        return train_sample(n)
    else:
        return test_sample(n)

#Returns the normalized version of [d]: with means subtracted then divided by
# standard deviation.
def normalize(d):
    m = data.get_mean()
    s = data.get_stdev()
    assert len(d) == len(m)
    assert len(d) == len(s)
    return tuple((d[i] - m[i]) / s[i] for i in range(len(d)))

#Returns a batch of [count] samples from [gen]. Each sample is in the format
# (s, t) where [s] is the sample and [t] is its type
def next_batch(gen, count):
    output = []
    while has_next(gen) and len(output) < count:
        output.append(next(gen))
    return output
