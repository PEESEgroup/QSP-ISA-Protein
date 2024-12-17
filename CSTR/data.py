import os

_parsing = True
_computing = False
#Given a CSV file [file], parses the data, copies, one sample per file, into
# directory [dir]. The file names will be [0], [1], ..., [n - 1] where [n] is 
# the number of data samples. Each file will be formatted with exactly one
# feature on each line.
# [s] is the data separator used in the spreadsheet. If not specified, [s] is
# a comma, and the input file is assumed to be in .csv format.
# Assumes each data row starts with an integer specifying which data point in
# the spreadsheet it is, and that each of [0], [1], ..., [n-1] is represented
# exactly once. Also, [dir] is assumed not to have a "/" as its final character.

def parse(file, dir, s=","):
    with open(file, "r") as f:
        line = f.readline()
        while len(line) > 0:
            tks = line.split(s)
            try:
                i = int(tks[0])
                with open(dir + "/" + str(i), "w") as g:
                     for j in range(1, len(tks)):
                         float(tks[j])
                         g.write(tks[j] + "\n")
            except ValueError:
                pass
            line = f.readline()
    
_norm = 0
_f1 = 1
_f2 = 2
_f3 = 3
_meta = 4

#Returns the path file of the [i]th sample of a data group. If [train], then the
# data group accessed is a training group, otherwise the data group accessed is
# a testing group. Type is an integer 0, 1, 2, or 3; if 0, then normal data is
# accessed, if 1, then fault 1 type, if 2, then fault 2 type, otherwise, fault
# 3 type. If [i] is not specified, then the directory where all the data samples
# are is returned instead. This function does not guarantee that the returned
# path leads to an existing file, if either the data has not yet been parsed
# or if the index [i] is out of bounds.
def _get_path(train, type, i=None):
    path = "CSTR-parsed/"
    if train: path += "Train/"
    else: path += "Test/"
    if type == _norm: path += "Norm"
    elif type == _f1: path += "F1"
    elif type == _f2: path += "F2"
    elif type == _f3: path += "F3"
    elif type == _meta: 
        path += "meta.txt"
        assert i == None
    if i == None: return path
    else: return path + "/" + str(i)

#Counts the number of samples of the specified type. [train] is a boolean for
# whether or not to look at the training data (looks at testing data otherwise),
# type is an integer either 0, 1, 2, or 3, 0 for normal data, 1 for fault 1, 2
# for fault 2, 3 for fault 3.
def num_samples(train, type):
    return len(os.listdir(_get_path(train, type)))
    
#Returns the [i]th data sample as a tuple. The data group that the sample is
# taken from is specified by (train, type), as in the other functions.
def get_sample(train, type, i):
    path = _get_path(train, type, i)
    output = []
    with open(path, "r") as f:
        line = f.readline()
        while len(line) > 0 and not line.isspace():
            output.append(float(line))
            line = f.readline()
    return tuple(output)

if _parsing:
    parse("CSTR-Train/Norm.csv", "CSTR-parsed/Train/Norm")
    parse("CSTR-Train/Fault1.csv", "CSTR-parsed/Train/F1")
    parse("CSTR-Train/Fault2.csv", "CSTR-parsed/Train/F2")
    parse("CSTR-Train/Fault3.csv", "CSTR-parsed/Train/F3")
    parse("CSTR-Test/Norm.csv", "CSTR-parsed/Test/Norm")
    parse("CSTR-Test/Fault1.csv", "CSTR-parsed/Test/F1")
    parse("CSTR-Test/Fault2.csv", "CSTR-parsed/Test/F2")
    parse("CSTR-Test/Fault3.csv", "CSTR-parsed/Test/F3")

mean = None
stdev = None
if _parsing or _computing:
    ty_v = (_norm, _f1, _f2, _f3)
    n = 0
    mean = [0, 0, 0, 0, 0, 0, 0]
    stdev = [0, 0, 0, 0, 0, 0, 0]
    tr = True
    for ty in ty_v:
        nn = num_samples(tr, ty)
        n += nn
        for i in range(nn):
            l = get_sample(tr, ty, i)
            assert len(l) == len(mean)
            for j in range(len(l)): mean[j] += l[j]
    for i in range(len(mean)): mean[i] /= n
        
    for ty in ty_v:
        nn = num_samples(tr, ty)
        for i in range(nn):
            l = get_sample(tr, ty, i)
            assert len(l) == len(stdev)
            for j in range(len(l)): stdev[j] += (l[j] - mean[j]) ** 2
    for i in range(len(stdev)): 
        stdev[i] /= n
        stdev[i] = stdev[i] ** 0.5
    
    with open(_get_path(tr, _meta), "w") as f:
        for i, s in enumerate(mean):
            f.write(str(s) + ", " + str(stdev[i]) + "\n")
    mean = tuple(mean)
    stdev = tuple(stdev)
else:
    mean = []
    stdev = []
    with open(_get_path(True, _meta), "r") as f:
        line = f.readline()
        while len(line) > 1:
            m, s = line.split(", ")
            mean.append(float(m))
            stdev.append(float(s))
            line = f.readline()
    mean = tuple(mean)
    stdev = tuple(stdev)

assert mean != None
assert stdev != None
assert len(mean) == len(stdev) == 7
#Getter methods for getting the mean and stdev of the training set.
def get_mean():
    return mean

def get_stdev():
    return stdev

    
