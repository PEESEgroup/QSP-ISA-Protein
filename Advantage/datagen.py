
import os

def _dir(train, t):
    output = "CSTR-parsed/"
    if(train): output += "Train/"
    else: output += "Test/"
    if(t == 0): return output + "Norm/"
    return output + "F" + str(t) + "/"

#Returns non-normalized sample. [t] is the type (0 for normal, 1-3 for faults
#1-3 respectively), [i] is the sample number.
def sample(train, t, i):
    filename = _dir(train, t) + str(i)
    output = []
    with open(filename) as f:
        line = f.readline()
        while(len(line) > 1): 
            output.append(float(line))
            line = f.readline()
    return tuple(output)

#Returns the number of samples of type [t].
def count(train, t):
    output = 0
    for s in os.listdir(_dir(train, t)):
        if s[0] != ".": output += 1
    print(output)
    return output

mean = []
stdev = []

with open("CSTR-parsed/Train/meta.txt") as f:
    line = f.readline()
    while(len(line) > 1):
        m, s = line.split(",")
        mean.append(float(m))
        stdev.append(float(s))
        line = f.readline()

#[mean] is the mean vector of all the training samples
mean = tuple(mean)
#[stdv] is the element-wise standard deviation vector of all the training
# samples
stdev = tuple(stdev)

#Returns the normalized version of unnormalized vector [v]
def normalize(v):
    output = []
    for i in range(len(mean)):
        output.append((v[i] - mean[i]) / stdev[i])
    return tuple(output)
