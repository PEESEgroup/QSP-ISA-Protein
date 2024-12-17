
import tensorflow.keras as tfk
import numpy as np

means = []
stdev = []
with open("CSTR/Train/meta.txt", "r") as f:
    line = f.readline()
    while len(line) > 0 and not line.isspace():
        (m, s) = line.split(",")
        means.append(float(m))
        stdev.append(float(s))
        line = f.readline()
assert len(means) == len(stdev) == 7

def get(type, index, train):
    path = "CSTR/"
    if train: path += "Train/"
    else: path += "Test/"
    if type == 0: path += "Norm/"
    if type == 1: path += "F1/"
    if type == 2: path += "F2/"
    if type == 3: path += "F3/"
    path += str(index)
    output = []
    with open(path, "r") as f:
        line = f.readline()
        while len(line) > 0 and not line.isspace():
            output.append(float(line))
            line = f.readline()
    assert len(output) == 7
    x = tuple((o - m) / s for o, m, s in zip(output, means, stdev))
    y = np.array(tuple(int(type == i) for i in range(4))).reshape((1, 4))
    return (np.array(x).reshape((1, 7)), y)

class TrainingDataSeq(tfk.utils.Sequence):
    def __len__(self):
        return 4004
    def __getitem__(self, idx):
        assert 0 <= idx < 4004
        type = 0
        while idx >= 1001:
            type += 1
            idx -= 1001
        assert type < 4
        assert 0 <= idx <= 1000
        return get(type, idx + 200, True)

class TestingDataSeq(tfk.utils.Sequence):
    def __len__(self):
        return 1604
    def __getitem__(self, idx):
        assert 0 <= idx < 1604
        type = 0
        while idx >= 401:
            type += 1
            idx -= 401
        assert type < 4
        assert 0 <= idx <= 400
        return get(type, idx + 200, False)

model = tfk.Sequential([
    tfk.layers.Dense(30, activation="sigmoid"),
    tfk.layers.Dense(30, activation="sigmoid"),
    tfk.layers.Dense(4, activation="softmax")
    ])

model.compile(optimizer="adam", loss=tfk.losses.CategoricalCrossentropy())

def evaluate(m):
    print("Evaluating on the training set")
    gen = TrainingDataSeq()
    tr = 0
    cr = 0
    for i in reversed(range(len(gen))):
        (data, type) = gen[i]
        output = m(data)[0].numpy().tolist()
        tr += 1
        type = type[0].tolist()
        t = type.index(max(type))
        #print("Training Type ", t)
        #print("Output: ", output)
        if t == output.index(max(output)): 
            cr += 1
            #print("CORRECT")
        #else: print("INCORRECT")
    te = 0
    ce = 0
    gen = TestingDataSeq()
    for i in reversed(range(len(gen))):
        (data, type) = gen[i]
        output = m(data)[0].numpy().tolist()
        te += 1
        type = type[0].tolist()
        t = type.index(max(type))
        #print("Testing Type ", t)
        #print("Output: ", output)
        if t == output.index(max(output)):
            ce += 1
            #print("CORRECT")
        #else: print("INCORRECT")
    print("Training correctness: ", cr, " out of ", tr)
    print("Testing correctness: ", ce, " out of ", te)

evaluate(model)
for i in range(10):
    print("Training epoch ", i)
    model.fit(TrainingDataSeq())
    evaluate(model)

