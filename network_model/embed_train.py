
#Using QNN to classify MNIST images

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

sample_3_train = []
sample_6_train = []
sample_3_test = []
sample_6_test = []

def process(x):
    small = np.array([[sum(x[7 * i + l][7 * j + m] for l in range(7) \
        for m in range(7)) for j in range(4)] for i in range(4)])
    scaled = small.flatten() / 255.0 / 49.0
    output = np.array([scaled[1], scaled[2], scaled[5], scaled[6], scaled[9],\
        scaled[10], scaled[13], scaled[14]])
    return output

count = 20
for x, y in zip(x_train, y_train):
    if y == 3 and len(sample_3_train) < count: sample_3_train.append(process(x))
    elif y == 6 and len(sample_6_train) < count: sample_6_train.append(process(x))
    if len(sample_3_train) >= count and len(sample_6_train) >= count: break

for x, y in zip(x_test, y_test):
    if y == 3 and len(sample_3_test) < count: sample_3_test.append(process(x))
    elif y == 6 and len(sample_6_test) < count: sample_6_test.append(process(x))
    if len(sample_3_test) >= count and len(sample_6_test) >= count: break

print("3 train: ", len(sample_3_train))
print("6 train: ", len(sample_6_train))

train_x = []
train_x.extend(sample_3_train)
train_x.extend(sample_6_train)
train_y = []
train_y.extend([1] * 20)
train_y.extend([-1] * 20)

import sklearn.svm

model = sklearn.svm.SVC(kernel='linear')
model.fit(train_x, train_y)
print(model.predict(train_x))
print(model.predict(sample_3_test))
print(model.predict(sample_6_test))

import qiskit as qk
from qiskit.circuit import Parameter
from qnn import Gate, QNNNode

#encoding circuit: RX(input), RY(param), CRZ(param), RX(input)
# similarity circuit: RX, RY, CRZ, RX, RX, CRZ, RY, RX
# first rx is input1 + layer1
# second rx is input1 - input2
circuit = qk.QuantumCircuit(8)
l1 = [Parameter("x1" + str(i)) for i in range(8)]
l2 = [Parameter("x2" + str(i)) for i in range(7)]
r1 = [Parameter("x3" + str(i)) for i in range(7)]
r2 = [Parameter("x4" + str(i)) for i in range(8)]
i1 = [Parameter("xi" + str(i)) for i in range(8)]
i2 = [Parameter("yi" + str(i)) for i in range(8)]

for i in range(8):
    circuit.rx(i1[i], i)
for i in range(8):
    circuit.ry(l1[i], i)
for i in range(7):
    circuit.cry(l2[i], i, i + 1)
for i in range(8):
    circuit.rx(i1[i], i)
for i in range(8):
    circuit.rx(i2[i], i)
for i in range(7):
    circuit.cry(r1[i], i, i + 1)
for i in range(8):
    circuit.ry(r2[i], i)
for i in range(8):
    circuit.rx(i2[i], i)

l1value = np.array([1.0 for _ in range(8)])
l2value = np.array([1.0 for _ in range(7)])

def compile_params(x, y, l1values, l2values):
    output = {}
    for p, xx in zip(i1, x):
        output[p] = xx
    for p, yy in zip(i2, y):
        output[p] = -yy
    for pl, pr, v in zip(l1, r2, l1values):
        output[pl] = v
        output[pr] = -v
    for pl, pr, v in zip(l2, r1, l2values):
        output[pl] = v
        output[pr] = -v
    return output

vectors = [np.array([(i >> j) % 2 for j in range(8)]) for i in range(256)]
def run(params):
    bc = circuit.bind_parameters(params)
    simulator = qk.Aer.get_backend("statevector_simulator")
    circ = qk.transpile(bc, simulator)
    result = simulator.run(circ).result().get_statevector(circ)
    exp = 1 - sum(r * v for r, v, in zip(result.probabilities(), vectors))
    output = 1
    for x in exp: output *= x
    return output

x = sample_3_train[0]
y = sample_6_train[0]
print(run(compile_params(x, y, l1value, l2value)))

import math

def l1_grad(x, y, l1value, l2value, index):
    params = compile_params(x, y, l1value, l2value)
    #param shift rule for the first layer
    params[l1[index]] += math.pi / 2
    g1 = run(params)
    print(g1)
    params[l1[index]] -= math.pi
    zzz = run(params)
    print(zzz)
    g1 -= zzz
    params[l1[index]] += math.pi / 2
    #param shift rule for the second layer
    params[r2[index]] += math.pi / 2
    g2 = run(params)
    print(g2)
    params[r2[index]] -= math.pi
    yyy = run(params)
    print(yyy)
    g2 -= yyy
    #add the results together
    print(g1 + g2)
    return g1 + g2

def l2_grad(x, y, l1value, l2value, index):
    params = compile_params(x, y, l1value, l2value)
    #param shift rule for first layer
    params[l2[index]] += math.pi / 2
    g1 = run(params)
    params[l2[index]] -= math.pi
    g1 -= run(params)
    params[l2[index]] += math.pi / 2
    #param shift rule for second layer
    params[r1[index]] += math.pi / 2
    g2 = run(params)
    params[r1[index]] -= math.pi
    g2 -= run(params)
    return g1 + g2

l1_grad_vec = np.array([l1_grad(x, y, l1value, l2value, i) for i in range(8)])
l2_grad_vec = np.array([l2_grad(x, y, l1value, l2value, i) for i in range(7)])

for i in range(3):
    l1value[i] += 0.01 * l1_grad_vec[i]
#l2value += 0.2 * l2_grad_vec

print(l1_grad_vec)
print(l2_grad_vec)
print(run(compile_params(x, y, l1value, l2value)))
