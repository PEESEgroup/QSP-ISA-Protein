import rnn
import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter
import math
import random

random.seed(42)
target_state = np.array([0.0, 1.0, 0.0, 0.0])

circuit = qk.QuantumCircuit(2)
a = Parameter("a")
b = Parameter("b")
circuit.ry(a, 0)
circuit.ry(b, 0)

simulator = qk.Aer.get_backend("statevector_simulator")

def run(pval, target):
    pd = {
        a : pval[0],
        b : pval[1],
    }
    bc = circuit.bind_parameters(pd)
    circ = qk.transpile(bc, simulator)
    result = np.asarray(simulator.run(circ).result().get_statevector(circ)).real
    return result.dot(target) ** 2

#runs the lstm, returns the gradient vector (ga, gb) for input values pval.
# Output formatted as numpy array.
def grad(pval, target):
    pc = [pval[0], pval[1]]
    pc[0] += math.pi / 2
    ga = run(pc, target)
    pc[0] -= math.pi
    ga -= run(pc, target)
    pc[0] += math.pi / 2
    pc[1] += math.pi / 2
    gb = run(pc, target)
    pc[1] -= math.pi
    gb -= run(pc, target)
    return np.array([ga, gb])

lstm_units = [rnn.lstm_unit(1, 2)]
steps = 5

#runs the lstm, returns the list of inputs to each lstm unit as a tuple (h, x)
#first element of the list will have cell state = 0, hidden state = 0, x = fid
#last element is the outputs of the last lstm unit, so the length of the output
#is [steps] + 1
def run_lstm(target):
    output = []
    cell = np.array([0, 0])
    values = np.array([0, 0])
    fid = np.array([run(values, target)])
    output.append((cell, fid, values))
    for _ in range(steps):
        cell, values = lstm_units[-1]((cell, fid, values))
        fid = np.array([run(values, target)])
        output.append((cell, fid, values))
    return output

#given a starting parameter value pval,updates the lstm by gradient descent to
# maximize the best fidelity generated
def train_lstm(step_size, target):
    outputs = run_lstm(target)
    max_index = 1
    max_value = outputs[1][1]
    for i, (c, x, h) in enumerate(outputs):
        if i == 0: continue
        if x > max_value:
            max_index = i
            max_value = x
    gc = np.array([0, 0])
    gh = grad(outputs[max_index][2], target)
    input_batch = []
    grad_batch = []
    for index in range(max_index - 1, -1, -1):
        grad_batch.append((gc, gh))
        input_batch.append(outputs[index])
        (gc, gx, gh) = lstm_units[-1].grad(outputs[index], (gc, gh))
        gh += gx[0] * grad(outputs[index][2], target)
    lstm_units.append(
    lstm_units[-1].update_batch(input_batch, grad_batch, step_size))

#generates a random vector of real numbers of unit length
def rand_target_state(d):
    a = random.random() * math.pi / 2
    b = random.random() * math.pi / 2
    return np.array([math.cos(a) * math.cos(b), math.cos(a) * math.sin(b),
    math.sin(a) * math.cos(b), math.sin(a) * math.sin(b)])

print(tuple(x[1][0] for x in run_lstm(target_state)))
for _ in range(1000):
    t = rand_target_state(4)
    train_lstm(1, t)
    out = tuple(x[1][0] for x in run_lstm(target_state))
    print(max(out))
