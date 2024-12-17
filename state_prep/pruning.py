import qiskit as qk
import numpy as np
from qiskit.circuit import Parameter
import math
import random

random.seed(42)
n = 6 #number of qubits
d = 2 ** n #dimension of quantum state spac3
l = 3 #number of layers
n_params = (2 * l + 1) * n - 1 #number of parameters in the model
t = 3 #number of parameters per layer trained at a time
threshold = 0.2 #pruning threshold
solexist = True

vectors = [np.array([(i >> j) % 2 for j in range(n)]) for i in range(d)]
rot_params = []
ent_params = []

circuit = qk.QuantumCircuit(n)
for i in range(l):
    rparams = [Parameter(str(2 * i) + str(j)) for j in range(n)]
    cparams = [Parameter(str(2 * i + 1) + str(j)) for j in range(n - 1)]
    rot_params.append(rparams)
    ent_params.append(cparams)
    for j in range(n):
        circuit.ry(rparams[j], j)
    for j in range(n - 1):
        circuit.cry(cparams[j], j, j + 1)
frot_params = [Parameter(str(2 * l) + str(j)) for j in range(n)]
rot_params.append(frot_params)
for j in range(n):
    circuit.ry(frot_params[j], j)

values = {}
for plist in rot_params:
    for p in plist:
        values[p] = 0
for plist in ent_params:
    for p in plist:
        values[p] = 0

simulator = qk.Aer.get_backend("statevector_simulator")

def run(values):
    bc = circuit.bind_parameters(values)
    circ = qk.transpile(bc, simulator)
    result = np.asarray(simulator.run(circ).result().get_statevector(circ)).real
    return result

def grad(values, param):
    values[param] += math.pi / 2
    g = (run(values).dot(target_state).real) ** 2
    values[param] -= math.pi
    g -= (run(values).dot(target_state).real) ** 2
    values[param] += math.pi / 2
    return g

target_state = np.array([random.gauss(0, 1) for _ in range(d)])
if solexist:
    target_values = {k:(random.random() - 0.5) * 2 * math.pi
        for k in values}
    target_state = run(target_values)
else:
    norm = target_state.dot(target_state)
    target_state /= (norm ** 0.5)

rot_params_training = []
ent_params_training = []
rot_params_pruned = []
ent_params_pruned = []

for rlist in rot_params:
    l = [i for i in range(len(rlist))]
    random.shuffle(l)
    rot_params_training.append(l[0:t])
    rot_params_pruned.append(l[t:])

for elist in ent_params:
    l = [i for i in range(len(elist))]
    random.shuffle(l)
    ent_params_training.append(l[0:t])
    ent_params_pruned.append(l[t:])

for epoch in range(10):
    for count in range(50):
        print("Epoch ", epoch)
        print("Iteration ", count)
        result = run(values)
        print("fidelity: ", result.dot(target_state).real ** 2)
        gv = {}
        for ilist, plist in zip(rot_params_training, rot_params):
            for i in ilist:
                p = plist[i]
                g = grad(values, p)
                gv[p] = g
        for ilist, plist in zip(ent_params_training, ent_params):
            for i in ilist:
                p = plist[i]
                g = grad(values, p)
                gv[p] = g
        for p in gv:
            values[p] += gv[p]
    for ilist, flist, plist in zip(rot_params_training, rot_params_pruned, \
        rot_params):
        #find which parameters can be pruned, prune them, add in a new parameter
        # to replace it
        prune = []
        for i in ilist:
            p = plist[i]
            while values[p] >= math.pi: values[p] -= 2 * math.pi
            while values[p] < -math.pi: values[p] += 2 * math.pi
            if abs(values[p]) < 0.2:
                prune.append(i)
        for i in prune:
            values[plist[i]] = 0
            ilist.remove(i)
            flist.append(i) #idea: pick the largest gradient instead of randomly
            ilist.append(flist.pop(0))
        random.shuffle(flist)
    for ilist, flist, plist in zip(ent_params_training, ent_params_pruned, \
        ent_params):
        prune = []
        for i in ilist:
            p = plist[i]
            while values[p] >= math.pi: values[p] -= 2 * math.pi
            while values[p] < -math.pi: values[p] += 2 * math.pi
            if abs(values[p]) < 0.2:
                prune.append(i)
        for i in prune:
            values[plist[i]] = 0
            ilist.remove(i)
            flist.append(i)
            ilist.append(flist.pop(0))
        random.shuffle(flist)

result = run(values)
print("final fidelity: ", result.dot(target_state).real ** 2)
print("final values: ", values)
