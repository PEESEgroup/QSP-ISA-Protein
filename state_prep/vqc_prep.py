
import qiskit as qk
import numpy as np
from qiskit.circuit import Parameter
import math
import random

random.seed(42)
n = 6
d = 2 ** n
l = 6
n_params = (2 * l + 1) * n - l
adam = True
solexist = True
dynamic = True
values = np.array([0.0] * n_params)

vectors = [np.array([(i >> j) % 2 for j in range(n)]) for i in range(d)]

circuit = qk.QuantumCircuit(n)

params = []
for i in range(l):
    rparams = [Parameter(str(2 * i) + str(j)) for j in range(n)]
    cparams = [Parameter(str(2 * i + 1) + str(j)) for j in range(n - 1)]
    params.extend(rparams)
    params.extend(cparams)
    for j in range(n):
        circuit.ry(rparams[j], j)
    for j in range(n - 1):
        circuit.cry(cparams[j], j, j + 1)
#TODO: add the last layer of ry rotations [oops]

simulator = qk.Aer.get_backend("statevector_simulator")

def to_param_dict(vv):
    output = {k : v for k, v in zip(params, vv)}
    return output

def run(vv):
    param_dict = to_param_dict(vv)
    bc = circuit.bind_parameters(param_dict)
    circ = qk.transpile(bc, simulator)
    result = np.asarray(simulator.run(circ).result().get_statevector(circ)).real
    return result

def grad(vv, index):
    vv[index] += math.pi / 2
    g = (run(vv).dot(target_state).real) ** 2
    vv[index] -= math.pi
    g -= (run(vv).dot(target_state).real) ** 2
    vv[index] += math.pi / 2
    return g

target_state = np.array([random.gauss(0, 1) for _ in range(d)])
if solexist:
    target_values = [(random.random() - 0.5) * 2 * math.pi 
        for _ in range(n_params)]
    target_state = run(target_values)
else:
    norm = target_state.dot(target_state)
    target_state /= (norm ** 0.5)
result = run(values)
print("var state: ", result)
print("fidelity: ", result.dot(target_state).real ** 2)

step_size = 0.1
beta1 = 0.9
beta2 = 0.999
eps = np.array([1e-8] * n_params)
m = np.array([0] * n_params)
v = np.array([0] * n_params)
fidelity_list = [0.0000001]
grad_mag_list = [1]
step_dist_list = [0]
try:
    for iteration in range(500):
        print("Iteration ", iteration)
        gg = np.array([grad(values, i) for i in range(n_params)])
        gm = gg.dot(gg) ** 0.5
        if dynamic: 
            ggm = gm
            step_size = (1.1 - fidelity_list[-1]) / ggm / 5
        if adam:
            m = beta1 * m + (1 - beta1) * gg
            v = beta2 * v + (1 - beta2) * (gg * gg)
            mm = m / (1 - beta1)
            vv = v / (1 - beta2)
            adj = vv ** 0.5 + eps
            step = step_size * mm / adj
        else:
            step = step_size * gg
        values += step
        step_dist_list.append(step.dot(step) ** 0.5)
        result = run(values)
        fidelity_list.append(result.dot(target_state).real ** 2)
        grad_mag_list.append(gm)
        print("grad mag: ", gm)
        print("fidelity: ", fidelity_list[-1])
        print("step dist: ", step_dist_list[-1])
        if fidelity_list[-1] > 0.95: break
except:
    print("Exited early.")
result = run(values)
print("final fidelity: ", result.dot(target_state).real ** 2)

print("final params: ", values)
#print("target params: ", target_values)
print("final state: ", result)
print("target state: ", target_state)

with open("result.csv", "w") as f:
    f.write("Iteration,Fidelity,Gradient Magnitude, Step Size\n")
    for i in range(1, len(fidelity_list)):
        f.write(str(i) + "," + str(fidelity_list[i]) + "," 
        + str(grad_mag_list[i]) + "," + str(step_dist_list[i]) + "\n")

