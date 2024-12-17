
import qiskit as qk
import math
import numpy as np

#Helper function
def hadamard_transform_helper(v, i, j):
    if j - i <= 1: return
    s = (j - i) >> 1
    vt = v[i:i + s]
    vb = v[i + s:j]
    for k in range(s):
        v[i + k] = vt[k] - vb[k]
        v[i + s + k] = vt[k] + vb[k]
    hadamard_transform_helper(v, i, i + s)
    hadamard_transform_helper(v, i + s, j)

#Given a list of rotation angles [v] to be used in a uniformly controlled
# rotation gate, returns a list of rotation angles to be used in the
# decomposition of that uniformly controlled rotation gate, according to the
# decomposition implemented in [apply_ucry] below
def hadamard_transform(v):
    output = v[:]
    hadamard_transform_helper(output, 0, len(output))
    return [a / (len(v)) for a in output]

#Applies a uniformly controlled rotation with angles [theta], control qubit
# indices [c], target qubit index [t], to the quantum circuit [circuit].
def apply_ucry(theta, c, t, circuit, reverse=False):
    assert(len(theta) == 1 << len(c))
    angles = hadamard_transform(theta)
    def control_index(i):
        if i == 0: return -len(c)
        out = -1
        while i % 2 == 0:
            out -= 1
            i >>= 1
        return out
    for i in range(len(angles)):
        ci = control_index(i)
        for j in range(ci, -1): circuit.cx(c[j], c[j + 1])
        circuit.cx(c[-1], t)
        for j in range(-1, ci, -1): circuit.cx(c[j - 1], c[j])
        circuit.ry(angles[i], t)

#Returns a QuantumCircuit with a circuit that prepares the target state [state]
# on [n] qubits. Require len(state) == 2 ** n
def state_prep(state, n):
    prl = [0 for _ in range(len(state))]
    prl.extend(state)
    for i in range(len(state) - 1, 0, -1):
        prl[i] = math.sqrt(prl[2 * i] ** 2 + prl[2 * i + 1] ** 2)
    angles = [0 for _ in range(len(state))]
    for i in range(1, len(state)):
        angles[i] = 2 * math.acos(prl[2 * i] / prl[i]) \
        * np.sign(prl[2 * i + 1])
    circuit = qk.QuantumCircuit(n)
    circuit.ry(angles[1], 0)
    for i in range(1, n):
        apply_ucry(angles[1 << i: 1 << (i + 1)], [j for j in range(i)], 
        i, circuit)
    return circuit

#state = [-0.4922, 0.6499, -0.0581, -0.3333, -0.2743, -0.1947, 0.1181, -0.3063]
#print(state)
#circuit = state_prep(state, 3)
#simulator = qk.Aer.get_backend('statevector_simulator')
#result = simulator.run(circuit).result()
#print(result.get_statevector(circuit))
#print(circuit.draw())
