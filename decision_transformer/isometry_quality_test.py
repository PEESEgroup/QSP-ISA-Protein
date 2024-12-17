from isometry import isometry_prepare
from manual import rand_complex_state, rand_state
import random
import qiskit
import numpy as np

random.seed(42)

n_qubits = 9
m_range = [0, 1, 2, 3, 4]
count = 100

def compute_fidelity(circuit, state):
    sv = qiskit.quantum_info.Statevector.from_instruction(circuit)
    sv = np.asarray(sv)
    return abs(np.vdot(sv, state)) ** 2

coupling_map = []
for i in range(n_qubits - 1):
    coupling_map.append([i, i + 1])
    coupling_map.append([i + 1, i])
basis_gates = ["u", "cx"]

cx_counts = []
fidelity = []
for m in m_range:
    cx = []
    fids = []
    for i in range(count):
        print(m, i)
        state = rand_complex_state(n_qubits)
        circ = isometry_prepare(state, n_qubits, m)
        fids.append(compute_fidelity(circ, state))
        transpiled = qiskit.transpile(circ, basis_gates=basis_gates, coupling_map=coupling_map, optimization_level=3)
        cx.append(transpiled.count_ops().get("cx", 0));
    cx_counts.append(sum(cx) / count)
    fidelity.append(sum(fids) / count)

print(cx_counts)
print(fidelity)
