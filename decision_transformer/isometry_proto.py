import qiskit
import numpy as np
from manual import rand_complex_state
from qclib.state_preparation import UCGInitialize

simulator = qiskit.Aer.get_backend("statevector_simulator")
def get_statevector(circ):
    c = qiskit.transpile(circ, simulator)
    sv = simulator.run(c).result().get_statevector()
    return sv

n_qubits = 6
l = 3
r = 3
m = 2
state = rand_complex_state(n_qubits)
mat = state.reshape((1 << r), (1 << l)).transpose()
u, s, vh = np.linalg.svd(mat)

circuit = qiskit.QuantumCircuit(n_qubits)
UCGInitialize.initialize(circuit, s[0:1 << m], qubits=[i for i in range(m)])
for i in range(m):
    circuit.cx(i, i + l)
circuit.isometry(u[:, 0:1 << m], [i for i in range(m)], [i for i in range(m, l)])
circuit.isometry(vh.transpose()[:, 0:1 << m], [i + l for i in range(m)], [i + l for i in range(m, r)])

coupling_map = []
for i in range(n_qubits - 1):
    coupling_map.append([i, i + 1])
    coupling_map.append([i + 1, i])
layout = [i for i in range(n_qubits)]
def compile(circ):
    return qiskit.transpile(circuit, basis_gates=['u', 'cx'], \
        coupling_map=coupling_map, optimization_level=3, \
        layout_method="trivial", initial_layout=layout)
transpiled = compile(circuit)
print(transpiled.count_ops().get("cx", 0))
circuit = qiskit.QuantumCircuit(n_qubits)
UCGInitialize.initialize(circuit, state)
transpiled = compile(circuit)
print(transpiled.count_ops().get("cx", 0))
circuit = qiskit.QuantumCircuit(n_qubits)
circuit.initialize(state)
transpiled = compile(circuit)
print(transpiled.count_ops().get("cx", 0))
