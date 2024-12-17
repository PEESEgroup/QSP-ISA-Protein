import qiskit
import numpy as np
from manual import rand_complex_state
from qclib.state_preparation import UCGInitialize

simulator = qiskit.Aer.get_backend("statevector_simulator")
def get_statevector(circ):
    c = qiskit.transpile(circ, simulator)
    sv = simulator.run(c).result().get_statevector()
    return sv

state = rand_complex_state(6)
mat = state.reshape(8, 8).transpose()
u, s, vh = np.linalg.svd(mat)

circuit = qiskit.QuantumCircuit(6)
circuit.initialize(s, [0, 1, 2])
circuit.cx(0, 3)
circuit.cx(1, 4)
circuit.cx(2, 5)
circuit.isometry(u, [0, 1, 2], [])
circuit.isometry(vh.transpose(), [3, 4, 5], [])

coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 4], [4, 3], [3, 2], [2, 1], [1, 0]]
transpiled = qiskit.transpile(circuit, basis_gates=['u', 'cx'], coupling_map=coupling_map, optimization_level=3)
print(transpiled.count_ops().get("cx", 0))

circuit = qiskit.QuantumCircuit(6)
UCGInitialize.initialize(circuit, state)
transpiled = qiskit.transpile(circuit, basis_gates=['u', 'cx'], coupling_map=coupling_map, optimization_level=3)
print(transpiled.count_ops().get("cx", 0))
