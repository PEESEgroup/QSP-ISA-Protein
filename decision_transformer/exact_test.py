import qiskit
from manual import rand_complex_state
from qclib.state_preparation import UCGInitialize

n_qubits = 10
coupling_map = []
for i in range(n_qubits - 1):
    coupling_map.append([i, i + 1])
    coupling_map.append([i + 1, i])

data = []
for i in range(100):
    print(i)
    circuit = qiskit.QuantumCircuit(n_qubits)
    state = rand_complex_state(n_qubits)
    UCGInitialize.initialize(circuit, state)
    transpiled = qiskit.transpile(circuit, basis_gates=['u', 'cx'], coupling_map=coupling_map, optimization_level=3)
    data.append(transpiled.count_ops().get("cx", 0))

print(min(data))
print(max(data))    
