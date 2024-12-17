import qiskit
import numpy as np
from manual import rand_complex_state
from qclib.state_preparation import UCGInitialize, IsometryInitialize

basis_gates = ["u", "cx"]
def isometry_prepare(state, n_qubits, m, coupling=None, layout=None, compiled=False):
    circuit = qiskit.QuantumCircuit(n_qubits)
    UCGInitialize.initialize(circuit, state);
#    l = n_qubits >> 1
#    r = n_qubits - l
#    mat = state.reshape(1 << r, 1 << l).transpose()
#    u, s, vh = np.linalg.svd(mat)
#    mr = [i for i in range(m)]
#    UCGInitialize.initialize(circuit, s[0:1 << m], qubits=mr)
#    for i in range(m):
#        circuit.cx(i, i + l)
#    circuit.isometry(u[:, 0:1 << m], mr, [i for i in range(m, l)])
#    circuit.isometry(vh.transpose()[:, 0:1 << m], [i + l for i in range(m)], \
#        [i + l for i in range(m, r)])
#    if not compiled: return circuit
#    if coupling is None:
#        coupling=[]
#        for i in range(n_qubits - 1):
#            coupling.append([i, i + 1])
#            coupling.append([i + 1, i])
#    if layout is None: layout = [i for i in range(n_qubits)]
    return qiskit.transpile(circuit, basis_gates=basis_gates, coupling_map=coupling, initial_layout=layout, optimization_level=3)

if __name__ == "__main__":
    simulator = qiskit.Aer.get_backend("statevector_simulator")
    def get_statevector(circ):
        return qiskit.quantum_info.Statevector.from_instruction(circ)
    
    n_qubits = 7
    m = 2
    
    coupling_map = []
    for i in range(n_qubits - 1):
        coupling_map.append([i, i + 1])
        coupling_map.append([i + 1, i])
    layout = [i for i in range(n_qubits)]
    
    def compile(circ):
        return qiskit.transpile(circuit, basis_gates=['u', 'cx'], \
            coupling_map=coupling_map, optimization_level=3, initial_layout=layout)
    
    state = rand_complex_state(n_qubits)
    circuit = isometry_prepare(state, n_qubits, m)
    transpiled = compile(circuit)
    sv = get_statevector(circuit)
    print(sum(a.conj() * b for a, b in zip(state, sv)))
    print(transpiled.count_ops().get("cx", 0))
    
    circuit = qiskit.QuantumCircuit(n_qubits)
    UCGInitialize.initialize(circuit, state)
    transpiled = compile(circuit)
    print(transpiled.count_ops().get("cx", 0))
    circuit = qiskit.QuantumCircuit(n_qubits)
    circuit.initialize(state)
    transpiled = compile(circuit)
    print(transpiled.count_ops().get("cx", 0))
