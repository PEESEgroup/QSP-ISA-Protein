import qiskit
from qclib.state_preparation import UCGInitialize
import random
import numpy
import cmath
import sys
import time

def rand_complex_state(n):
    amps = numpy.array([random.gauss(0, 1) for _ in range(1 << n)])
    norm = sum(amps * amps) ** 0.5
    amps /= norm
    phases = numpy.array([random.random() * 2 * cmath.pi for _ in range(1 << n)])
    return amps * phases

n_qubits = int(sys.argv[1])

coupling_map = []
for i in range(n_qubits - 1):
    coupling_map.append([i, i + 1])
    coupling_map.append([i + 1, i])

states = [rand_complex_state(n_qubits) for _ in range(100)]

start = time.time()
for i in range(100):
    circuit = qiskit.QuantumCircuit(n_qubits)
    state = states[i]
    UCGInitialize.initialize(circuit, state)
    transpiled = qiskit.transpile(circuit, basis_gates=['u', 'cx'], coupling_map=coupling_map, optimization_level=1)
end = time.time()
print("Total time taken (s):", end - start)
