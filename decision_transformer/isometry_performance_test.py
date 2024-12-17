from isometry import isometry_prepare
import qiskit
import random
from manual import rand_complex_state
import time

n_qubits = 6
m = 2
count = 100

coupling_map = []
for i in range(n_qubits - 1):
    coupling_map.append([i, i + 1])
    coupling_map.append([i + 1, i])
layout = [i for i in range(n_qubits)]

states = []
for _ in range(count): states.append(rand_complex_state(n_qubits))

start = time.perf_counter()
for state in states:
    circuit = isometry_prepare(state, n_qubits, m, coupling_map, layout, compiled=True)
end = time.perf_counter()
print("Total elapsed time: ", end - start)
