import qiskit
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
import math

error_1 = noise.depolarizing_error(0.001, 1)
error_2 = noise.depolarizing_error(0.01, 2)

#error_1 = noise.depolarizing_error(0, 1)
#error_2 = noise.depolarizing_error(0, 2)

noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

backend = AerSimulator(noise_model=noise_model)

circuit = qiskit.QuantumCircuit(2)
circuit.ry(math.pi / 2, 0)
circuit.cx(0, 1)
circuit.measure_all()

result = backend.run(circuit).result()
counts = result.get_counts(circuit)
print(counts)
