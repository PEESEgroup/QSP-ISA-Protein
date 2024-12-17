import random
import qiskit
import numpy
from bqskit import Circuit, compile
import nncx

state = numpy.array([random.gauss(0, 1) for _ in range(8)])
print(state)
norm = state.dot(state) ** 0.5
state /= norm

circuit = nncx.state_prep(state, 3)
circuit.draw()
circuit.qasm(filename="temp.qasm")

circuit = Circuit.from_file("temp.qasm")
compiled = compile(circuit)
compiled.save("output.qasm")

