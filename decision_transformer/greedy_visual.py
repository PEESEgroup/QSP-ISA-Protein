from greedy2 import StateTracker, prepare_state
from manual import rand_complex_state
from math import sqrt
import qiskit

def to_qiskit(tracker):
    circuit = qiskit.QuantumCircuit(tracker.n_qubits)
    gates = []
    for gate in tracker.gates:
        if gate.gate_type == "rx":
            circuit.rx(gate.angle, gate.target)
        elif gate.gate_type == "ry":
            circuit.ry(gate.angle, gate.target)
        elif gate.gate_type == "rz":
            circuit.rz(gate.angle, gate.target)
        elif gate.gate_type == "cx":
            circuit.cx(gate.target, gate.target + 1)
        elif gate.gate_type == "xc":
            circuit.cx(gate.target + 1, gate.target)
        else: raise "incomplete case match"
    return circuit

state = rand_complex_state(5)
tracker = prepare_state(state)
circuit = to_qiskit(tracker)
print(circuit.draw())
