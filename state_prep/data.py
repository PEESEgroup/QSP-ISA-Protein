
#Gate types are:
u = "rot" #rotation gates
cx = "cx" #control x

class Gate:
    def __init__(self, gate_type, qubits, angles=None):
        self.type = gate_type
        if angles is None: self.angles = None
        else: self.angles = tuple(angles)
        self.qubits = tuple(qubits)
    def __str__(self):
        if self.type == u:
            return u + str(self.angles) + "[" + str(self.qubits[0]) + "]"
        elif self.type == cx:
            return cx + str(list(self.qubits))
        else: raise Exception()
    @staticmethod
    def from_string(s):
        if len(s) < 5: raise Exception()
        if s[0:2] == cx:
            i_start = s.index("[")
            i_end = s.index("]")
            (control, target) = s[i_start + 1: i_end].split(",")
            return Gate(cx, (int(control), int(target),))
        if s[0:3] == u:
            a_start = s.index("(")
            a_end = s.index(")")
            a_str = s[a_start + 1: a_end]
            a, b, c = a_str.split(",")
            q_start = s.index("[")
            q_end = s.index("]")
            q = s[q_start + 1: q_end]
            return Gate(u, (int(q),), (float(a), float(b), float(c)))

import qiskit as qk
from qiskit.providers.fake_provider import FakeQasmSimulator

simulator = FakeQasmSimulator()

def parse_circuit_instruction(ins):
    if ins.operation.name == "u1":
        angles = (0, 0, float(ins.operation.params[0]))
        qubits = ins.qubits[0].index
        return Gate(u, (qubits,), angles)
    elif ins.operation.name == "u2":
        params = ins.operation.params
        angles = (0, float(params[0]), float(params[1]))
        qubits = ins.qubits[0].index
        return Gate(u, (qubits,), angles)
    elif ins.operation.name == "u3":
        params = ins.operation.params
        angles = (float(params[0]), float(params[1]), float(params[2]))
        qubits = ins.qubits[0].index
        return Gate(u, (qubits,), angles)
    elif ins.operation.name == "cx":
        qubits = (ins.qubits[0].index, ins.qubits[1].index)
        return Gate(cx, qubits)
    else: raise Exception

def get_gate_sequence(state, n_qubits):
    circuit = qk.QuantumCircuit(n_qubits)
    length = 1 << n_qubits
    if len(state) > length: raise Exception()
    if len(state) < length: 
        state = np.array(state.extend([0]*(length - len(state))))
    circuit.initialize(state)
    circuit = qk.transpile(circuit, simulator)
    ins_list = []
    for x in circuit.data:
        ins_list.append(parse_circuit_instruction(x))
    return ins_list

#checks if the data file already exists. If it does, then parses the data
#file. Otherwise, generates data, saves it to a file.

import random
import numpy as np
import os
random.seed(42)

n_qubits = 3
n_data = 100
filename = "state_prep_data_n=" + str(n_qubits) + "_samples=" + str(n_data)
#data formatted as a list of (state, gate_seq) pairs where [state] is a
# 2^n_qubit length numpy array, gate_seq is a list of gates used to 
# prepare that state
data = []
if filename in os.listdir():
    #parse the data
    with open(filename) as f:
        line = f.readline()
        while len(line) > 1:
            state_str, seq_str = line.split("|")
            amp_str = state_str.split(";")
            state = np.array([float(x) for x in amp_str])
            gate_str = seq_str.split(";")
            gates = [Gate.from_string(gs) for gs in gate_str]
            data.append((state, gates))
            line = f.readline()
else:
    #generate the data
    for _ in range(n_data):
        state = np.array([random.gauss(0, 1) for _ in range(8)])
        norm = sum(x * x for x in state)
        state /= norm ** 0.5
        gate_seq = get_gate_sequence(state, n_qubits)
        data.append((state, gate_seq))
    #write the data
    #format: each data point takes one line, form [data] | [gate_seq]
    # [data] will be the list of quantum amplitudes, semicolon separated
    # [gate_seq] will be the list of gates, semicolon separated.
    with open(filename, "w") as f:
        for state, seq in data:
            state_buffer = [str(a) for a in state]
            seq_buffer = [str(g) for g in seq]
            output = ";".join(state_buffer) + "|" + ";".join(seq_buffer)
            f.write(output)
            f.write("\n")
