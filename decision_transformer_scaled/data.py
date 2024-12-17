
import random
import math
import cmath
import qiskit
import numpy as np

random.seed(42)

#Convenince function
def one_hot(i, n):
    return [int(j == i) for j in range(n)]

#Class representing quantum gates. Use the static methods to construct new
# objects. 
class Gate:
    types = ["rx", "ry", "rz", "cx", "xc", "stop"]
    def __init__(self, gate_type, target, angle, n_qubits):
        self.gate_type = gate_type
        self.target = target
        self.angle = angle
        self.n_qubits = n_qubits

    def is_cx(self):
        return self.gate_type in ["cx", "xc"]
    
    def is_stop(self):
        return self.gate_type == "stop"

    def inverse(self):
        if self.is_cx() or self.is_stop(): return self
        return Gate(self.gate_type, self.target, -self.angle, self.n_qubits)

    def encode(self):
        i1 = Gate.types.index(self.gate_type)
        if i1 == -1: raise RuntimeError()
        output = one_hot(i1, len(Gate.types))
        output.extend(one_hot(self.target, self.n_qubits))
        output.extend([self.angle])
        return output

    def to_string(self):
        if self.is_stop(): return "STOP"
        if self.is_cx():
            control = self.target
            target = control + 1
            if self.gate_type == "xc": 
                control -= 1
                target += 1
            return "CX control=" + str(control) + " target=" + str(target)
        return self.gate_type.upper() + " target=" + str(self.target) + \
        " angle=" + str(self.angle)

    @staticmethod
    def decode(vec):
        vec_type = vec[0:len(Gate.types)]
        vec_index = vec[len(Gate.types):-1]
        gate_type = Gate.types[vec_type.index(max(vec_type))]
        target = vec_index.index(max(vec_index))
        return Gate(gate_type, target, vec[-1], len(vec_index))

    #CX requires control and target to be on nearest neighbor qubits
    @staticmethod
    def CX(control, target, n_qubits):
        if target == control + 1:
            return Gate("cx", control, 0, n_qubits)
        if control == target + 1:
            return Gate("xc", target, 0, n_qubits)
        raise RuntimeError()

    @staticmethod
    def RY(target, angle, n_qubits):
        return Gate("ry", target, angle, n_qubits)

    @staticmethod
    def RX(target, angle, n_qubits):
        return Gate("rx", target, angle, n_qubits)

    @staticmethod
    def RZ(target, angle, n_qubits):
        return Gate("rz", target, angle, n_qubits)

    @staticmethod
    def STOP(n_qubits):
        return Gate("stop", -1, 0, n_qubits)

#Helper function
def apply_rotation_gate(gate, state, index):
    output = [None for _ in range(len(state))]
    for i in range(len(output)):
        if output[i] is not None: continue
        e0 = state[i]
        e1 = state[i + (1 << index)]
        output[i] = gate[0][0] * e0 + gate[0][1] * e1
        output[i + (1 << index)] = gate[1][0] * e0 + gate[1][1] * e1
    return output

#Helper function
def apply_cx_gate(state, control, target):
    output = [None for _ in range(len(state))]
    for i in range(len(output)):
        if (i >> control) % 2: output[i] = state[i ^ (1 << target)]
        else: output[i] = state[i]
    return output

#Given Gate object [gate] and list of complex amplitudes [state] representing a
# quantum state, returns a new list of complex amplitudes representing the
# result of applying quantum gate [gate] to quantum state [state]. Does not
# modify [state].
def apply_gate(gate, state):
    if gate.gate_type == "rx":
        mat = [[math.cos(gate.angle / 2), -1.0j * math.sin(gate.angle / 2)], \
               [-1.0j * math.sin(gate.angle / 2), math.cos(gate.angle / 2)]]
        return apply_rotation_gate(mat, state, gate.target)
    if gate.gate_type == "ry":
        mat = [[math.cos(gate.angle / 2), -math.sin(gate.angle / 2)], \
               [math.sin(gate.angle / 2), math.cos(gate.angle / 2)]]
        return apply_rotation_gate(mat, state, gate.target)
    if gate.gate_type == "rz":
        mat = [[cmath.exp(-1.0j * gate.angle / 2), 0], \
               [0, cmath.exp(1.0j * gate.angle / 2)]]
        return apply_rotation_gate(mat, state, gate.target)
    if gate.gate_type == "cx":
        return apply_cx_gate(state, gate.target, gate.target + 1)
    if gate.gate_type == "xc":
        return apply_cx_gate(state, gate.target + 1, gate.target)
    if gate.gate_type == "stop":
        return [a for a in state]

#Helper function
def validate_apply_gate():
    circuit = qiskit.QuantumCircuit(2)
    circuit.ry(1, 0)
    circuit.rz(2, 1)
    circuit.rx(3, 0)
    circuit.rx(4, 1)
    circuit.cx(0, 1)
    circuit.ry(2, 0)
    circuit.ry(1, 1)
    circuit.cx(1, 0)
    simulator = qiskit.Aer.get_backend("statevector_simulator")
    sv1 = np.array(simulator.run(circuit).result().get_statevector())
    sv2 = one_hot(0, 4)
    for gate in [Gate.RY(0, 1, 2), Gate.RZ(1, 2, 2), Gate.RX(0, 3, 2), 
    Gate.RX(1, 4, 2), Gate.CX(0, 1, 2), Gate.RY(0, 2, 2), Gate.RY(1, 1, 2), 
    Gate.CX(1, 0, 2)]:
        sv2 = apply_gate(gate, sv2)
    sv2 = np.array(sv2)
    delta = sv1 - sv2
    assert(delta.dot(delta) < 0.00001)

#Encodes a quantum state [state] into a list of real numbers.
def encode_state(state):
    output = [abs(s) for s in state]
    output.extend([cmath.phase(s) / math.pi for s in state])
    return output

#Takes the encoded form of a quantum state (output from encode_state function)
# and returns it back to a list of complex amplitudes. Guaranteed that for
# all lists of complex numbers [state], 
#   state == decode_state(encode_state(state))
# up to rounding error
def decode_state(state):
    amps = state[0:len(state) >> 1]
    phases = state[len(state) >> 1:]
    output = [a * cmath.exp(p * math.pi * 1.0j) for a, p in zip(amps, phases)]
    return output
 
#Generates a random sequence of gates for [n_qubits] qubits, including exactly
# [num_cx] CX gates.
def generate_gate_sequence(num_cx, n_qubits):
    output = []
    for i in range(n_qubits):
        types = random.sample(["rx", "ry", "rz"], 2)
        a1 = random.uniform(0, math.pi)
        a2 = random.uniform(-math.pi, math.pi)
        output.append(Gate.RY(i, a1, n_qubits))
        output.append(Gate.RZ(i, a2, n_qubits))
    for _ in range(num_cx):
        control = random.randint(0, n_qubits - 2)
        target = control + 1
        if random.randint(0, 1):
            control += 1
            target -= 1
        a1, a2, a3, a4 = [random.uniform(-math.pi, math.pi) for _ in range(4)]
        output.append(Gate.CX(control, target, n_qubits))
        output.append(Gate.RY(control, a1, n_qubits))
        output.append(Gate.RZ(control, a2, n_qubits))
        output.append(Gate.RY(target, a3, n_qubits))
        output.append(Gate.RX(target, a4, n_qubits))
    return output

#Helper function
def noisy_start(n_qubits, factor=0.1):
    a = np.array([int(i == 0) for i in range(1 << n_qubits)]).astype("complex")
    noise = np.array([random.gauss(0, 1) * cmath.exp(random.random() * 2.0j * cmath.pi) for _ in range(1 << n_qubits)])
    noise *= factor
    a += noise
    norm = sum(abs(x) ** 2 for x in a) ** 0.5
    a /= norm
    return a

#Returns a 2-tuple (training, testing). Both [training] and [testing] are lists
# of sequences of the form reward, state, gate, reward, state, gate, ...
# See the demo file for an example
def generate_dataset(n_qubits, count, frac_testing=0.2, noise=0.1, max_cx=None):
    training = []
    testing = []
    if max_cx is None: max_cx = 2 * n_qubits
    for i in range(count):
        for num_cx in range(max_cx):
            #Alignment as follows:
            # index:  0      1      2       3
            # gates:  <stop> g0     g1      g2
            # states: |0>    g0|0>  g1g0|0> g2g1g0|0>
            # reward: 0, 0   f(g0|0>)
            gates = generate_gate_sequence(num_cx, n_qubits)
            states = [noisy_start(n_qubits, factor=noise).tolist()] 
            rewards = [(0, 0)]
            cx_count = 0
            start_fid = abs(states[0][0])
            for g in gates:
                s = apply_gate(g, states[-1])
                states.append(s)
                fid_to_go = start_fid - abs(s[0])
                if g.is_cx(): cx_count += 1
                rewards.append((fid_to_go, cx_count))
            assert(cx_count == num_cx)
            gates.insert(0, Gate.STOP(n_qubits))
            zipped = []
            for g, s, r in zip(gates, states, rewards): 
                zipped.extend([g.inverse().encode(), encode_state(s), r])
            zipped.reverse()
            assert(len(zipped) % 3 == 0)
            if i > frac_testing * count: training.append(zipped)
            else: testing.append(zipped)
    return training, testing

#Helper function
def validate_dataset(dataset):
    for sequence in dataset:
        assert(len(sequence) % 3 == 0)
        for i in range(0, len(sequence) - 3, 3):
            (f1, c1), s1, a1 = sequence[i:i + 3]
            (f2, c2), s2, a2 = sequence[i + 3:i + 6]
            s1 = decode_state(s1)
            s2 = decode_state(s2)
            a1 = Gate.decode(a1)
            delta = np.array(apply_gate(a1, s1)) - np.array(s2)
            assert(delta.dot(delta) < 0.0001)
            assert(abs(f1 - f2 - (abs(s2[0]) - abs(s1[0]))) < 0.0001)
            assert(c1 - c2 == int(a1.is_cx()))
            if Gate.decode(a2).is_stop(): 
                assert(abs(f2) < 0.0001)
                assert(c2 == 0)

#Converts [data] output from generate_dataset function into a list of batches.
#Each batch contains [batch_size] subsequences taken from sequences in the
# original dataset, where each subsequences contains [previews] previews of past
# reward, state, gate triples, plus the reward and state token for the current
# time step.
def make_batches(data, n_qubits, previews=4, batch_size=60):
    inputs = []
    outputs = []
    for seq in data:
        seq_in = seq[:-1]
        for i in range(0, len(seq) - 3, 3):
            seq_input = seq_in[i:i + previews * 3 + 2]
            while len(seq_input) < previews * 3 + 2:
                seq_input.append(Gate.STOP(n_qubits).encode())
                seq_input.extend(seq_input[-3:-1])
            seq_output = [seq_input[j] for j in range(2, len(seq_input), 3)]
            if i + len(seq_input) >= len(seq): 
                seq_output.append(Gate.STOP(n_qubits).encode())
            else: seq_output.append(seq[i + len(seq_input)])
            inputs.append(seq_input)
            outputs.append(seq_output)
    order = [i for i in range(len(inputs))]
    random.shuffle(order)
    output = []
    for i in range(0, len(order) - batch_size, batch_size):
        batch_x = [inputs[order[j]] for j in range(i, i + batch_size)]
        batch_y = [outputs[order[j]] for j in range(i, i + batch_size)]
        output.append((batch_x, batch_y))
    return output

import os
import json

#Helper function
def write_batches_to_file(batches, filename):
    with open(filename, "w") as f:
        for batch in batches:
            f.write(json.dumps(batch) + "\n")

if __name__ == "__main__" and "training_data.json" not in os.listdir():
    validate_apply_gate()
    num_qubits = 5
    max_cx = 20
    training_data, testing_data = generate_dataset(num_qubits, 1000)
    validate_dataset(testing_data)
    training_batches = make_batches(training_data, num_qubits, previews=4)
    testing_batches = make_batches(testing_data, num_qubits, previews=4)
    write_batches_to_file(training_batches, "training_data.json")
    write_batches_to_file(testing_batches, "testing_data.json")

#Class for reading batches that have been written to a file.
# Eg:
#     stream = BatchStream("testing_data.json")
#     model = <model>
#     loss_fn = <loss function>
#     loss = 0
#     while stream.hasNext():
#         x, y = stream.next()
#         prediction = model(x)
#         loss += loss_fn(prediction, y)
#     print("Total loss: ", loss)
class BatchStream:
    def __init__(self, filename):
        self.f = open(filename)
        self.buffer = None
    def _pull(self):
        if self.buffer is not None: return
        line = self.f.readline()
        if len(line) < 2: return
        batch = json.loads(line)
        self.buffer = batch
    def hasNext(self):
        self._pull()
        return self.buffer is not None
    def next(self):
        self._pull()
        if self.buffer is None: raise
        output = self.buffer
        self.buffer = None
        return output
    def close(self):
        self.f.close()
