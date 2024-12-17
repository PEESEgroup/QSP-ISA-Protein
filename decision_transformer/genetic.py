from decision_transformer import DecisionTransformer
from data import encode_state, decode_state, apply_gate, Gate
import torch
import numpy as np
import random

model = DecisionTransformer(32, 3, 4, n_layers=4, n_heads=4)
model.load_state_dict(torch.load("save.pth"))
model.eval()

state = np.array([random.gauss(0, 1) for _ in range(8)])
norm = state.dot(state) ** 0.5
state /= norm
print(state)

num_cx = 5
fidelity_to_go = 0.95 - abs(state[0])
sequence = []
sequence.append((fidelity_to_go, num_cx))
sequence.append(encode_state(state))

gates = []
for i in range(100):
    next_action = model.forward([sequence])[0][-1].tolist()
    gate = Gate.decode(next_action)
    print(i, ":", gate.gate_type, gate.target, gate.angle)
    if gate.is_cx(): num_cx -= 1
    new_state = apply_gate(gate, decode_state(sequence[-1]))
    if gate.is_stop(): break
    sequence.append(gate.encode())
    sequence.append((1 - abs(new_state[0]), num_cx))
    sequence.append(encode_state(new_state))
    gates.append(gate)
    if len(sequence) > 14: sequence = sequence[-14:]

final_state = sequence[-1]
print("Final fidelity: ", abs(final_state[0]) ** 2)

def mutate(gates, amount):
    output = []
    for gate in gates: 
        new_gate = Gate(gate.gate_type, gate.target, \
            gate.angle + amount * random.gauss(0, 1), gate.n_qubits)
        output.append(new_gate)
    return output

def criterion(gates):
    current_state = state
    for g in gates: current_state = apply_gate(g, current_state)
    return abs(current_state[0])

candy = [gates]

for _ in range(20):
    for i in range(len(candy)):
        candy.append(mutate(candy[i], 0.1))
        candy.append(mutate(candy[i], 0.1))
        candy.append(mutate(candy[i], 0.1))
    candy.sort(reverse=True, key=criterion)
    candy = candy[0:3]
    print(criterion(candy[0]) ** 2)
