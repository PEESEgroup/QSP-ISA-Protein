from decision_transformer import DecisionTransformer
from data import encode_state, decode_state, apply_gate, Gate
import torch

import numpy as np
import random

model = DecisionTransformer(256, 6, 4, n_layers=4, n_heads=4)
model.load_state_dict(torch.load("save.pth"))
model.eval()

#state = np.array([random.gauss(0, 1) for _ in range(64)])
#norm = state.dot(state) ** 0.5
#state /= norm
#print(state)
state = np.array([0 for _ in range(64)])
state[0] += 1

num_cx = 0
fidelity_to_go = 0.95 - abs(state[0])

sequence = []
sequence.append((fidelity_to_go, num_cx))
sequence.append(encode_state(state))

def print_state(state):
    output = []
    output.append("[")
    for a in state: output.append("{:.3f}, ".format(a))
    output.append("]")
    print("".join(output))

for i in range(100):
    next_action = model.forward([sequence])[0][-1].tolist()
    gate = Gate.decode(next_action)
    print(i, ":", gate.gate_type, gate.target, gate.angle)
    if gate.is_cx(): num_cx -= 1
    new_state = apply_gate(gate, decode_state(sequence[-1]))
    #print_state(new_state)
    #print(abs(new_state[0]))
    if gate.is_stop(): break
    sequence.append(gate.encode())
    sequence.append((1 - abs(new_state[0]), num_cx))
    sequence.append(encode_state(new_state))
    if len(sequence) > 14: sequence = sequence[-14:]

final_state = sequence[-1]
print("Final fidelity: ", abs(final_state[0]) ** 2)
