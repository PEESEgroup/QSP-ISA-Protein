import data
import math
import numpy as np
import cmath
import torch

LOAD = True
TRAIN = False

def bin_index(num, position):
    return num >> position & 1

def flip_index(num, position):
    if bin_index(num, position):
        return num - (1 << position)
    return num + (1 << position)

def gate_to_matrix(gate, n_qubits):
    dim = 2 ** n_qubits
    if gate.type == data.u:
        target = gate.qubits[0]
        angles = gate.angles
        one_qubit = [math.cos(angles[0] / 2), \
            -cmath.exp(1.0j * angles[2]) * math.sin(angles[0] / 2)], \
            [cmath.exp(1.0j * angles[1]) * math.sin(angles[0] / 2), \
            cmath.exp(1.0j * (angles[1] + angles[2])) * math.cos(angles[0] / 2)]
        output = [[0 for _ in range(dim)] for _ in range(dim)]
        for col in range(dim):
            if bin_index(col, target):
                output[col][col] = one_qubit[1][1]
                output[flip_index(col, target)][col] = one_qubit[0][1]
            else:
                output[col][col] = one_qubit[0][0]
                output[flip_index(col, target)][col] = one_qubit[1][0]
        return np.array(output)
    elif gate.type == data.cx:
        target = gate.qubits[1]
        control = gate.qubits[0]
        output = [[0 for _ in range(dim)] for _ in range(dim)]
        for col in range(dim):
            if bin_index(col, control):
                output[flip_index(col, target)][col] = 1
            else:
                output[col][col] = 1
        return np.array(output)

def to_torch(x):
    return torch.tensor(x).float()
#Takes a data.Gate object, converts it to a (tuple of) data vectors, returns
# the result. [n] is the number of qubits
#Current implementation: returns a 3-tuple of vectors. The first vector is the
# three rotation angles, the second vector represents the target qubit, one-hot
# encoded, and the third vector represents the control qubit. The control-qubit
# vector is an [n+1] dimensional vector, where each of the first [n] indices 
# represents a qubit, and the last index represents "no-control"
def process_gate(gate, n):
    if gate.type == data.u:
        angles = np.array(gate.angles)
        target_q = np.array([0 for _ in range(n)])
        target_q[gate.qubits[0]] += 1
        control_q = np.array([0 for _ in range(n + 1)])
        control_q[-1] += 1
        return (to_torch(angles), to_torch(target_q), to_torch(control_q))
    elif gate.type == data.cx:
        angles = np.array([math.pi, -math.pi / 2, math.pi / 2])
        target_q = np.array([0 for _ in range(n)])
        target_q[gate.qubits[1]] += 1
        control_q = np.array([0 for _ in range(n + 1)])
        control_q[gate.qubits[0]] += 1
        return (to_torch(angles), to_torch(target_q), to_torch(control_q))

def process_state(state):
    output = []
    for x in state:
        output.append(np.abs(x))
        output.append(np.angle(x))
    return to_torch(output)

#process data. states and expected zip to form state, gate pairs where the gate
# is the last gate to apply in the preparation of the state [state]
states = []
expected = []

for state, seq in data.data:
    state = np.array([0 for _ in range(2 ** data.n_qubits)])
    state[0] += 1
    for gate in seq:
        m_gate = gate_to_matrix(gate, data.n_qubits)
        state = m_gate.dot(state)
        states.append(process_state(state))
        expected.append(process_gate(gate, data.n_qubits))

print("Ready")
#Now define the neural networks
import torch.nn as nn

print("Loaded torch")
class AngleNN(nn.Module):
    def __init__(self):
        super(AngleNN, self).__init__()
        self.l1 = nn.Linear(16, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 16)
        self.l4 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(self.l4(x))
        return math.pi * x

class TargetNN(nn.Module):
    def __init__(self):
        super(TargetNN, self).__init__()
        self.l1 = nn.Linear(16, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 16)
        self.l4 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.nn.functional.softmax(self.l4(x), dim=0)
        return x
    
class ControlNN(nn.Module):
    def __init__(self):
        super(ControlNN, self).__init__()
        self.l1 = nn.Linear(16, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 16)
        self.l4 = nn.Linear(16, 4)
    
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.nn.functional.softmax(self.l4(x), dim=0)
        return x

mse = nn.MSELoss()
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def forward(self, angles, target, control, expected):
        a, t, c = expected
        return 4 * mse(a, angles) + mse(target, t) + mse(control, c)

angles = AngleNN()
target = TargetNN()
control = ControlNN()
a_save = "a_temp.pth"
c_save = "c_temp.pth"
t_save = "t_temp.pth"
if LOAD:
    angles.load_state_dict(torch.load(a_save))
    target.load_state_dict(torch.load(t_save))
    control.load_state_dict(torch.load(c_save))
if TRAIN:
    #now train
    import torch.optim as optim
    
    parameters = []
    parameters.extend(angles.parameters())
    parameters.extend(target.parameters())
    parameters.extend(control.parameters())
    optimizer = optim.SGD((x for x in parameters), lr=0.1)
    criterion = CustomLoss()
    
    print("Begin training")
    print("Data points: ", len(states))
    for i in range(10):
        print("Epoch ", i)
        for state, gate in zip(states, expected):
            i += 1
            optimizer.zero_grad()
            a, t, c = angles(state), target(state), control(state)
            loss = criterion(a, t, c, gate)
            loss.backward()
            optimizer.step()
        
        torch.save(angles.state_dict(), "a_temp.pth")
        torch.save(target.state_dict(), "t_temp.pth")
        torch.save(control.state_dict(), "c_temp.pth")

def gatev_to_matrix(gate, n):
    a, t, c = gate
    d = 2 ** n
    a = a.abs().tolist()
    t = t.tolist()
    c = c.tolist()
    target = t.index(max(t))
    control = c.index(max(c))
    if control == target or control == n:
        return gate_to_matrix(data.Gate(data.u, (target,), a), n)
    one_qubit = [math.cos(a[0] / 2), \
        -cmath.exp(1.0j * a[2]) * math.sin(a[0] / 2)], \
        [cmath.exp(1.0j * a[1]) * math.sin(a[0] / 2), \
        cmath.exp(1.0j * (a[1] + a[2])) * math.cos(a[0] / 2)]
    output = [[0 for _ in range(d)] for _ in range(d)]
    for col in range(d):
        if bin_index(col, control):
            if bin_index(col, target):
                output[col][col] = one_qubit[1][1]
                output[flip_index(col, target)][col] = one_qubit[0][1]
            else:
                output[col][col] = one_qubit[0][0]
                output[flip_index(col, target)][col] = one_qubit[1][0]
        else:
            output[col][col] = 1
    return np.array(output)

state = data.data[0][0]
for _ in range(20):
    print(state)
    s = process_state(state)
    gate = (angles(s), target(s), control(s))
    print(gate)
    m = gatev_to_matrix(gate, data.n_qubits)
    state = np.linalg.inv(m) @ state
