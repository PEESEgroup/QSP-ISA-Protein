import numpy as np
import random
import cmath
from math import atan, sin, cos
from qc import Gate, apply_gate, print_state

def normalize(v):
    norm = sum(abs(v) * abs(v))
    return v / (norm ** 0.5)

def phase_factor(p):
    return cmath.exp(1.0j * p)

def eig_x(abt):
    v1 = normalize(np.array([-abt[0][1], abt[0][0] - 1]))
    v2 = normalize(np.array([-abt[0][1], abt[0][0] + 1]))
    p = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
    u = p / cmath.sqrt(2) @ np.array([[1, 1], [1, -1]])
    return u

def zyz(mat):
    global_phase = cmath.phase(mat[0][0])
    z1 = cmath.phase(mat[1][0]) - global_phase
    y = 2 * atan(abs(mat[1][0]) / abs(mat[0][0]))
    z2 = cmath.phase(mat[0][1]) - global_phase + cmath.pi
    return [Gate.RZ(1, z1, 3), Gate.RY(1, y, 3), Gate.RZ(1, z2, 3)]

def prepare(state):
    #step d1, set the ratios ...
    A = state[4] * state[1] - state[0] * state[5]
    B = state[4] * state[3] - state[0] * state[7]
    C = state[6] * state[1] - state[2] * state[5]
    D = state[6] * state[3] - state[2] * state[7]
    A_c = A.conjugate()
    B_c = B.conjugate()
    C_c = C.conjugate()
    D_c = D.conjugate()
    
    a = -(C_c * D + A_c * B)
    b = A_c * A - B_c * B + C_c * C - D_c * D
    c = C * D_c + A * B_c

    r = (-b + cmath.sqrt(b * b - 4 * a * c)) / (2 * a)
    l = -(C_c * r + D_c) / (A_c * r + B_c)
    tr = atan(abs(r))
    pr = r / abs(r)
    tl = atan(abs(l))
    pl = l / abs(l)

    ml = np.array([[cos(tl), -sin(tl) * pl], [sin(tl), cos(tl) * pl]])
    mr = np.array([[cos(tr), -sin(tr) * pr], [sin(tr), cos(tr) * pr]])

    abt = ml @ mr.transpose().conj()
    global_phase = cmath.phase(abt[0][0])
    gamma = cmath.pi - cmath.phase(abt[1][1]) + global_phase

    ml = np.diag([1, phase_factor(gamma)]) @ ml / phase_factor(global_phase)
    abt = ml @ mr.transpose().conj()
    u = eig_x(abt)
    v = u.transpose().conj() @ ml
    
    gates_d1 = []
    gates_d1.extend(zyz(u))
    gates_d1.append(Gate.CX(0, 1, 3))
    gates_d1.extend(zyz(v))
    gates_d1.reverse()
    for gate in gates_d1: state = apply_gate(gate, state)
    
    #step d2: set qubit 3 to zero
    gates_d2 = []
    z1 = cmath.phase(state[0]) - cmath.phase(state[4])
    y1 = -2 * atan(abs(state[4]) / abs(state[0]))
    gates_d2.append(Gate.RZ(2, z1, 3))
    gates_d2.append(Gate.RY(2, y1, 3))
    state = apply_gate(Gate.RZ(2, z1, 3), state)
    state = apply_gate(Gate.RY(2, y1, 3), state)
    z2 = cmath.phase(state[2]) - cmath.phase(state[6])
    y2 = atan(abs(state[2]) / abs(state[6]))
    gates_d2.append(Gate.RZ(2, z2, 3))
    gates_d2.append(Gate.RY(2, y2, 3))
    gates_d2.append(Gate.CX(1, 2, 3))
    gates_d2.append(Gate.RY(2, -y2, 3))
    for gate in gates_d2[2:]: state = apply_gate(gate, state)
    
    #step sp2: finish state preparation
    gates_sp2 = []
    z1 = cmath.phase(state[0]) - cmath.phase(state[2])
    y1 = -2 * atan(abs(state[2]) / abs(state[0]))
    gates_sp2.append(Gate.RZ(1, z1, 3))
    gates_sp2.append(Gate.RY(1, y1, 3))
    state = apply_gate(Gate.RZ(1, z1, 3), state)
    state = apply_gate(Gate.RY(1, y1, 3), state)
    z2 = cmath.phase(state[1]) - cmath.phase(state[3])
    y2 = atan(abs(state[1]) / abs(state[3]))
    gates_sp2.append(Gate.RZ(1, z2, 3))
    gates_sp2.append(Gate.RY(1, y2, 3))
    gates_sp2.append(Gate.CX(0, 1, 3))
    gates_sp2.append(Gate.RY(1, -y2, 3))
    for gate in gates_sp2[2:]: state = apply_gate(gate, state)
    z3 = cmath.phase(state[0]) - cmath.phase(state[1])
    y3 = -2 * atan(abs(state[1]) / abs(state[0]))
    gates_sp2.append(Gate.RZ(0, z3, 3))
    gates_sp2.append(Gate.RY(0, y3, 3))
    for gate in gates_sp2[-2:]: state = apply_gate(gate, state)
    print_state(state)

state = np.array([random.gauss(0, 1) * phase_factor(2 * cmath.pi * random.random()) for _ in range(8)])
state = normalize(state)
prepare(state)
