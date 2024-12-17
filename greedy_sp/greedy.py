
from qc import Gate, apply_gate, rand_state, print_state
from util import bit_flip, get_bit, max_index
from math import sqrt, atan
import numpy as np
import cmath

#DEBUG = __name__ == "__main__"
DEBUG = False

#Helper functions
def cflips(number, bits):
    output = []
    for i in range(bits):
        if get_bit(number, i):
            if i - 1 >= 0 and i - 1 not in output: output.append(i - 1)
            if i + 1 < bits: output.append(i + 1)
    return output

def one_range(number):
    top = int(number).bit_length()
    arr = np.array([get_bit(number, i) for i in range(top)])
    ones = np.where(arr == 1)[0]
    return ones[0], ones[-1]

def cx_dist(number):
    if number == 0: raise
    start, end = one_range(number)
    zeroes = 0
    for i in range(start + 1, end):
        if not get_bit(number, i): zeroes += 1
    return zeroes + end - start

def is_merge(number, index):
    start, end = one_range(number)
    if index < start or index > end: return True
    if index == start or index == end: return False
    return get_bit(number, index)

class StateTracker:
    def __init__(self, state):
        self.state = state
        self.gates = []
        self.abs_state = abs(state)
        self.phases = np.array([cmath.phase(s) for s in state])
        self.n_qubits = len(state).bit_length() - 1
    def apply_ry(self, target, angle):
        gate = Gate.RY(target, angle, self.n_qubits)
        self.apply_gate(gate)
    def apply_cx(self, control, target):
        gate = Gate.CX(control, target, self.n_qubits)
        self.apply_gate(gate)
    def apply_rz(self, target, angle):
        gate = Gate.RZ(target, angle, self.n_qubits)
        self.apply_gate(gate)
    def apply_gate(self, gate):
        self.gates.append(gate)
        self.state = apply_gate(gate, self.state)
        if type(self.state) is list: self.state = np.array(self.state)
        self.abs_state = abs(self.state)
        self.phases = np.array([cmath.phase(s) for s in self.state])
    def cx_count(self):
        output = 0
        for g in self.gates:
            if g.is_cx(): output += 1
        return output

def unify_phase(tracker, src, dest):
    start, end = one_range(src ^ dest)
    assert(start == end)
    dphase = tracker.phases[dest] - tracker.phases[src]
    if dest > src: dphase *= -1
    tracker.apply_rz(start, dphase)

def control_rotate_merge(tracker, src, dest):
    unify_phase(tracker, src, dest)
    start, end = one_range(src ^ dest)
    assert(start == end)
    control = start - 1 if get_bit(src, start - 1) else start + 1
    assert(get_bit(src, control))
    angle = atan(tracker.abs_state[dest] / tracker.abs_state[src])
    if dest > src: angle *= -1
    tracker.apply_ry(start, angle)
    tracker.apply_cx(control, start)
    tracker.apply_ry(start, -angle)

def rotate_merge(tracker, src, dest):
    unify_phase(tracker, src, dest)
    start, end = one_range(src ^ dest)
    assert(start == end)
    angle = -2 * atan(tracker.abs_state[src] / tracker.abs_state[dest])
    if dest > src: angle *= -1
    tracker.apply_ry(start, angle)

#Input [state] is a real-valued, unit length numpy array
def prepare_state(state):
    assert(abs(1 - np.vdot(state, state)) < 0.001)
    tracker = StateTracker(state)

    #Phase 1: move maximal amplitude to the |0> position
    phase_1_candy = [i for i in range(tracker.n_qubits)]
    index = max_index(tracker.abs_state)
    while len(phase_1_candy) > 0:
        values = [tracker.abs_state[bit_flip(index, m)] for m in phase_1_candy]
        target = phase_1_candy[max_index(values)]
        src = max(index, bit_flip(index, target))
        dest = min(index, bit_flip(index, target))
        rotate_merge(tracker, src, dest)
        phase_1_candy.remove(target)
        index = dest
    #Phase 2: repeatedly join the next biggest amplitude to the |0> position
    while tracker.abs_state[0] ** 2 < 0.95:
        if DEBUG: print("cx:", tracker.cx_count(), "  fidelity:", tracker.abs_state[0] ** 2)
        values = [a ** 2 / (1 + cx_dist(i + 1)) for i, a in enumerate(tracker.abs_state[1:])]
        index = max_index(values) + 1
        while cx_dist(index) > 0:
            candy = cflips(index, tracker.n_qubits)
            values = []
            for target in candy:
                amp = tracker.abs_state[index] ** 2
                amp += tracker.abs_state[bit_flip(index, target)] ** 2
                denom = 1 + int(is_merge(index, target)) + cx_dist(index)
                values.append(amp / denom)
            target = candy[max_index(values)]
            src_index = index
            dest_index = index
            if is_merge(index, target): src_index = bit_flip(src_index, target)
            else: dest_index = bit_flip(dest_index, target)
            control_rotate_merge(tracker, src_index, dest_index)
            index = dest_index
        rotate_merge(tracker, index, 0)
    return tracker

if __name__ == "__main__":
    n_qubits = 5
    target_state = rand_state(n_qubits)
    print_state(target_state)
    tracker = prepare_state(target_state)
    count = 0
    for g in tracker.gates:
        if g.is_cx(): count += 1
    print("n qubits:", n_qubits)
    print("cx count:", count)
    init_state = np.array([0 for _ in range(1 << n_qubits)])
    init_state[0] += 1
    for i in range(len(tracker.gates) - 1, -1, -1):
        gate = tracker.gates[i]
        init_state = apply_gate(gate.inverse(), init_state)
    print("final fidelity:", abs(np.vdot(init_state, target_state)) ** 2)
