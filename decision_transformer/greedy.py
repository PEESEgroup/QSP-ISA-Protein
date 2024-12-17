
from data import Gate, apply_gate
from math import sqrt, atan
import numpy as np
from manual import rand_state, rand_complex_state, print_state
import cmath

DEBUG = False

#Helper functions
def bit_flip(number, index):
    return number ^ (1 << index)

def get_bit(number, index):
    return (number >> index) % 2

def max_index(array):
    return np.where(array == max(array))[0][0]

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

#Returns a list of (control, target) pairs where applying the corresponding
# cx gate would decrease the cx_dist of [number]
def candy_cx(number):
    start, end = one_range(number)
    if start == end: return []
    output = []
    for i in range(start, end):
        if get_bit(number, i) and not get_bit(number, i + 1):
            output.append((i, i + 1))
        if get_bit(number, i + 1) and not get_bit(number, i):
            output.append((i + 1, i))
    if len(output) == 0: return [(start + 1, start), (end - 1, end)]
    return output

#Returns the sublist of seq that maximize criterion
def filter_max(seq, criterion=None):
    if criterion is None: criterion = lambda a : a
    values = [criterion(s) for s in seq]
    max_value = max(values)
    output = [s for s in seq if criterion(s) == max_value]
    return output

#Same as filter_max, but returns the sublist that minimizes criterion
def filter_min(seq, criterion=None):
    if criterion is None: return filter_max(seq, lambda a : -a)
    return filter_max(seq, lambda a : -criterion(a))

def cx(control, target, number):
    if get_bit(number, control): return bit_flip(number, target)
    return number

class MergeDistTable:
    def __init__(self, n):
        self.n = n
        self.totals = [[None for j in range(1 << n)] for i in range(1 << n)]
        self.merges = [[None for j in range(1 << n)] for i in range(1 << n)]
        self.move_table = [[None for j in range(1 << n)] for i in range(1 << n)]

    #Returns the number of CX gates required to merge the amplitudes at
    # basis positions |i> and |j>, then merge them into the state |0>
    # Neither input may be 0, and the inputs must not be equal to each other
    def total_dist(self, i, j):
        if i == 0 or j == 0 or i == j: raise
        if self.totals[i][j] is not None: return self.totals[i][j]
        xor_string = i ^ j
        start, end = one_range(xor_string)
        if start == end:
            ls = [-abs(k - start) if get_bit(i, k) else -self.n \
              for k in range(self.n)]
            ls[start] = -self.n
            index = max_index(ls)
            extra_cx_count = -ls[index]
            ii = i
            jj = j
            for k in range(min(start, index) + 1, max(start, index)):
                ii = bit_flip(ii, k)
                jj = bit_flip(jj, k)
            output = extra_cx_count + min(cx_dist(ii), cx_dist(jj))
        else:
            candy = candy_cx(xor_string)
            def criterion(cx_gate):
                control, target = cx_gate
                ii = cx(control, target, i)
                jj = cx(control, target, j)
                return 1 + self.total_dist(ii, jj)
            output = min([criterion(cx) for cx in candy])
        self.totals[i][j] = output
        self.totals[j][i] = output
        return output
    #Returns the total number of CX gates required to merge two amplitudes
    # together into some basis vector. Neither input may be 0, and i must not
    # equal j
    def merge_dist(self, i, j):
        if i == 0 or j == 0 or i == j: raise
        if self.merges[i][j] is not None: return self.merges[i][j]
        xor_string = i ^ j
        start, end = one_range(xor_string)
        if start == end:
            values = [abs(k - start) if get_bit(i, k) else self.n \
              for k in range(self.n)]
            values[start] = self.n
            extra_cx_count = min(values)
            output = extra_cx_count
        else:
            candy = candy_cx(xor_string)
            def criterion(cx_gate):
                control, target = cx_gate
                ii = cx(control, target, i)
                jj = cx(control, target, j)
                return 1 + self.merge_dist(ii, jj)
            output = min([criterion(cx) for cx in candy])
        self.merges[i][j] = output
        self.merges[j][i] = output
        return output
    #Returns a list of (control, target) pairs such that applying the
    # corresponding cx gate makes progress towards merging the amplitudes at
    # |i> and |j>. Returns an empty list of the last gate for merging the
    # amplitudes is not a cx gate, but rather an ry gate. Requires:
    # i, j not equal to zero, i not equal to j
    def moves(self, i, j):
        if i == 0 or j == 0 or i == j: raise
        if self.move_table[i][j] is not None: return self.move_table[i][j]
        if self.merge_dist(i, j) == 1: 
            self.move_table[i][j] = []
            self.move_table[j][i] = []
            return []
        candy = []
        for k in range(self.n - 1):
            candy.append((k, k + 1))
            candy.append((k + 1, k))
        def criterion(cx_gate):
            control, target = cx_gate
            ii = cx(control, target, i)
            jj = cx(control, target, j)
            return self.total_dist(ii, jj)
        candy = filter_min(candy, criterion)
        def criterion(cx_gate):
            control, target = cx_gate
            ii = cx(control, target, i)
            jj = cx(control, target, j)
            return self.merge_dist(ii, jj)
        candy = filter_min(candy, criterion)
        self.move_table[i][j] = candy
        self.move_table[j][i] = candy
        return candy

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

#helper function
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
    tracker.apply_ry(start, angle)

#Input [state] is a real-valued, unit length numpy array
def prepare_state(state):
    assert(abs(np.vdot(state, state) - 1) < 0.001)
    tracker = StateTracker(state)

    #Phase 1: move maximal amplitude to the |0> position
    phase_1_candy = [i for i in range(tracker.n_qubits)]
    index = max_index(tracker.abs_state)
    while len(phase_1_candy) > 0:
        values = [tracker.abs_state[bit_flip(index, m)] for m in phase_1_candy]
        target = phase_1_candy[max_index(values)]
        other = bit_flip(index, target)
        if other > index: rotate_merge(tracker, other, index)
        else:
            rotate_merge(tracker, index, other)
            index = other
        phase_1_candy.remove(target)

    #Phase 2: repeatedly join the next biggest amplitude to the |0> position
    merge_table = MergeDistTable(tracker.n_qubits)
    while tracker.abs_state[0] ** 2 < 0.95:
        values = [a ** 2 / (1 + cx_dist(i + 1)) \
          for i, a in enumerate(tracker.abs_state[1:])]
        index = max_index(values) + 1
        #next, check for mergers
        def criterion(merger):
            if merger == 0: return 0
            if merger == index: return values[index - 1]
            denom = 1 + merge_table.total_dist(index, merger)
            return (tracker.abs_state[index] ** 2 + tracker.abs_state[merger] ** 2) / denom
        merger_values = [criterion(i) for i in range(1 << tracker.n_qubits)]
        merger = max_index(merger_values)
        if merger != index: 
            if DEBUG: print("merge", index, merger)
            i = index
            j = merger
            while True:
                candy = merge_table.moves(i, j)
                if len(candy) == 0: break
                control, target = candy[0]
                tracker.apply_cx(control, target)
                i = cx(control, target, i)
                j = cx(control, target, j)
            if cx_dist(i) > cx_dist(j):
                control_rotate_merge(tracker, i, j)
            else: 
                control_rotate_merge(tracker, j, i)
        elif cx_dist(index) == 0:
            rotate_merge(tracker, index, 0)
        else:
            candy = candy_cx(index)
            def criterion(cx_gate):
                control, target = cx_gate
                return tracker.abs_state[bit_flip(index, target)]
            candy = filter_max(candy, criterion)
            control, target = candy[0]
            control_rotate_merge(tracker, index, bit_flip(index, target))
    return tracker

if __name__ == "__main__":
    n_qubits = 9
    target_state = rand_complex_state(n_qubits)
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
