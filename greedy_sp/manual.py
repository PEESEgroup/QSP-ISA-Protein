import numpy
import random
import cmath
from math import atan

def rand_state(n):
    state = numpy.array([random.gauss(0, 1) for _ in range(1 << n)])
    norm = state.dot(state) ** 0.5
    return state / norm

def rand_complex_state(n):
    amps = rand_state(n)
    phases = numpy.array([cmath.exp(cmath.pi * 2.0j * random.random()) \
      for _ in range(1 << n)])
    return amps * phases

def bin_string(n, bits):
    buf = []
    for i in range(bits - 1, -1, -1):
        buf.append(str((n >> i) % 2))
    return "".join(buf)

def print_state(state):
    bits = len(state).bit_length() - 1
    for i, a in enumerate(state):
        space = "  "
        if a < 0: space = " "
        print(bin_string(i, bits), space, a)

def matches(string, pattern):
    if len(string) != len(pattern): return False
    for i, p in enumerate(pattern):
        if p != "*" and p != string[i]: return False
    return True

def print_substate(state, pattern):
    bits = len(pattern)
    for i, a in enumerate(state):
        string = bin_string(i, bits)
        if matches(string, pattern):
            space = "  "
            if a < 0: space = " "
            print(string, space, a)

def norm_substate(state, pattern):
    bits = len(pattern)
    output = 0
    for i, a in enumerate(state):
        string = bin_string(i, bits)
        if matches(string, pattern):
            output += abs(a) ** 2
    return output

def angle(src, dest):
    dot = sum(s * d for s, d in zip(src, dest))
    src2 = sum(s * s for s in src)
    dest2 = sum(d * d for d in dest)
    return 0.5 * atan(2 * dot / (src2 - dest2))

def substate(state, pattern):
    bits = len(pattern)
    output = []
    for i, a in enumerate(state):
        string = bin_string(i, bits)
        if matches(string, pattern): output.append(a)
    return numpy.array(output)

if __name__ == "__main__":
    state = rand_state(5)
    print_state(state)
