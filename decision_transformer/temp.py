import cmath
import numpy as np
from manual import rand_complex_state, bin_string
import random

random.seed(42)

state = rand_complex_state(5)
state = rand_complex_state(5)

with open("test.q", "w") as f:
    for i, a in enumerate(state):
        f.write(bin_string(i, 5))
        f.write(" ")
        f.write(str(abs(a)))
        f.write(" ")
        f.write(str(cmath.phase(a)))
        f.write("\n")

