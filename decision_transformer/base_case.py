import numpy as np
import random
import cmath
from math import sin, cos, atan

random.seed(42)

state = np.array([random.gauss(0, 1) * cmath.exp(random.random() * 2.0j * cmath.pi) for _ in range(4)])
norm = sum(abs(state) * abs(state))
state /= norm ** 0.5
print(state)
print(cmath.phase(state[1]) - cmath.phase(state[0]))
n1 = abs(state[0]) * abs(state[0]) + abs(state[1]) * abs(state[1])
n1 = n1 ** 0.5
a = np.array([[state[0].conj(), state[1].conj()], [-state[1], state[0]]]) / n1

n2 = abs(state[2]) * abs(state[2]) + abs(state[3]) * abs(state[3])
n2 = n2 ** 0.5
b = np.array([[state[2].conj(), state[3].conj()], [-state[3], state[2]]]) / n2

x = a @ b.conj().transpose()

dd, u = np.linalg.eig(x)
print(x)
print(dd)
print(u)
print(u @ dd @ u.conj().transpose())
