import numpy as np
import random

def normalize(state):
    norm = sum(abs(state) * abs(state))
    return state / (norm ** 0.5)

def random_state(n):
    output = np.array([random.gauss(0, 1) for _ in range(1 << n)])
    return normalize(output)

def add_noise(state, mag):
    noise = np.array([random.gauss(0, 1) for _ in range(len(state))])
    noise = normalize(noise) * mag
    state += noise
    return normalize(state)
