
# The functions.py module provides some mathematical functions that might be
# useful in constructing neural networks

from math import sqrt
import random

# Returns a random sample from the distribution [-x, x] where [x] is
# [sqrt(6 / (in_dims + out_dims))]
def glorot_uniform(in_dims, out_dims):
    x = sqrt(6 / (in_dims + out_dims))
    return random.uniform(-x, x)
