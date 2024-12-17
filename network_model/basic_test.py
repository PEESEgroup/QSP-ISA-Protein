import basic_nodes
from composite_nodes import SequentialNode
import numpy as np

square_fn = SequentialNode([basic_nodes.CopyNode(1, 2), basic_nodes.MultiplyNode(1)])

print(square_fn((np.array([1]),)))
print(square_fn((np.array([2]),)))
print(square_fn((np.array([3]),)))
print(square_fn((np.array([4]),)))
print(square_fn((np.array([5]),)))
print(square_fn((np.array([6]),)))
print(square_fn.grad((np.array([1]),), (np.array([1]),)))
print(square_fn.grad((np.array([2]),), (np.array([1]),)))
print(square_fn.grad((np.array([3]),), (np.array([1]),)))
print(square_fn.grad((np.array([4]),), (np.array([1]),)))
print(square_fn.grad((np.array([5]),), (np.array([1]),)))
print(square_fn.grad((np.array([6]),), (np.array([1]),)))
