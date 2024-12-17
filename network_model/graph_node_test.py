import basic_nodes
from composite_nodes import GraphNode
import numpy as np

nodes = {
    "w1" : basic_nodes.LinearTransformNode(2, 2),
    "b1" : basic_nodes.BiasNode(2),
    "s1" : basic_nodes.SigmoidNode(2),
    "w2" : basic_nodes.LinearTransformNode(2, 1),
    "b2" : basic_nodes.BiasNode(1),
    "s2" : basic_nodes.SigmoidNode(1),
}

edges = {
    ("in", 0) : ("w1", 0),
    ("w1", 0) : ("b1", 0),
    ("b1", 0) : ("s1", 0),
    ("s1", 0) : ("w2", 0),
    ("w2", 0) : ("b2", 0),
    ("b2", 0) : ("s2", 0),
    ("s2", 0) : ("out", 0),
}

g = GraphNode((2,), (1,), nodes, edges)
print(g((np.array([1, 1]),)))
