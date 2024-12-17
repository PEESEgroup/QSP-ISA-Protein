
import basic_nodes
import composite_nodes
import numpy as np
from node import Node

# Module for handling recurrent neural network functionality

# Creates an lstm unit that takes a single input of dimension [x_dim], holds
# a short term memory vector of dimension [h_dim]
# the long term memory cell has dimension = h_dim
# first input is long term cell memory (c_t)
# second input is the actual data vector (x_t)
# third input is the short term memory (h_t)
# first output is long term memory (c_{t + 1})
# second output is short term memory (h_{t + 1})
def lstm_unit(x_dim, h_dim):
    c_dim = x_dim + h_dim
    nodes = {
        "concat": basic_nodes.ConcatNode((x_dim, h_dim)),
        "copy4": basic_nodes.CopyNode(c_dim, 4),
        "copy2": basic_nodes.CopyNode(h_dim, 2),
        "mult1": basic_nodes.MultiplyNode(h_dim),
        "mult2": basic_nodes.MultiplyNode(h_dim),
        "mult3": basic_nodes.MultiplyNode(h_dim),
        "lt1": basic_nodes.LinearTransformNode(c_dim, h_dim),
        "b1": basic_nodes.BiasNode(h_dim),
        "s1": basic_nodes.SigmoidNode(h_dim),
        "lt2": basic_nodes.LinearTransformNode(c_dim, h_dim),
        "b2": basic_nodes.BiasNode(h_dim),
        "s2": basic_nodes.SigmoidNode(h_dim),
        "lt3": basic_nodes.LinearTransformNode(c_dim, h_dim),
        "b3": basic_nodes.BiasNode(h_dim),
        "s3": basic_nodes.SigmoidNode(h_dim),
        "lt4": basic_nodes.LinearTransformNode(c_dim, h_dim),
        "b4": basic_nodes.BiasNode(h_dim),
        "th4": basic_nodes.TanhNode(h_dim),
        "plus": basic_nodes.AddNode(h_dim, 2),
        "th": basic_nodes.TanhNode(h_dim)
    }

    edges = {
        ("in", 0) : ("mult1", 0),
        ("in", 1) : ("concat", 0),
        ("in", 2) : ("concat", 1),
        ("concat", 0) : ("copy4", 0),
        ("copy4", 0) : ("lt1", 0),
        ("lt1", 0) : ("b1", 0),
        ("b1", 0) : ("s1", 0),
        ("s1", 0) : ("mult1", 1),
        ("mult1", 0) : ("plus", 0),
        ("copy4", 1) : ("lt2", 0),
        ("lt2", 0) : ("b2", 0),
        ("b2", 0) : ("s2", 0),
        ("s2", 0) : ("mult2", 0),
        ("copy4", 2) : ("lt4", 0),
        ("lt4", 0) : ("b4", 0),
        ("b4", 0) : ("th4", 0),
        ("th4", 0) : ("mult2", 1),
        ("mult2", 0) : ("plus", 1),
        ("plus", 0) : ("copy2", 0),
        ("copy2", 0) : ("out", 0),
        ("copy4", 3) : ("lt3", 0),
        ("lt3", 0) : ("b3", 0),
        ("b3", 0) : ("s3", 0),
        ("s3", 0) : ("mult3", 1),
        ("copy2", 1) : ("th", 0),
        ("th", 0) : ("mult3", 0),
        ("mult3", 0) : ("out", 1)
    }

    return composite_nodes.GraphNode((h_dim, x_dim, h_dim), (h_dim, h_dim), 
        nodes, edges)

#LSTMNode is a node consisting of a series of LSTM cells. Takes in [t]
# vectors, where [t] is the number of timesteps, and returns [h_t] at the end
class LSTMNode(Node):
    #Constructs an LSTM node that takes in vectors of dimension x_dim, has
    # hidden state of dimension h_dim, and accepts [timesteps] timesteps.
    # if [lstm_cell] is specified, then the lstm unit within the node will
    # be set to [lstm_cell], otherwise parameters are initialized randomly.
    def __init__(self, x_dim, h_dim, timesteps, lstm_cell=None):
        super().__init__(tuple(x_dim for _ in range(timesteps)), (h_dim,))
        if lstm_cell is None: self._cell = lstm_unit(x_dim, h_dim)
        else: self._cell = lstm_cell
        self._h_dim = h_dim
        self._x_dim = x_dim
    
    def __call__(self, inputs):
        self.check_input_shape(inputs)
        h = np.array([0 for _ in range(self._h_dim)])
        c = np.array([0 for _ in range(self._h_dim)])
        for x in inputs:
            c, h = self._cell((c, x, h))
        return (h,)

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        inputs_list = []
        h = np.array([0 for _ in range(self._h_dim)])
        c = np.array([0 for _ in range(self._h_dim)])
        for x in inputs:
            inputs_list.append((c, x, h))
            c, h = self._cell((c, x, h))
        h_grad = output_gradient[0]
        c_grad = np.array([0 for _ in range(self._h_dim)])
        output = []
        for x in reversed(inputs_list):
            (c_grad, x_grad, h_grad) = self._cell.grad(x, (c_grad, h_grad))
            output.append(x_grad)
        output.reverse()
        return output

    def update(self, inputs, output_gradient, step_size):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        inputs_list = []
        h = np.array([0 for _ in range(self._h_dim)])
        c = np.array([0 for _ in range(self._h_dim)])
        for x in inputs:
            inputs_list.append((c, x, h))
            c, h = self._cell((c, x, h))
        grad_list = []
        h_grad = output_gradient[0]
        c_grad = np.array([0 for _ in range(self._h_dim)])
        for x in reversed(inputs_list):
            grad_list.append((c_grad, h_grad))
            (c_grad, _, h_grad) = self._cell.grad(x, (c_grad, h_grad))
        grad_list.reverse()
        return LSTMNode(self.input_dims(1), self._h_dim, self.num_inputs(), \
            self._cell.update_batch(inputs_list, grad_list, step_size))
    
    def update_batch(self, inputs, output_gradient, step_size):
        # Not going to need this function for now, but easy enough to
        # implement later if needed
        raise("Unimplemented")
