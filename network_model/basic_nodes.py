
# basic_nodes.py is a module containing building block nodes for neural networks

import numpy as np
from node import Node
import math

#LinearTransformNode is a Node that takes in one vector, multiplies it by some
# matrix, then returns the vector result.
class LinearTransformNode(Node):
    #Constructs a linear transform node that takes in a vector of dimension
    # [in_dims] and returns a vector of output [out_dims]. Optional argument
    # [m], of type numpy array with [out_dims] rows and [in_dims] columns. If
    # [m] is specified, then this LinearTransformationNode will be the linear
    # transformation corresponding to matrix [m]. Otherwise, the matrix of this
    # linear transformation will be initialized with the glorot uniform
    # distribution
    def __init__(self, in_dims, out_dims, m=None):
        super().__init__((in_dims,), (out_dims,))
        if (m is None):
            x = math.sqrt(24 / (in_dims + out_dims))
            self.w = x * (np.random.rand(out_dims, in_dims) - 0.5)
        else:
            assert(m.shape == (out_dims, in_dims))
            self.w = m

    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (self.w.dot(inputs[0]),)
    
    def grad(self, inputs, output_gradient):
        self.check_output_shape(output_gradient)
        return (self.w.transpose().dot(output_gradient[0]),)
    
    def update(self, inputs, output_gradient, step_size):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        p_grad = output_gradient[0].reshape((self.output_dims(0), 1)) \
            .dot(inputs[0].reshape((1, self.input_dims(0))))
        m = self.w + step_size * p_grad
        output = LinearTransformNode(self.input_dims(0), self.output_dims(0), m)
        return output

    def update_batch(self, input_list, grad_list, step_size):
        assert(len(input_list) == len(grad_list))
        p_grad = 0
        for inputs, grad in zip(input_list, grad_list):
            self.check_input_shape(inputs)
            self.check_output_shape(grad)
            p_grad += grad[0].reshape((self.output_dims(0), 1)) \
                .dot(inputs[0].reshape((1, self.input_dims(0))))
        m = self.w + step_size * p_grad / len(input_list)
        return LinearTransformNode(self.input_dims(0), self.output_dims(0), m)

#BiasNode is a Node that takes in a vector, adds some other vector to it, and
# returns the result
class BiasNode(Node):
    #Constructs a BiasNode to take in a vector of dimension [dims]. Optional
    # argument [b]. If [b] is specified, then this BiasNode represents the
    # function that adds [b] to its input. Otherwise, this BiasNode will add
    # some vector chosen from the glorot uniform distribution.
    def __init__(self, dims, b=None):
        super().__init__((dims,), (dims,))
        if b is None:
            x = math.sqrt(12 / dims)
            self.b = x * (np.random.rand(dims) - 0.5)
        else:
            assert(b.shape == (dims,))
            self.b = b
    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (self.b + inputs[0],)
    def grad(self, _, output_gradient):
        self.check_output_shape(output_gradient)
        return output_gradient
    def update(self, _, output_gradient, step_size):
        self.check_output_shape(output_gradient)
        b = self.b + step_size * output_gradient[0]
        return BiasNode(self.input_dims(0), b)
    def update_batch(self, input_list, grad_list, step_size):
        assert(len(input_list) == len(grad_list))
        for grad in grad_list:
            self.check_output_shape(grad)
        p_grad = sum(grad[0] for grad in grad_list) / len(grad_list)
        b = self.b + step_size * p_grad
        return BiasNode(self.input_dims(0), b)

# SigmoidNode is a Node that takes in a single vector, applies the sigmoid
# function s(x) = 1 / (1 + e^-x) to each element individually, then returns 
# the result.
class SigmoidNode(Node):
    # Constructs a SigmoidNode that takes a vector of [dims] dimensions
    def __init__(self, dims):
        super().__init__((dims,), (dims,))

    def __call__(self, inputs):
        self.check_input_shape(inputs)
        output = []
        for x in inputs[0]:
            if x < -100: output.append(0)
            else: output.append(1 / (1 + math.exp(-x)))
        return (np.array(output),)

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        (s,) = self(inputs)
        s = s - s * s
        return (s * output_gradient[0],)

    def update(self, inputs, output_gradient, step_size):
        return self

    def update_batch(self, input_list, grad_list, step_size):
        return self

# TanhNode is a Node that takes in a single vector, applies the hyperbolic
# tangent function tanh(x) = (e^x - e^-x) / (e^x + e^-x) to each element
# individually, then returns the result
class TanhNode(Node):
    #Constructs a TanhNode that takes a vector of [dims] dimensions
    def __init__(self, dims):
        super().__init__((dims,), (dims,))

    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (np.tanh(inputs[0]),)

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        return (output_gradient[0] / (np.cosh(inputs[0]) ** 2),)

    def update(self, inputs, output_gradient, step_size):
        return self

    def update_batch(self, input_list, grad_list, step_size):
        return self

# CopyNode is a Node that takes in a vector and returns multiple copies of it.
class CopyNode(Node):
    #Constructs a CopyNode that takes a single vector of dimension [dims] and
    # returns [copies] copies of it
    def __init__(self, dims, copies):
        super().__init__((dims,), tuple(dims for _ in range(copies)))

    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return tuple(inputs[0] for _ in range(self.num_outputs()))

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        return (sum(x for x in output_gradient),)

    def update(self, inputs, output_gradient, step_size):
        return self
    
    def update_batch(self, input_list, grad_list, step_size):
        return self

#ConcatNode is a node that takes in several vectors and concatenates them into
# a single vector.
class ConcatNode(Node):
    #Constructs a ConcatNode that takes inputs as specified by [input_shape].
    # [input_shape] is a tuple of integers. The constructed node will take in
    # [n] inputs, where [n] is the length of [input_shape]; also, the dimension
    # of the [i]th array of the input will be the [i]th element of [input_shape]
    # So ConcatNode((1, 2, 3)) takes three vectors for input, the first of
    # dimension 1, the second of dimension 2, and the third of dimension 3.
    def __init__(self, input_shape):
        super().__init__(input_shape, (sum(x for x in input_shape),))
        self.breakpoints = [0]
        for x in input_shape:
            self.breakpoints.append(self.breakpoints[-1] + x)
    
    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (np.concatenate(inputs),)

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        output = []
        for i in range(self.num_inputs()):
            output.append( \
                output_gradient[0][self.breakpoints[i]:self.breakpoints[i + 1]])
        self.check_input_shape(output)
        return tuple(output)

    def update(self, inputs, output_gradient, step_size):
        return self

    def update_batch(self, input_list, grad_list, step_size):
        return self

#AddNode is a Node that takes in several vectors of the same dimension and
# returns their sum.
class AddNode(Node):
    #Constructs a new AddNode that takes [count] vectors of dimension [dims].
    def __init__(self, dims, count):
        super().__init__(tuple(dims for _ in range(count)), (dims,))

    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (sum(x for x in inputs),)

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        return tuple(output_gradient[0] for _ in range(self.num_inputs()))

    def update(self, inputs, output_gradient, step_size):
        return self

    def update_batch(self, input_list, grad_list, step_size):
        return self

#MultiplyNode is a Node that takes in two vectors of the same dimension and
# returns their elementwise product.
class MultiplyNode(Node):
    #Constructs a new MultiplyNode that takes in vectors of dimension [dims].
    def __init__(self, dims):
        super().__init__((dims, dims), (dims,))

    def __call__(self, inputs):
        self.check_input_shape(inputs)
        return (inputs[0] * inputs[1],)

    def grad(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        out1 = inputs[1] * output_gradient[0]
        out2 = inputs[0] * output_gradient[0]
        return (out1, out2)

    def update(self, inputs, output_gradient, step_size):
        return self

    def update_batch(self, input_list, grad_list, step_size):
        return self
