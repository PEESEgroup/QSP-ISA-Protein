
#The node.py module contains the definition of an abstract Node. For
# implementation of specific nodes, see nodes.py

#A Node abstractly represents a parameterized differentiable function that 
# takes in a variable number of numpy vectors and outputs a variable number of 
# numpy vectors. Node also includes functionality for computing the gradient 
# with respect to the input, and for performing gradient descent with its
# internal parameters.
# The dimension of each input and output vector is fixed and specified at 
# construction time.
class Node:
    #Initializes this Node so that its inputs have dimension [input_dims] and
    # outputs have dimension [output_dims]. [input_dims] and [output_dims] are 
    # tuples of integers, required to have length at least 1. The length of 
    # [input_dims] is the number of input vectors to this Node; the elements 
    # of [input_dims] specifies the dimension of the input arrays, similar 
    # rules apply to [output_dims]
    # 
    # For example, __init__((7,), (3,)) creates a Node that takes one input
    # vector of dimension 7 and returns one output vector of dimension 3;
    # __init__((1, 2, 3), (4, 5, 6, 7)) creates a Node that takes three inputs,
    # the first with dimension 1, the second with dimension 2, the third with
    # dimension 3, and returns four vectors, the first with dimension 4, the
    # second with dimension 5, the third with dimension 6, and the fourth with
    # dimension 7.
    def __init__(self, input_dims, output_dims):
        assert(len(input_dims) > 0)
        assert(len(output_dims) > 0)
        for x in input_dims: assert(x > 0)
        for y in output_dims: assert(y > 0)
        self._inputs = input_dims
        self._outputs = output_dims

    # Calls this function with a tuple of vector inputs [inputs]. Requires that
    # [inputs] has the appropriate number and dimension of vectors. Returns a
    # tuple of vectors that are the result of calling this function on [inputs]
    def __call__(self, inputs):
        raise RuntimeError("Unimplemented")
    
    #Returns [g] where [g] is a tuple of vectors with the same shape as the 
    # input, also adjusting [inputs] in the direction of [g] will adjust the
    # output of calling this function, self(inputs), in the direction of 
    # [output_gradient]. Essentially, for small [a], 
    # self(inputs + a * grad(inputs, output_gradient))
    #   ~= self(inputs) + output_gradient
    def grad(self, inputs, output_gradient):
        raise RuntimeError("Unimplemented")

    #Returns a copy of self but with the internal parameters updated so as to 
    # adjust the output of calling self(inputs) in the direction of 
    # output_gradient by an amount step_size. Essentially, if [self'] is the 
    # output, then 
    #   [self'(inputs)] ~= [self(inputs)] + output_gradient * step_size
    def update(self, inputs, output_gradient, step_size):
        raise RuntimeError("Unimplemented")
    
    #Does the same thing as update, but for an entire batch of inputs and
    # output gradients, moves the parameters in the average direction dictated
    # by the inputs and output gradients.
    # [input_list] is a list of inputs and must not be empty
    # [grad_list] is a list of output gradients and must not be empty
    # [step_size] is a float and dictates how much to move the parameters
    def update_batch(self, input_list, grad_list, step_size):
        raise RuntimeError("Unimplemented")

    #Returns the number of input vectors this function takes
    def num_inputs(self):
        return len(self._inputs)

    #Returns the number of output vectors this function returns
    def num_outputs(self):
        return len(self._outputs)

    #Returns the dimension of the [index]th vector in this function's input
    def input_dims(self, index):
        return self._inputs[index]

    #Returns the dimension of the [index]th vector in this function's output
    def output_dims(self, index):
        return self._outputs[index]

    #Helper function
    def _check_shape(self, inputs, shape):
        assert(len(inputs) == len(shape))
        for v, d in zip(inputs, shape):
            assert(v.shape == (d,))

    #Checks that [inputs] is a valid input to this function. This base
    # implementation only checks that the number of inputs and dimension of
    # each input is correct; subclasses may add additional checks by overriding
    # this function. If the input shape is invalid, then an error will be 
    # thrown, otherwise, nothing is returned.
    def check_input_shape(self, inputs):
        self._check_shape(inputs, self._inputs)
    #Checks that the [output] object conforms to the shape of outputs promised
    # by this function. This base implementation checks that the number of
    # outputs and the dimension of each output is correct; subclasses may add
    # additional checks. If the output shape is incorrect, an error will be
    # thrown, otherwise, nothing is returned.
    def check_output_shape(self, outputs):
        self._check_shape(outputs, self._outputs)
