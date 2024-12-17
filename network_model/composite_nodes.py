
# composite_nodes.py contains functionality for nodes that are composed of other
# nodes.

from node import Node

#SequentialNode is a Node that is a composition of several other nodes.
# For example, a typical dense neural network layer with input dimension 3 and
# output dimension 4 might be represented as
#   [SequentialNode([LinearTransformNode(3, 4), BiasNode(4), SigmoidNode(4)])]
class SequentialNode(Node):
    #Constructs a SequentialNode where args is the list of Nodes to pass the
    # input through. If the input is [a, b, c], then self(input) will be
    # evaluated as c(b(a(input))). Required that [*args] contains at least one
    # argument.
    def __init__(self, args):
        super().__init__(args[0]._inputs, args[-1]._outputs)
        self.nodes_list = []
        for arg in args:
            if len(self.nodes_list) != 0:
                assert(self.nodes_list[-1]._outputs == arg._inputs)
            self.nodes_list.append(arg)

    def __call__(self, inputs):
        output = inputs
        for node in self.nodes_list:
            output = node(output)
        return output

    def grad(self, inputs, output_gradient):
        input_list = [inputs]
        for node in self.nodes_list[0:-1]:
            input_list.append(node(input_list[-1]))
        output = output_gradient
        for x, node in zip(reversed(input_list), reversed(self.nodes_list)):
            output = node.grad(x, output)
        return output

    def update(self, inputs, output_gradient, step_size):
        input_list = [inputs]
        for node in self.nodes_list[0:-1]:
            input_list.append(node(input_list[-1]))
        output = []
        gradient = output_gradient
        for x, node in zip(reversed(input_list), reversed(self.nodes_list)):
            output.append(node.update(x, gradient, step_size))
            gradient = node.grad(x, gradient)
        output.reverse()
        return SequentialNode(output)

    def update_batch(self, input_list, grad_list, step_size):
        assert(len(input_list) == len(grad_list))
        input_list_list = [input_list]
        for node in self.nodes_list[0:-1]:
            il = [node(i) for i in input_list_list[-1]]
            input_list_list.append(il)
        output = []
        gb = grad_list
        for x, node in \
          zip(reversed(input_list_list), reversed(self.nodes_list)):
            output.append(node.update_batch(x, gb, step_size))
            gb = [node.grad(xx, gg) for xx, gg in zip(x, gb)]
        output.reverse()
        return SequentialNode(output)

#GraphNode is a Node that consists of other nodes connected in a generic graph
# structure
class GraphNode(Node):
    _in_label = "in"   #the reserved keyword for labelling the input
    _out_label = "out" #the reserved keyword for labelling the output
    #Constructs a GraphNode that takes inputs in the shape of [input_shape],
    # returns outputs in the shape of [output_shape], contains nodes [nodes],
    # and contains edges as specified in [edges].
    # 
    # [input_shape] is a tuple of integers, where its length is the number of
    # vector inputs this Node takes, and where the [i]th entry corresponds to
    # the dimension of the [i]th input vector.
    # 
    # [output_shape] is a tuple of integers, specifying the number of outputs
    # and the dimensions with the same rules as for [input_shape]
    #
    # [nodes] is a map from strings to Nodes, where, abstractly, the strings
    # represent the names of the nodes, and the Nodes are the components of
    # this computation graph. The strings "in" and "out" may not be used as
    # names, as these strings are reserved
    #
    # [edges] is a map from (string, integer) to (string, integer). If [edges]
    # maps (s1, i1) to (s2, i2), then during the computation, the [i1]th output
    # from node [s1] is passed as the [i2]th input to [s2].
    #
    # Requires:
    #   - Every vector input is routed as an input to some computation node.
    #   Specifically, for 0 <= i < len(input_shape), ("in", i) is a key in
    #   [edges].
    #   - Every output vector is the output of some computation node. 
    #   Specifically, for 0 <= i < len(output_shape), ("out", i) is the value
    #   that some key is mapped to
    #   - No circular dependencies. There may not be any path from a node to
    #   itself along any edges.
    #   - [edges] is bijective. That is, there are not two different tuples [a]
    #   and [b] such that [edges[a] == edges[b]].
    #   - "in" and "out" are not keys to the [nodes] dictionary. These are
    #   reserved keywords specifying inputs and outputs, respectively.
    #   - All node names in [edges] corresponds with a node in [nodes]
    #   - Every computation node has the right number of inputs and every output
    #   is directed into the input of some other node.
    def __init__(self, input_shape, output_shape, nodes, edges):
        super().__init__(input_shape, output_shape)
        # rev_edges is the inverse map of edges
        self._edges = {}
        self._rev_edges = {}
        self._nodes = nodes
        for k in edges:
            v = edges[k]
            self._edges[k] = v
            assert(v not in self._rev_edges)
            self._rev_edges[v] = k
        #check preconditions
        assert(GraphNode._in_label not in nodes)
        assert(GraphNode._out_label not in nodes)
        for i, d in enumerate(input_shape):
            assert((GraphNode._in_label, i) in edges)
            n, j = edges[(GraphNode._in_label, i)]
            assert(nodes[n].input_dims(j) == d)
        for i, d in enumerate(output_shape):
            assert((GraphNode._out_label, i) in self._rev_edges)
            n, j = self._rev_edges[(GraphNode._out_label, i)]
            assert(nodes[n].output_dims(j) == d)
        for n in nodes:
            inputs = nodes[n].num_inputs()
            for i in range(inputs):
                assert((n, i) in self._rev_edges)
                src, j = self._rev_edges[(n, i)]
                if src != GraphNode._in_label:
                    assert(nodes[n].input_dims(i) == nodes[src].output_dims(j))
            outputs = nodes[n].num_outputs()
            for i in range(outputs):
                assert((n, i) in edges)
                src, j = edges[(n, i)]
                if src != GraphNode._out_label:
                    assert(nodes[n].output_dims(i) == nodes[src].input_dims(j))
        #Compute topo-sort order
        self._topo = [GraphNode._out_label]
        stack = [GraphNode._in_label]
        while len(stack) > 0:
            n = stack[-1]
            i = None
            if n == GraphNode._in_label: i = self.num_inputs()
            else: i = nodes[n].num_outputs()
            for j in range(i):
                #check the child, if it's already handled then we're good
                # else if it's on the stack, throw exception
                # else push it to the stack, break out of this loop
                m, k = edges[(n, j)]
                assert(m not in stack)
                if m not in self._topo: 
                    stack.append(m)
                    break
            if stack[-1] == n:
                stack.pop()
                self._topo.append(n)
        self._topo.pop(0)
        self._topo.pop()
        assert(GraphNode._in_label not in self._topo)
        assert(len(self._topo) == len(nodes))
        self._topo.reverse()
        #Set up instance variables for caching computations, for efficiency
        self._input_map_cache = None
        self._gradient_map_cache = None
        self._last_input = None
        self._last_grad = None
    
    #Checks if tuples of numpy arrays x and y are equal to each other in value
    # Requires x and y are the same shape.
    def _check_data_eq(self, x, y):
        return all(all(xx == yy) for xx, yy in zip(x, y))
    #Helper method. For a set of inputs [inputs], returns a map from
    # (string, int) pairs to numpy vectors. For all nodes in this GraphNode,
    # if [n] is the name of that node, and [i] is an index, then running
    # [inputs] through this graph node would cause the [i]th input to node [n]
    # to receive the vector [output[(n, i)]] where [output] is the output of
    # this function
    def _compute_inputs(self, inputs):
        self.check_input_shape(inputs)
        if self._last_input is not None \
            and self._check_data_eq(inputs, self._last_input):
            return self._input_map_cache
        input_map = {}
        for i, v in enumerate(inputs):
            input_map[self._edges[(GraphNode._in_label, i)]] = v
        for n in self._topo:
            node = self._nodes[n]
            n_input = tuple(input_map[(n, i)] for i in range(node.num_inputs()))
            n_output = node(n_input)
            for i, v in enumerate(n_output):
                input_map[self._edges[(n, i)]] = v
        self._last_input = inputs
        self._input_map_cache = input_map
        return input_map

    #Helper method. Returns a map [m] from (string, int) pairs to numpy vectors,
    # such that for all nodes in this GraphNode, if [n] is the name of that
    # node, and [i] is an index, then running back-propagation for input
    # [inputs] and output_gradient [output_gradient] will result in the
    # gradient of the [i]th output of node [n] being [m[(n, i)]].
    def _compute_output_gradients(self, inputs, output_gradient):
        self.check_input_shape(inputs)
        self.check_output_shape(output_gradient)
        if self._last_grad is not None and self._last_input is not None \
            and self._check_data_eq(self._last_input, inputs) \
            and self._check_data_eq(self._last_grad, output_gradient):
            return self._gradient_map_cache
        input_map = self._compute_inputs(inputs)
        self.check_output_shape(output_gradient)
        output_map = {}
        for i, v in enumerate(output_gradient):
            output_map[self._rev_edges[GraphNode._out_label, i]] = v
        for n in reversed(self._topo):
            node = self._nodes[n]
            grad = tuple(output_map[n, i] for i in range(node.num_outputs()))
            n_input = tuple(input_map[n, i] for i in range(node.num_inputs()))
            n_grad = node.grad(n_input, grad)
            for i, v in enumerate(n_grad):
                output_map[self._rev_edges[(n, i)]] = v
        self._last_grad = output_gradient
        self._gradient_map_cache = output_map
        return output_map

    def __call__(self, inputs):
        input_map = self._compute_inputs(inputs)
        return tuple(input_map[(GraphNode._out_label, i)] \
            for i in range(self.num_outputs()))

    def grad(self, inputs, output_gradient):
        output_map = self._compute_output_gradients(inputs, output_gradient)
        return tuple(output_map[(GraphNode._in_label, i)] 
            for i in range(self.num_inputs()))

    def update(self, inputs, output_gradient, step_size):
        input_map = self._compute_inputs(inputs)
        output_map = self._compute_output_gradients(inputs, output_gradient)
        new_nodes = {}
        for n in self._nodes:
            node = self._nodes[n]
            n_input = tuple(input_map[n, i] for i in range(node.num_inputs()))
            n_grad = tuple(output_map[n, i] for i in range(node.num_outputs()))
            new_nodes[n] = node.update(n_input, n_grad, step_size)
        return GraphNode(self._inputs, self._outputs, new_nodes, self._edges)

    def update_batch(self, input_list, grad_list, step_size):
        assert(len(input_list) == len(grad_list))
        input_map_list = []
        output_map_list = []
        for inputs, grad in zip(input_list, grad_list):
            input_map_list.append(self._compute_inputs(inputs))
            output_map_list.append(self._compute_output_gradients(inputs, grad))
        new_nodes = {}
        for n in self._nodes:
            node = self._nodes[n]
            n_inputs = tuple( \
                tuple(input_map[n, i] for i in range(node.num_inputs())) \
                for input_map in input_map_list)
            n_grad = tuple( \
                tuple(output_map[n, i] for i in range(node.num_outputs())) \
                for output_map in output_map_list)
            new_nodes[n] = node.update_batch(n_inputs, n_grad, step_size)
        return GraphNode(self._inputs, self._outputs, new_nodes, self._edges)
