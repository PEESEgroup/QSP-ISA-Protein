
#Module for quantum neural network functionality

from node import Node
import qiskit as qk
from qiskit.circuit import Parameter
import math
import random
import numpy as np

class Gate():
    #List of supported gates
    H, X, Y, Z, RX, RY, RZ, CX, CY, CZ, CRX, CRY, CRZ = [i for i in range(13)]
    #Constructor for a quantum circuit gate of type [gate_type], acting on
    # the qubit of index [index1], or indices [index1] and [index2] if it's a
    # 2-qubit gate, and may be parameterized by [param], where [param] is a 
    # qiskit Parameter
    # [gate_type] is one of Gate.X, Gate.Y, ..., Gate.CZ
    def __init__(self, gate_type, index1, index2=None, param=None):
        self.type = gate_type
        self.index1 = index1
        if gate_type >= Gate.CX and gate_type <= Gate.CRZ:
            assert(index2 is not None)
        self.index2 = index2
        if (gate_type >= Gate.RX and gate_type <= Gate.RZ) or \
            (gate_type >= CRX and gate_type <= CRZ):
            assert(param is not None)
        self.param = param

class QNNNode(Node):
    _simulator = qk.Aer.get_backend("statevector_simulator")

    #Constructs a parameterized quantum circuit with [num_qubits] qubits, 
    # parameters [input_params] and trainable parameters [trainable_params], 
    # and gates [gates]. [input_params] is a list of qiskit Parameters that 
    # correspond to the classical input this Node takes; [trainable_params] is 
    # a list of qiskit Parameters used in [gates]. [gates] is a list of 
    # possibly parameterized Gates.
    #
    # Parameters are initialized using uniform random over [0, 2pi)
    def __init__(self, num_qubits, input_params, trainable_params, gates):
        super().__init__((len(input_params),), (num_qubits,))
        circuit = qk.QuantumCircuit(num_qubits)
        self._temp_counter = 0
        def fresh_temp():
            self._temp_counter += 1
            return 
        for gate in gates:
            if gate.type == Gate.H: circuit.h(gate.index1)
            elif gate.type == Gate.X: circuit.x(gate.index1)
            elif gate.type == Gate.Y: circuit.y(gate.index1)
            elif gate.type == Gate.Z: circuit.z(gate.index1)
            elif gate.type == Gate.RX: circuit.rx(gate.param, gate.index1)
            elif gate.type == Gate.RY: circuit.ry(gate.param, gate.index1)
            elif gate.type == Gate.RZ: circuit.rz(gate.param, gate.index1)
            elif gate.type == Gate.CX: circuit.cx(gate.index1, gate.index2)
            elif gate.type == Gate.CY: circuit.cy(gate.index1, gate.index2)
            elif gate.type == Gate.CZ: circuit.cz(gate.index1, gate.index2)
            elif gate.type == Gate.CRX: 
                circuit.crx(gate.param, gate.index1, gate.index2)
            elif gate.type == Gate.CRY: 
                circuit.cry(gate.param, gate.index1, gate.index2)
            elif gate.type == Gate.CRZ: 
                circuit.crz(gate.param, gate.index1, gate.index2)
            else: raise RuntimeError("Incomplete case match")
        self._circuit = circuit
        self._input_params = input_params
        self._trainable_params = trainable_params
        self._trainable_values = np.array([random.uniform(0, 2 * math.pi) \
            for _ in range(len(trainable_params))])
        self._vectors = [np.array([(i >> j) % 2 for j in range(num_qubits)]) \
            for i in range(2 ** num_qubits)]
    def _run(self, params):
        bc = self._circuit.bind_parameters(params)
        circ = qk.transpile(bc, QNNNode._simulator)
        result = QNNNode._simulator.run(circ).result().get_statevector(circ)
        return sum(r * v for r, v in zip(result.probabilities(), self._vectors))
    def _compute_params(self, inputs):
        self.check_input_shape(inputs)
        param_values = {k:v \
            for k, v in zip(self._trainable_params, self._trainable_values)}
        for k, v in zip(self._input_params, inputs[0]): param_values[k] = v
        return param_values
    def __call__(self, inputs):
        return (self._run(self._compute_params(inputs)),)

    def grad(self, inputs, output_gradient):
        self.check_output_shape(output_gradient)
        base = self._compute_params(inputs)
        output = []
        for k in self._input_params:
            base[k] += math.pi / 2
            a = self._run(base)
            base[k] -= math.pi
            b = self._run(base)
            output.append((a - b) / 2)
            base[k] += math.pi / 2
        return (output,)

    def update(self, inputs, output_gradient, step_size):
        self.check_output_shape(output_gradient)
        base = self._compute_params(inputs)
        updates = []
        for k in self._trainable_params:
            base[k] += math.pi / 2
            a = self._run(base)
            base[k] -= math.pi
            b = self._run(base)
            updates.append((a - b).dot(output_gradient[0]) / 2)
            base[k] += math.pi / 2
        output = QNNNode(self.output_dims(0), self._input_params, 
            self._trainable_params, [])
        output._circuit = self._circuit
        output._trainable_values = self._trainable_values + updates
        return output

    def update_batch(self, inputs, output_gradient, step_size):
        #Not expecting this to be needed for now
        raise RuntimeError("Unimplemented")

#QNN sanity check. Should have the second printed vector be much closer to
# [0, 0] than the first.
#a = Parameter("a")
#b = Parameter("b")
#t1 = Parameter("c")
#t2 = Parameter("d")
#qnn = QNNNode(2, [a, b], [t1, t2], \
#    [Gate(Gate.RX, 0, param=a), \
#    Gate(Gate.RX, 1, param=b), \
#    Gate(Gate.RX, 0, param=t1), \
#    Gate(Gate.RX, 1, param=t2)])
#x = (np.array([1, 0]),)
#(res,) = qnn(x)
#grad = (-res,)
#for _ in range(10): qnn = qnn.update(x, grad, 0.1)
#print(res)
#print(qnn(x)[0])
