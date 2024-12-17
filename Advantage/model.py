
from enum import Enum, auto
import qiskit as qk

#List of supported VQC gates
class Gates(Enum):
    X = auto()
    Y = auto()
    Z = auto()
    H = auto()
    RX = auto()
    RY = auto()
    RZ = auto()
    CX = auto()
    CY = auto()
    CZ = auto()

#Variational quantum circuit module. Supported functions are
#  construction of a VQC
#  simulaton of the VQC
#  getting the parameter set
class VQC:
    _circuit = None
    _encode = None
    _simulator = qk.Aer.get_backend('aer_simulator')
    _n = None
    _params = []
    #Constructor
    # @param [n] is an integer representing the number of qubits in this
    #   circuit
    # @param [encode] is a function mapping list-like objects of length [n] to
    #   qiskit circuits that encode those vectors
    # @param [gates] is a list of [Gates] to be applied to the circuit after
    #   the initial encoding step
    # @param [measure] is a function mapping bitstrings of length [n] to
    #   real number values. When the circuit is simulated, the weighted average
    #   of measure(outcome) over all the measurement outcomes will be reported
    #   as the result
    def __init__(self, n, encode, gates):
        self._n = n
        self._encode = encode
        self._circuit = qk.QuantumCircuit(n)
        for i, g in enumerate(gates):
            if g == Gates.X: self._circuit.x(range(n))
            elif g == Gates.Y: self._circuit.y(range(n))
            elif g == Gates.Z: self._circuit.z(range(n))
            elif g == Gates.H: self._circuit.h(range(n))
            elif g == Gates.RX:
                for j in range(n):
                    a = qk.circuit.Parameter(str(i) + ":" + str(j))
                    self._circuit.rx(a, j)
                    self._params.append(a)
            elif g == Gates.RY:
                for j in range(n):
                    a = qk.circuit.Parameter(str(i) + ":" + str(j))
                    self._circuit.ry(a, j)
                    self._params.append(a)
            elif g == Gates.RZ:
                for j in range(n):
                    a = qk.circuit.Parameter(str(i) + ":" + str(j))
                    self._circuit.rz(a, j)
                    self._params.append(a)
            elif g == Gates.CX:
                for j in range(n - 1):
                    self._circuit.cx(j, j + 1)
                self._circuit.cx(n - 1, 0)
            elif g == Gates.CY:
                for j in range(n - 1):
                    self._circuit.cy(j, j + 1)
                self._circuit.cy(n - 1, 0)
            elif g == Gates.CZ:
                for j in range(n - 1):
                    self._circuit.cz(j, j + 1)
                self._circuit.cz(n - 1, 0)
            else: raise "Unimplemented"
    #Runs the variational quantum circuit on the input vector [v], using
    # parameter values [p]. Returns the measurement result under [measure]
    # @param [v] is a list-like object with [n] entries
    # @param [p] is a dictionary mapping parameters to values
    # @param [measure] is a function mapping length [n] bitstrings to
    #   real numbers
    def run(self, v, p, measure):
        c1 = self._encode(v)
        c1.append(self._circuit.to_instruction(), 
            [x for x in range(c1.num_qubits)])
        c1.measure_all()
        print(c1.draw())
        c1.assign_parameters(p, inplace=True)
        print(c1.draw())
        circ = qk.transpile(c1, self._simulator)
        rr = self._simulator.run(circ)
        result = rr.result().get_counts(circ)
        output = 0
        print(result)
        for key in result:
            output += measure(key) * result[key]
        output /= 1024.0
        return output
        
    #Returns a list of parameters for this circuit
    def params(self):
        return tuple(self._params)
