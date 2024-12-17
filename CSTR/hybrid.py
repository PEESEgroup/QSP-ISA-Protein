
import tensorflow.keras as tfk
import tensorflow_quantum as tfq
import cirq
import sympy
import proto_model as pm
import numpy as np

model = pm.new_model(10, 5)

pm.add_layer(model, pm.RX, "x1")
pm.add_layer(model, pm.RY, "y1")
pm.add_layer(model, pm.CRX)
pm.add_layer(model, pm.RX, "x2")
pm.add_layer(model, pm.RY, "y2")
pm.add_layer(model, pm.S_CX)
pm.add_layer(model, pm.RX, "x3")
pm.add_layer(model, pm.RY, "y3")

circuit = pm.get_circuit(model)
operators = (cirq.Z(p) for p in pm.get_outputs(model))

class InputQuantum(tfk.layers.Layer):
    #Requires a list of input qubits.
    def __init__(self, input_list):
        self.num_inputs = len(input_list)
        self.inputs = tuple(input_list)
    def call(self, inputs):
        print("Called shape: ", inputs.shape)
        assert False

tfm = tfk.Sequential([
    tfk.layers.Dense(14),
    tfk.layers.LeakyReLU(),
    tfk.layers.Dense(10, activation="sigmoid"),
    InputQuantum(pm.get_inputs(model)),
    tfq.layers.PQC(circuit, operators),
    tfk.layers.Dense(10),
    tfk.layers.LeakyReLU(),
    tfk.layers.Dense(4, activation="softmax")
])

tfm(np.array([1, 0, 0, 0, 0, 1, 1]))
