
import qk_kernel
import qiskit as qk

def encode(v):
    n = len(v)
    circuit = qk.QuantumCircuit(n)
    for i in range(n):
        circuit.ry(v[i], i)
    return circuit

model = qk_kernel.QKSVM(encode)

x = [[0, 0], [0, 1]]
y = [1, -1]

model.fit(x, y)

print(model._kernel_matrix_(x))
print(model.predict([0.5, 0.4]))
print(model.predict([0.5, 0.6]))


