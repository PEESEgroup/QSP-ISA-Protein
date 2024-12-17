
import model
import qiskit as qk

def enc(v):
    output = qk.QuantumCircuit(3)
    output.rx(v[0], 0)
    output.rx(v[1], 1)
    output.rx(v[2], 2)
    return output

def measure(z):
    if z[2] == '0': return 0
    return 1

m = model.VQC(3, enc, [model.Gates.RX], measure)
ls = m.params()
d = {}
for x in ls: d[x] = 1
print(m.run([0, 0, 0], d))
