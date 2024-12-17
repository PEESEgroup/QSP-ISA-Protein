
import datagen
from qk_kernel import QKSVM
import qiskit as qk

#Assumes normalized input [v]
def encode(v):
    n = len(v)
    circuit = qk.QuantumCircuit(n)
    for i in range(n):
        circuit.ry(v[i], i)
    return circuit

train_x = []
train_y = []
test_x = []
test_y = []

t1 = 0
t2 = 3

count = 10
#for i in range(200, datagen.count(True, t1)):
for i in range(200, 200 + count):
    train_x.append(datagen.normalize(datagen.sample(True, t1, i)))
    train_y.append(1)

#for i in range(200, datagen.count(True, t2)):
for i in range(200, 200 + count):
    train_x.append(datagen.normalize(datagen.sample(True, t2, i)))
    train_y.append(-1)

#for i in range(200, datagen.count(False, t1)):
for i in range(200, 200 + count):
    test_x.append(datagen.normalize(datagen.sample(False, t1, i)))
    test_y.append(1)

#for i in range(200, datagen.count(False, t2)):
for i in range(200, 200 + count):
    test_x.append(datagen.normalize(datagen.sample(False, t2, i)))
    test_y.append(-1)

print("Initialized data")

model = QKSVM(encode)

model.fit(train_x, train_y)

total = 0
correct = 0
for x, y in zip(test_x, test_y):
    if(model.predict(x) == y): correct += 1
    total += 1

print("Total test samples: ", total)
print("Total correct: ", correct)

print("Completed fitting")
