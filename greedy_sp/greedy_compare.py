from qc import rand_state, apply_gate
from greedy3 import greedy3_prepare
from greedy import prepare_state
import random

n_qubits = 7
count = 100

random.seed(42)
data_greedy = [1]
data_greedy3 = []

for i in range(count):
    print(i)
    state = rand_state(n_qubits)
    gates3 = greedy3_prepare(state)
    data_greedy3.append(sum(int(g.is_cx()) for g in gates3))
    #tracker = prepare_state(state)
    #data_greedy.append(tracker.cx_count())
    #state3 = state
    #for g in gates3: state3 = apply_gate(g, state3)
    #assert(abs(state3[0]) ** 2 > 0.95)

#mean = sum(data_greedy) / count
mean3 = sum(data_greedy3) / count
#stdev = (sum((d - mean) ** 2 for d in data_greedy) / count) ** 0.5
stdev3 = (sum((d - mean3) ** 2 for d in data_greedy3) / count) ** 0.5
#outliers = sum(int((d - mean) / stdev >= 3) for d in data_greedy)
outliers3 = sum(int((d - mean3) / stdev3 >= 3) for d in data_greedy3)
#print("Greedy, original: ")
#print("    mean:", mean)
#print("    stdev:", stdev)
#print("    outliers:", outliers)
#print()
print("Greedy, v3:")
print("    mean:", mean3)
print("    stdev:", stdev3)
print("    outliers:", outliers3)
