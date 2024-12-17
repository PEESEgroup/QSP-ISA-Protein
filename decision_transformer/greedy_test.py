from greedy2 import StateTracker, prepare_state
from manual import rand_complex_state
from math import sqrt

count = 100
qubit_counts = [9]
cx_counts = []
for n_qubits in qubit_counts:
    total_cx = []
    for i in range(count):
        print(i)
        target_state = rand_complex_state(n_qubits)
        tracker = prepare_state(target_state)
        total_cx.append(tracker.cx_count())
    cx_counts.append(total_cx)

with open("temp.txt", "w") as f:
    f.write(str(cx_counts))

for l, c in zip(qubit_counts, cx_counts):
    mean = sum(c) / len(c)
    stdev = sqrt(sum((cc- mean) ** 2 for cc in c) / len(c))
    print("n qubits =", l)
    print("  mean:", mean)
    print("  stdev:", stdev)
    print("  min:", min(c))
    print("  max:", max(c))
