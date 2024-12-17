import numpy as np
import random
import qiskit as qk
import mottonen
import nncx_efficient

random.seed(42)

def qubit_reverse(state, n):
    def bit_reverse(k):
        l = k
        bits = []
        for _ in range(n):
            bits.append(l % 2)
            l >>= 1
        bits.reverse()
        return sum(b * (1 << i) for i, b in enumerate(bits))
    return np.array([state[bit_reverse(i)] for i in range(len(state))])

simulator = qk.Aer.get_backend("statevector_simulator")
basis_gates = ['rx', 'ry', 'rz', 'cx']
def test_circuit(circuit, target_state, n):
    coupling_map = []
    initial_layout = [i for i in range(n)]
    for i in range(n - 1):
        coupling_map.append([i, i + 1])
        coupling_map.append([i + 1, i])
    compiled = qk.transpile(circuit, simulator, coupling_map=coupling_map, \
      initial_layout=initial_layout, layout_method="trivial", \
      basis_gates=basis_gates, optimization_level=1)
    optimized = qk.transpile(compiled, simulator, coupling_map=coupling_map, \
      initial_layout=initial_layout, layout_method="trivial", \
      basis_gates=basis_gates, optimization_level=3)
    sv = simulator.run(circuit).result().get_statevector()
    fidelity = abs(np.array(sv).dot(target_state)) ** 2
    cx_simple = compiled.count_ops()['cx']
    cx_opt = optimized.count_ops()['cx']
    depth_simple = compiled.depth()
    depth_opt = optimized.depth()
    return {
      "fidelity": fidelity,
      "cx_simple": cx_simple,
      "cx_optimized": cx_opt,
      "depth_simple": depth_simple,
      "depth_optimized": depth_opt,
    }
def test(n, count=100):
    r1l = []
    r2l = []
    for _ in range(100):
        state = np.array([random.gauss(0, 1) for _ in range(1 << n)])
        qr = qubit_reverse(state, n)
        norm = state.dot(state) ** 0.5
        state /= norm
        c1 = mottonen.state_prep(qr, n)
        c2 = nncx_efficient.state_prep(qr, n)
        r1 = test_circuit(c1, state, n)
        r2 = test_circuit(c2, state, n)
        r1l.append(r1)
        r2l.append(r2)
    af1 = sum(r["fidelity"] for r in r1l) / count
    af2 = sum(r["fidelity"] for r in r2l) / count
    acs1 = sum(r["cx_simple"] for r in r1l) / count
    acs2 = sum(r["cx_simple"] for r in r2l) / count
    aco1 = sum(r["cx_optimized"] for r in r1l) / count
    aco2 = sum(r["cx_optimized"] for r in r2l) / count
    ads1 = sum(r["depth_simple"] for r in r1l) / count
    ads2 = sum(r["depth_simple"] for r in r2l) / count
    ado1 = sum(r["depth_optimized"] for r in r1l) / count
    ado2 = sum(r["depth_optimized"] for r in r2l) / count
    return (af1, af2, acs1, acs2, aco1, aco2, ads1, ads2, ado1, ado2)

stats = []
n = [3, 5, 7, 9]
for nn in n:
    print("testing n = ", nn)
    stats.append(test(nn))

with open("results.csv", "w") as f:
    f.write("Qubit number,Average Fidelity (MOT),Average Fidelity (CXE),CX Count before optimization (MOT),CX Count before optimization (CXE),CX Count after optimization (MOT),CX Count after optimization(CXE),Circuit Depth before optimization (MOT),Circuit Depth before optimization (CXE),Circuit Depth after optimization (MOT),Circuit Depth after optimization (CXE)\n");
    for nn, ss in zip(n, stats):
        f.write(str(nn))
        for s in ss:
            f.write(",")
            f.write(str(s))
        f.write("\n")


