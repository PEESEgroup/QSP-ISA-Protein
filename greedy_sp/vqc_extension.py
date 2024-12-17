from qc import Gate, apply_gate
import cmath

def g_plus(gate):
    return Gate(gate.gate_type, gate.target, gate.angle + cmath.pi / 2, gate.n_qubits)

def g_minus(gate):
    return Gate(gate.gate_type, gate.target, gate.angle - cmath.pi / 2, gate.n_qubits)

def apply_delta(gate, delta):
    if gate.is_cx(): return gate
    return Gate(gate.gate_type, gate.target, gate.angle + delta, gate.n_qubits)

def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2

def compute_gradient(q_start, q_end, gates):
    start_cache = []
    end_cache = [None for _ in range(len(gates))]
    start_cache.append(q_start)
    for i in range(len(gates) - 1):
        start_cache.append(apply_gate(gates[i], start_cache[i]))
    end_cache[-1] = q_end
    for i in range(len(gates) - 2, -1, -1):
        end_cache[i] = apply_gate(gates[i + 1].inverse(), end_cache[i + 1])
    output = []
    for i in range(len(gates)):
        base = gates[i]
        if base.is_cx():
            output.append(0)
            continue
        init = start_cache[i]
        final = end_cache[i]
        plus = apply_gate(g_plus(base), init)
        minus = apply_gate(g_minus(base), final)
        cost_plus = fidelity(plus, end)
        cost_minus = fideltiy(minus, end)
        output[i] = cost_plus - cost_minus
    return output

def flatten_list(l):
    output = []
    for sublist in l: output.extend(sublist)
    return output

def unflatten_list(flat, shape):
    output = []
    i = 0
    for sublist in shape:
        output.append(flat[i:i + len(sublist)])
        i += len(sublist)
    return output

#Performs gradient descent on the block indices specified in [blocks], or all
#the blocks if blocks is None
def gradient_descent(self, step_size, step_count, blocks=None):
    if blocks is None: blocks = [i for i in range(len(self.gates))]
    flat_gates = flatten_list(self.gates)
    multiplier = []
    for i, sublist in enumerate(self.gates):
        if i in blocks: multiplier.append([1] * len(sublist))
        else: multiplier.append([0] * len(sublist))
    pass
