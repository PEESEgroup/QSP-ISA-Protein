
import cirq
import sympy
import random
import math
import time

_debug = True

#Implementation note: PQC models are 2-tuples (c, q, p, v) where [c] is the 
# cirq circuit, [q] is a list of input qubits, [p] is a list of output qubits,
# and [v] is a list of parameters. [v] does not include the input parameters.
# [v] is a list of parameter lists, with each parameter list corresponding to
# a different layer. 

_input_prefix = "INPUT"
#Returns a fresh model with [i] input qubits and [j] output qubits. Input qubits
# are allowed to take real number inputs ranging betwen -pi/2 and pi/2, while 
# the output qubits return real numbers between -1 and 1.
def new_model(i, j):
    m = cirq.Circuit()
    q = cirq.GridQubit.rect(rows=1, cols=i, top=1, left=0)
    p = cirq.GridQubit.rect(rows=1, cols=j, top=0, left=0)
    assert len(set(q) & set(p)) == 0
    v = []
    for i, b in enumerate(q):
        s = sympy.Symbol(_input_prefix + str(i))
        m.append(cirq.rx(s).on(b))
    return (m, q, p, v)

def param_set(model):
    output = set()
    for l in model[3]:
        output.update(l)
    return output

def get_circuit(model):
    return model[0]

def get_inputs(model):
    return model[1]

def get_outputs(model):
    return model[2]

def layers(model):
    return len(model[3])

def layer_params(model, i):
    return model[3][i]

X = "x"
Y = "y"
Z = "z"
RX = "rx"
RY = "ry"
RZ = "rz"
ORX = "orx"
CX = "cx"
CY = "cy"
CZ = "cz"
CRX = "crx"
CRY = "cry"
CRZ = "crz"
#S_CX is the gate for control-X'ing the output bits where every input bit is a
# control.
S_CX = "scx"

def gateop(g, bit):
    if g == X or g == CX or g == S_CX or g == ORX:
        return cirq.X(bit)
    if g == Y or g == CY:
        return cirq.Y(bit)
    if g == Z or g == CZ:
        return cirq.Z(bit)
    if g == RX or g == CRX:
        return cirq.rx(1).on(bit)
    if g == RY or g == CRY:
        return cirq.ry(1).on(bit)
    if g == RZ or g == CRZ:
        return cirq.rz(1).on(bit)
    print("Incomplete pattern matching: no match for input " + g)
    assert False

one_qubit_gates = (X, Y, Z, RX, RY, RZ)
output_gates = (ORX,)
ctrl_one_qubit_gates = (CX, CY, CZ, CRX, CRY, CRZ)
s_ctrl_one_qubit_gates = (S_CX,)
#Adds a layer of gates. [model] is the model to add a layer of gates to; [gate]
# is the string representation of the gate to add, and prefix is the string
# prefix of the parameter name. Prefixes may be not be reused and should not
# contain uppercase characters to avoid unintentional name conflicts.
# If prefix is None (as the default), the layer added will not be parameterized.
def add_layer(model, gate, prefix=None, scalar=1):
    (m, q, p, v) = model
    if gate in one_qubit_gates:
        l = []
        #1-qubit gates are applied to everybody
        for i, b in enumerate(q):
            g = gateop(gate, b)
            if prefix == None:
                m.append(g)
            else:
                s = sympy.Symbol(prefix + "Q" + str(i))
                m.append(g**(scalar * s))
                l.append(s)
        for i, b in enumerate(p):
            g = gateop(gate, b)
            if prefix == None:
                m.append(g)
            else:
                s = sympy.Symbol(prefix + "P" + str(i))
                m.append(g**(scalar * s))
                l.append(s)
        if len(l) > 0: v.append(tuple(l))
        return
    if gate in ctrl_one_qubit_gates:
        #Ctrl-1 qubit gates form a cyclic chain of ctrl-actions
        l = []
        bits = list(q)
        bits.extend(p)
        cg = None
        ag = bits[-1]
        for i, b in enumerate(bits):
            cg = ag
            ag = b
            g = gateop(gate, ag)
            if prefix == None:
                m.append(g.controlled_by(cg))
            else:
                s = sympy.Symbol(prefix + "B" + str(i))
                m.append(g.controlled_by(cg)**s)
                l.append(s)
        if len(l) > 0: v.append(tuple(l))
        return
    if gate in s_ctrl_one_qubit_gates:
        #Use every input bit as a control
        for j, c in enumerate(p):
            g = gateop(gate, c)
            for b in q:
                g = g.controlled_by(b)
            m.append(g)
        return
    if gate in output_gates:
        #One qubit manipulations on the output bits
        l = []
        for j, c in enumerate(p):
            g = gateop(gate, c)
            if prefix == None: m.append(g)
            else:
                s = sympy.Symbol(prefix + "P" + str(j))
                m.append(g**s)
                l.append(s)
        if len(l) > 0: v.append(tuple(l))
        return
    print("Incomplete case match for gate: " + gate)
    assert False

#Simulates the model [model] on inputs [i] using parameters [p]. Returns the 
# expectations of Z on the output qubits. [i] is a tuple and [params] is a
# dictionary mapping sympy symbols to values. Since the parameter naming
# convention is not well specified, it is recommended that [params] be
# constructed using the other functions given here.
def compute(model, i, params):
    (m, q, p, _) = model
    ps = {}
    for k in range(len(q)):
        ps[sympy.Symbol(_input_prefix + str(k))] = i[k]
    for s in param_set(model):
        ps[s] = params.get(s, 0)
    simulator = cirq.Simulator()
    param_res = cirq.ParamResolver(ps)
    output = simulator.simulate(m, param_res)
    zs = (cirq.Z(pp) for pp in p)
    state = output.final_state_vector
    qm = output.qubit_map
    exzs = tuple(z.expectation_from_state_vector(state, qm).real for z in zs)
    return exzs


#Given a model [model], model inputs [i], runs the circuit once for each
# parameter set [pl], and returns the list of outputs.
# [pl] is a list of parameter sets; the list of outputs will be in the order
# corresponding to the ordering in [pl].
def compute_sweep(model, i, pl):
    (m, q, p, _) = model
    s = cirq.Simulator()
    def to_resolver_form(p):
        output = {}
        for k in range(len(i)):
            output[sympy.Symbol(_input_prefix + str(k))] = i[k]
        for s in param_set(model):
            output[s] = p.get(s, value=0)
        return cirq.ParamResolver(output)
    params = map(to_resolver_form, pl)
    observables = list(cirq.Z(pp) for pp in p)
    return s.simulate_expectation_values_sweep(
        program=m, 
        observables=observables, params=cirq.study.ListSweep(params))

#Returns a random set of parameters for the model [model]. Samples each
# parameter from a uniform distribution between [-1, 1].
def rand_params(model):
    output = {}
    for s in param_set(model): output[s] = math.pi * (random.random() * 2 - 1)
    return output

#Returns a new parameter set, with every parameter within [step] of its
# corresponding value in params. If [discrete], then the new parameters will be
# exactly [step] away from the value in [params] (could be smaller or larger),
# otherwise, the difference is sampled from a uniform random distribution
# [-step, step]
def child(params, step, discrete=False):
    output = {}
    if discrete:
        for s in params:
            output[s] = params[s] + (2 * random.randint(0, 1) - 1) * step
            output[s] %= 2 * math.pi
    else:
        for s in params:
            output[s] = params[s] + (2 * random.random() - 1) * step
            output[s] %= 2 * math.pi
    return output

#Computes the gradient of the output with respect to the parameter [param].
# The computation method depends on the type. If [type] is 0, then the gradient
# is computed as model[param + epsilon/2] - model[param + epsilon/2] / epsilon.
# if the type is larger than 0, then the gradient is computed as
#   (model[param + epsilon] - model[param]) / epsilon
# If the type is less than 0, then the gradient is computed as
#   (model[param] - model[param - epsilon]) / epsilon
def grad_int(model, param, params, inputs, type=0, epsilon=1e-4):
    assert param in params
    mod = params.copy()
    if type == 0:
        mod[param] += epsilon / 2
        a = compute(model, inputs, mod)
        mod[param] -= epsilon
        b = compute(model, inputs, mod)
        return tuple((aa - bb) / epsilon for aa, bb in zip(a, b))
    if type > 0:
        b = compute(model, inputs, mod)
        mod[param] += epsilon
        a = compute(model, inputs, mod)
        return tuple((aa - bb) / epsilon for aa, bb in zip(a, b))
    else:
        assert type < 0
        a = compute(model, inputs, mod)
        mod[param] -= epsilon
        b = compute(model, inputs, mod)
        return tuple((aa - bb) / epsilon for aa, bb in zip(a, b))

#Returns the gradient of the circuit output with respect to the [index]th input.
# The gradient is evaluated at the point with inputs [inputs] and parameters
# [params]. The gradient is computed in an analogous way as grad_int, depending
# on [type]
def grad_ext(model, index, inputs, params, type=0, epsilon=1e-4):
    assert 0 <= index < len(inputs)
    mod = list(inputs)
    if type == 0:
        mod[index] += epsilon / 2
        a = compute(model, mod, params)
        mod[index] -= epsilon
        b = compute(model, mod, params)
        return tuple((aa - bb) / epsilon for aa, bb in zip(a, b))
    if type > 0:
        b = compute(model, mod, params)
        mod[index] += epsilon
        a = compute(model, mod, params)
        return tuple((aa - bb) / epsilon for aa, bb in zip(a, b))
    else:
        assert type < 0
        a = compute(model, mod, params)
        mod[index] -= epsilon
        b = compute(model, mod, params)
        return tuple((aa - bb) / epsilon for aa, bb in zip(a, b))

#Saves parameter set d into file path [f]
def save(d, f):
    with open(f, "w") as ff:
        for k in d:
            ff.write(str(k) + ": " + str(d[k]) + "\n")

#Loads the parameter set in file path [f]
def load(f):
    output = {}
    with open(f, "r") as ff:
        line = ff.readline()
        while len(line) > 0 and not line.isspace():
            k, v = line.split(":")
            k = sympy.Symbol(k.strip())
            v = float(v)
            output[k] = v
            line = ff.readline()
    return output

