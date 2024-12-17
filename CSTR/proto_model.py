
import cirq
import sympy
import random
import math

_debug = True
#Prototype code for making PQC models. The current implementation expects this
# to be used mainly with a genetic algorithm.

#Implementation note: PQC models are 2-tuples (c, q, p, v) where [c] is the 
# cirq circuit, [q] is a list of input qubits, [p] is a list of output qubits,
# and [v] is a list of parameters. [v] does not include the input parameters.

_input_prefix = "INPUT"
#Returns a fresh model with [i] input qubits and [j] output qubits. Input qubits
# are allowed to take real number inputs ranging betwen -pi/2 and pi/2, while 
# the output qubits return real numbers between -1 and 1.
def new_model(i, j):
    m = cirq.Circuit()
    q = cirq.LineQubit.range(1, i + 1)
    p = cirq.LineQubit.range(-j, 0)
    v = set()
    for i, b in enumerate(q):
        s = sympy.Symbol(_input_prefix + str(i))
        m.append(cirq.rx(s).on(b))
    return (m, q, p, v)

X = "x"
Y = "y"
Z = "z"
RX = "rx"
RY = "ry"
RZ = "rz"
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
    if g == X or g == CX or g == S_CX:
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
ctrl_one_qubit_gates = (CX, CY, CZ, CRX, CRY, CRZ)
s_ctrl_one_qubit_gates = (S_CX,)
#Adds a layer of gates. [model] is the model to add a layer of gates to; [gate]
# is the string representation of the gate to add, and prefix is the string
# prefix of the parameter name. Prefixes may not be reused, also, they should 
# not contain uppercase characters to avoid name conflicts
def add_layer(model, gate, prefix, full=False):
    (m, q, p, v) = model
    if gate in one_qubit_gates:
        #1-qubit gates are applied to everybody
        for i, b in enumerate(q):
            s = sympy.Symbol(prefix + "Q" + str(i))
            g = gateop(gate, b)
            m.append(g**s)
            assert s not in v
            v.add(s)
        for i, b in enumerate(p):
            s = sympy.Symbol(prefix + "P" + str(i))
            g = gateop(gate, b)
            m.append(g**s)
            assert s not in v
            v.add(s)
    elif gate in ctrl_one_qubit_gates:
        #Ctrl-1 qubit gates act on output bits, controlled by input bits
        for i, b in enumerate(q):
            for j, c in enumerate(p):
                s = sympy.Symbol(prefix + "QP" + str(i) + str(j))
                g = gateop(gate, c)
                m.append(g.controlled_by(b))
                assert s not in v
                v.add(s)
    elif gate in s_ctrl_one_qubit_gates:
        #Use every input bit as a control
        for j, c in enumerate(p):
            s = sympy.Symbol(prefix + "SP" + str(j))
            g = gateop(gate, c)
            for b in q:
                g = g.controlled_by(b)
            m.append(g)
            assert s not in v
            v.add(s)
    else:
        print("Incomplete case match for gate: " + gate)
        assert False

#Simulates the model [model] on inputs [i] using parameters [p]. Returns the 
# expectations of Z on the output qubits. [i] is a tuple and [params] is a
# dictionary mapping sympy symbols to values. Since the parameter naming
# convention is not well specified, it is recommended that [params] be
# constructed using the other functions given here.
def compute(model, i, params):
    (m, q, p, v) = model
    for s in v: assert s in params
    ps = {}
    for k in range(len(q)):
        ps[sympy.Symbol(_input_prefix + str(k))] = i[k]
    ps.update(params)
    simulator = cirq.Simulator()
    param_res = cirq.ParamResolver(ps)
    output = simulator.simulate(m, param_res)
    zs = (cirq.Z(pp) for pp in p)
    state = output.final_state_vector
    qm = output.qubit_map
    exzs = tuple(z.expectation_from_state_vector(state, qm).real for z in zs)
    return exzs

#Computes the hinge error of model [model] with parameters [params] on input
# [i], expecting output [e]
def sample_hinge_err(model, i, params, e):
    a = compute(model, i, params)
    return sum(abs(e[i] - z) for i, z in enumerate(a))

#Returns a random set of parameters for the model [model]. Samples each
# parameter from a uniform distribution between [-1, 1].
def rand_params(model):
    (_, _, _, v) = model
    output = {}
    for s in v: output[s] = random.random() * 2 - 1
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

#Returns a new list of parameter sets, generated by taking each parameter set
# in pl and making [felicity] number of children from them, children step size
# [step]. The parameter sets in [pl] are not included in the output.
def reproduce(pl, felicity, step):
    output = []
    for p in pl:
        output.extend((child(p, step) for _ in range(felicity)))
    return output

import datagen
#Performs genetic algorithm for model [model] using initial list of candidate
# parameter values [init], data being generated from [generator] (from datagen
# module). The error function [erf] given is used to determine the surviving
# candidates in each round. [erf] takes arguments (model, param, batch) and
# returns the error. Batches are taken from generator until there's no more data
# in the generator left. Returns the surviving candidates at the end of training
# At each iteration, each candidate produces [children] offspring, and then all
# the candidates are narrowed down to [survivors] candidates. If [mono], then
# the parents are considered candidates alongside their children, otherwise, the
# parents are removed from the candidates pool after they produce offspring.
def genetic_train(model, init, generator, erf, step,
    children=8, survivors=4, batch_size=1, mono=False):
    while datagen.has_next(generator):
        if _debug: print("Next batch!")
        batch = datagen.next_batch(generator, batch_size)
        if _debug:
            print("Batch size: ", len(batch))
            print("Normal samples count: ", sum(int(t == 0) for (_, t) in batch))
            print("Faulty samples count: ", sum(int(t != 0) for (_, t) in batch))
        c = reproduce(init, children, step)
        if mono: c.extend(init)
        assert len(c) == mono * len(init) + len(init) * children
        if _debug: print("Number of candidates: ", len(c))
        if len(c) <= survivors: init = c
        else:
            def err(cc):
                return erf(model, cc, batch)
            c.sort(key=err)
            init = c[:survivors]
            if _debug: print("Best error: ", err(init[0]))
    return init

#Returns the Euclidean, wrap around distance between p1 and p2
def dist(p1, p2):
    def diffsq(v1, v2):
        o = v1 - v2
        while o > math.pi: o -= 2 * math.pi
        while o < -math.pi: o += 2 * math.pi
        return o * o
    return sum(diffsq(p1[k], p2[k]) for k in p1)

#Genetic algorithm but with bias towards diverse populations
def variety_train(model, init, generator, erf, step, 
    children=4, survivors=8, batch_size=1, mono=False, var_weight=1):
    while datagen.has_next(generator):
        if _debug: print("Next batch!")
        batch = datagen.next_batch(generator, batch_size)
        c = reproduce(init, children, step)
        if mono: c.extend(init)
        assert len(c) == mono * len(init) + len(init) * children
        if _debug: print("Number of candidates: ", len(c))
        if len(c) <= survivors: init = c
        else:
            def err(cc):
                return erf(model, cc, batch)
            errors = [err(cc) for cc in c]
            s = []
            while len(s) < survivors:
                i = errors.index(min(errors))
                if _debug and len(s) == 0: print("Best error: ", min(errors))
                s.append(c[i])
                for j in range(len(errors)):
                    errors[j] -= var_weight * dist(c[i], c[j])
                c.remove(c[i])
                errors.remove(errors[i])
            init = s
    return init
        
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
        output.update(p)
        return cirq.ParamResolver(output)
    params = map(to_resolver_form, pl)
    observables = list(cirq.Z(pp) for pp in p)
    return s.simulate_expectation_values_sweep(
        program=m, 
        observables=observables, params=cirq.study.ListSweep(params))
    
#Genetic algorithm, but implemented using cirq param sweeps. Difference is that
# the erf input is replaced with [sample_erf]. [sample_erf] takes an expectation
# output and the expected type, and returns the error.
def genetic_sweep(model, init, generator, sample_erf, step, 
    children=8, survivors=4, batch_size=1, mono=False):
    while datagen.has_next(generator):
        if _debug: print("Next batch!")
        #Generate a batch + candidates
        batch = datagen.next_batch(generator, batch_size)
        c = reproduce(init, children, step)
        if mono: c.extend(init)
        if _debug: print("Number of candidates: ", len(c))
        #Compute the errors
        errors = [0 for _ in range(len(c))]
        for (d, t) in batch:
            d = datagen.normalize(d)
            o = compute_sweep(model, d, c)
            for i in range(len(c)):
                errors[i] += sample_erf(o[i], t)
        if _debug: print("Errors computed: ", errors)
        #Now select the best.
        indices = [i for i in range(len(c))]
        indices.sort(key=lambda x : errors[x])
        if _debug: print("Sorted indices: ", indices)
        init = tuple(c[i] for i in indices[:survivors])
    return init
        
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

