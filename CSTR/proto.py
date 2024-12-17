
import cirq
import sympy

model = cirq.Circuit()
#Each input consists of 7 floats. Represent the initial floats by applying
# X-rotation an amount equal to the z-score.
# Next, add dense layers of qubits, keep track of the parameters. Then, train

#Input qubits q1 ... q7, output qubit p1
q1, q2, q3, q4, q5, q6, q7 = cirq.LineQubit.range(1, 8)
inputs = (q1, q2, q3, q4, q5, q6, q7)
p1 = cirq.LineQubit(-1)
outputs = (p1,)

#Input parameters i1 ... i7
i1, i2, i3, i4, i5, i6, i7 = sympy.symbols("i1 i2 i3 i4 i5 i6 i7")
b = sympy.symbols("b")
model.append(cirq.rx(i1).on(q1))
model.append(cirq.rx(i2).on(q2))
model.append(cirq.rx(i3).on(q3))
model.append(cirq.rx(i4).on(q4))
model.append(cirq.rx(i5).on(q5))
model.append(cirq.rx(i6).on(q6))
model.append(cirq.rx(i7).on(q7))
model.append(cirq.rx(b).on(p1))
params = [b]

def add_x(prefix):
    for i, q in enumerate(inputs):
        s = sympy.Symbol(prefix + "q" + str(i))
        params.append(s)
        model.append(cirq.rx(s).on(q))
    for j, p in enumerate(outputs):
        s = sympy.Symbol(prefix + "p" + str(j))
        params.append(s)
        model.append(cirq.rx(s).on(p))

def add_layer_doubles(circuit, gate, prefix):
    for i, q in enumerate(inputs):
        for j, p in enumerate(outputs):
            s = sympy.Symbol(prefix + "qp" + str(i) + "/" + str(j))
            params.append(s)
            circuit.append(gate(q, p)**s)

add_x("x1")
add_layer_doubles(model, cirq.CX, "cx1")
add_x("x2")
add_layer_doubles(model, cirq.CX, "cx2")
c = sympy.Symbol("c")
params.append(c)
model.append(cirq.rx(c).on(p1))

#Now to make params happen
import random

vals = {}
for s in params: vals[s] = random.random()

#Simulates the circuit [model] on parameter values [v], taking non-normalized 
# inputs [d]. Returns the expectation of Z on the readout qubit.
def compute(v, d):
    mean = data.get_mean()
    stdev = data.get_stdev()
    vv = {
        i1 : (d[0] - mean[0]) / stdev[0],
        i2 : (d[1] - mean[1]) / stdev[1],
        i3 : (d[2] - mean[2]) / stdev[2],
        i4 : (d[3] - mean[3]) / stdev[3],
        i5 : (d[4] - mean[4]) / stdev[4],
        i6 : (d[5] - mean[5]) / stdev[5],
        i7 : (d[6] - mean[6]) / stdev[6],
    }
    vv.update(v)
    s = cirq.Simulator()
    resolver = cirq.ParamResolver(vv)
    output = s.simulate(model, resolver)
    z = cirq.Z(outputs[0])
    final_state = output.final_state_vector
    qm = output.qubit_map
    return z.expectation_from_state_vector(final_state, qm)

#Helper functions for training
import data
import math
#Computes the error of parameter set [v] on [count] random samples of the
# training set. If not specified, count will be set to 100
def train_err(v, count=50, categorical=False):
    err = 0
    for _ in range(count):
        type = random.randint(0, 3)
        n = data.num_samples(True, type)
        i = random.randint(0, n - 1)
        sample = data.get_sample(True, type, i)
        output = compute(v, sample).real
        print(output)
        if categorical and ((type > 0) != (output > 0.5)): err += 1
        elif not categorical: err += 1 - output if type != 0 else output
    return err

#Generates a random parameter set, with all parameters values equal to that in
# [v] within an amount [step]. 
def child(v, step):
    vv = {}
    vv.update(v)
    for s in vv:
        if random.randint(0, 1): 
            vv[s] += step
            while(vv[s] > math.pi): vv[s] -= 2 * math.pi
        else: 
            vv[s] -= step
            while(vv[s] < -math.pi): vv[s] += 2 * math.pi
    return vv

#Function for performing the genetic training algorithm for PQC. [init_v] is
# the starting set of parameters; each parameter set generates [num_kids]
# new parameter sets, then the offspring are filtered until the [num_survive]
# best remain. The algorithm proceeds for [iter] iterations. If [immortal], then
# the parent parameter sets are kept in the candidate survivor pool, otherwise
# the parent parameter sets are immediately killed off after producing children.
# Each child parameter set has each parameter value [step] different from its
# parent.
# [init_v] is a list of parameter sets, must contain at least one parameter set
# but no more than [num_survive]. The output is a new list containing the
# parameter sets that survived training.
def genetic_train(init_v, num_kids, num_survive, iter, step, 
    immortal=False, cat=False):
    #Iterate two steps: make kids, then kill them off.
    output = init_v
    for _ in range(iter):
        c = []
        if immortal: c.extend(output)
        for v in output:
            for _ in range(num_kids):
                c.append(child(v, step))
        #Now filter the candidates
        if len(c) <= num_survive: output = c
        else:
            indices = [i for i in range(len(c))]
            errors = [train_err(c[i], categorical=cat) for i in range(len(c))]
            def get_err(i):
                return errors[i]
            indices.sort(key=get_err)
            output = [c[i] for i in indices[:num_survive]]
    return output

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
