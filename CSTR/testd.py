import proto_model as pm
import datagen
import data

_save_prefix = "results/Diag"
_type_flag = "t"

m = pm.new_model(7, 1)
pm.add_layer(m, pm.RX, "x1")
pm.add_layer(m, pm.RY, "y1")
pm.add_layer(m, pm.CRX)
pm.add_layer(m, pm.RX, "x2")
pm.add_layer(m, pm.RY, "y2")
pm.add_layer(m, pm.S_CX)
pm.add_layer(m, pm.RX, "x3")
pm.add_layer(m, pm.RY, "y3")

model = []

def filename(type, index):
    return _save_prefix + str(index) + _type_flag + str(type)

for i in range(4):
    model.append(pm.load(filename(i, 0)))

means = data.get_mean()
stdev = data.get_stdev()

def normalize(sample):
    return tuple((sample[i] - means[i]) / stdev[i] for i in range(len(sample)))

def compute(sample):
    def zi(i):
        (z,) = pm.compute(m, normalize(sample), model[i])
        return z.real
    return tuple(zi(i) for i in range(len(model)))

correct = [0, 0, 0, 0]
totals = [0, 0, 0, 0]
for i in range(datagen.train_tot):
    (s, t) = datagen.train_sample(i)
    output = compute(s)
    tt = output.index(max(output))
    def ind_correct(i):
        if i == t: return int(output[i] > 0)
        else: return int(output[i] < 0)
    ind = tuple(ind_correct(i) for i in range(len(output)))
    print("Testing Point ", i)
    print("Raw output: ", output)
    print("Individual correctness: ", ind)
    totals[t] += 1
    if tt == t:
        print("Overall correctness: CORRECT")
        correct[t] += 1
    else:
        print("Overall correctness: INCORRECT")

print("Totals (by type): ", totals)
print("Correct (by type): ", correct)
