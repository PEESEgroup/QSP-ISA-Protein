
import proto_model as pm
import datagen

#Config variables
_save_prefix = "results/Diag"
_type_flag = "t"
_from_save = True
_save_count = 4

m = pm.new_model(7, 1)

pm.add_layer(m, pm.RX, "x1")
pm.add_layer(m, pm.RY, "y1")
pm.add_layer(m, pm.CRX)
pm.add_layer(m, pm.RX, "x2")
pm.add_layer(m, pm.RY, "y2")
pm.add_layer(m, pm.S_CX)
pm.add_layer(m, pm.RX, "x3")
pm.add_layer(m, pm.RY, "y3")

init = [[], [], [], []]

def filename(type, index):
    return _save_prefix + str(index) + _type_flag + str(type)

if _from_save:
    for i, l in enumerate(init):
        for j in range(_save_count):
            l.append(pm.load(filename(i, j)))
else:
    for i, l in enumerate(init):
        for j in range(4):
            l.append(pm.rand_params(m))

weight_d = {
    0 : (1.0, 0.5, 0.5, 0.5),
    1 : (0.2, 1.0, 0.2, 0.2),
    2 : (1.0 / 3, 1.0, 1.0 / 3, 1.0 / 3),
    3 : (1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0)
}
for i, l in enumerate(init):
    print("Training for type ", i)
    weights = weight_d[i]
    gen = datagen.get_gen_weighted(weights)
    #square error loss
    def sample_erf(output, t):
        (z,) = output
        if t == i: return (1 - z.real) ** 2
        else: return (z.real + 1) ** 2
    init[i] = pm.genetic_sweep(m, l, gen, sample_erf, 0.1, batch_size=50)

for i, l in enumerate(init):
    for j in range(_save_count):
        pm.save(l[j], filename(i, j))

