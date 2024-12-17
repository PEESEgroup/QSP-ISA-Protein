
import layer
import layers
import datagen
import vqc_model as vm
import nn

_from_save = False
_true_type = 0
_false_type = 1
_save_path = "Saves/vqc_two_cathle" + str(_true_type) + str(_false_type)
vqc = vm.new_model(7, 1)
vm.add_layer(vqc, vm.RX, "x1")
vm.add_layer(vqc, vm.CX)
vm.add_layer(vqc, vm.RX, "x2")
vm.add_layer(vqc, vm.S_CX)
vm.add_layer(vqc, vm.ORX, "x3")

param = None
if _from_save:
    param = vm.load(_save_path)

l = layers.prunable_vqc_to_layer(vqc, param, s=0.8)
model = nn.from_layers((l,))

weights = tuple(int(i in(_true_type, _false_type)) for i in range(4))
gen = datagen.cstr_gen_normal_weighted(1, True, weights)

def evaluate(steps):
    print("Evaluating on the training set...")
    vm.save(l[1][1], _save_path)
    train_gen = datagen.cstr_gen_normal_weighted(1, True, weights)
    test_gen = datagen.cstr_gen_normal_weighted(1, False, weights)
    tr = [0, 0, 0, 0]
    cr = [0, 0, 0, 0]
    while datagen.has_next(train_gen):
        ((s,), t) = datagen.next(train_gen)
        (z,) = nn.compute(model, s)
        if (z > 0) == (t == _true_type): cr[t] += 1
        tr[t] += 1
    print("Evaluating on the testing set...")
    te = [0, 0, 0, 0]
    ce = [0, 0, 0, 0]
    while datagen.has_next(test_gen):
        ((s,), t) = datagen.next(test_gen)
        (z,) = nn.compute(model, s)
        if (z > 0) == (t == _true_type): ce[t] += 1
        te[t] += 1
    print("Training correctness after ", steps, " samples: ")
    print("    ", cr, " out of ", tr)
    print("Testing correctness after ", steps, " samples: ")
    print("    ", ce, " out of ", te)

def msg(v, w):
    return tuple(vv - ww for vv, ww in zip(v, w))

def mcg(v, w):
    return tuple((vv - ww)**3 / 4 for vv, ww in zip(v, w))

def hle(v, w):
    return (-w[0],)

def bceg(v, w, threshold=1e-4):
    assert len(v) == len(w) == 1
    if w[0] >= 1 - threshold: 
        if v[0] + 1 < threshold: return -1 / threshold
        return (-2 / (1 + v[0]) + 1,)
    if w[0] <= -1 + threshold:
        if 1 - v[0] < threshold: return 1 / threshold
        return (2 / (1 - v[0]) - 1,)
    print("Error. w is not 1 or -1, instead is ", w)
samples = 0 
while datagen.has_next(gen):
    if samples % 100 == 0: evaluate(samples)
    print("Training! Looking at sample ", samples)
    samples += 1
    (d,), t = datagen.next(gen)
    w = (2 * int(t == _true_type) - 1,)
    nn.train(model, d, w, hle, 0.1)

evaluate(samples)
