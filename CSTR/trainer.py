
import proto_model
import datagen
import data

_save_prefix = "results/Detect"
_from_save = True
_save_count = 4

gen = datagen.get_gen_weighted((1, 1.0 / 3, 1.0 / 3, 1.0 / 3))

mean = data.get_mean()
stdev = data.get_stdev()

#Returns the normalized version of raw data tuple [d]
def normalize(d):
    return tuple((d[i] - mean[i]) / stdev[i] for i in range(len(d)))

#build a model for distinguishing normal from faulty
m = proto_model.new_model(7, 1)
proto_model.add_layer(m, proto_model.RX, "x1")
proto_model.add_layer(m, proto_model.RY, "y1")
proto_model.add_layer(m, proto_model.CRX)
proto_model.add_layer(m, proto_model.RX, "x2")
proto_model.add_layer(m, proto_model.RY, "y2")
proto_model.add_layer(m, proto_model.RX, "x2", -1)
proto_model.add_layer(m, proto_model.RY, "y2", -1)
proto_model.add_layer(m, proto_model.S_CX)
proto_model.add_layer(m, proto_model.RX, "x3")
proto_model.add_layer(m, proto_model.RY, "y3")

init = []
if _from_save:
    for i in range(_save_count):
        init.append(proto_model.load(_save_prefix + str(i)))
else: init.extend([proto_model.rand_params(m) for _ in range(4)])

#square error loss for now, for 1 output
def sample_erf(output, type):
    (z,) = output
    z = z.real
    if type == 0: return (z + 1) ** 2
    else: return (1 - z) ** 2

init = proto_model.genetic_sweep(m, init, gen, sample_erf, 0.1, batch_size=50)

for i, d in enumerate(init):
    proto_model.save(d, _save_prefix + str(i))
