
import proto_model
import data
import datagen

import cirq

_save_prefix = "results/Detect"

finalists = []
for i in range(4):
    finalists.append(proto_model.load(_save_prefix + str(i)))

m = proto_model.new_model(7, 1)
proto_model.add_layer(m, proto_model.RX, "x1")
proto_model.add_layer(m, proto_model.RY, "y1")
proto_model.add_layer(m, proto_model.CRX, "cx1")
proto_model.add_layer(m, proto_model.RX, "x2")
proto_model.add_layer(m, proto_model.RY, "y2")
proto_model.add_layer(m, proto_model.S_CX, "scx2")
proto_model.add_layer(m, proto_model.RX, "x3")
proto_model.add_layer(m, proto_model.RY, "y3")

means = data.get_mean()
stdev = data.get_stdev()

results = []
for i, f in enumerate(finalists):
    print("Testing parameter set ", i, "... ")
    results.append("For parameter set " + str(i))
    train_errs = 0
    for i in range(datagen.train_tot):
        print("Train sample ", i)
        (s, t) = datagen.train_sample(i)
        s = tuple((s[j] - means[j]) / stdev[j] for j in range(len(s)))
        (z,) = proto_model.compute(m, s, f)
        print(z)
        if (z > 0) == (t == 0): 
            print("INCORRECT")
            train_errs += 1
        else:
            print("CORRECT")
    results.append("Training set. " + str(datagen.train_tot) + " samples, " + 
        str(train_errs) + " incorrect.")
    test_errs = 0
    for i in range(datagen.test_tot):
        print("Test sample ", i)
        (s, t) = datagen.test_sample(i)
        (z,) = proto_model.compute(m, s, f)
        print(z)
        if (z > 0) == (t == 0): 
            print("INCORRECT")
            test_errs += 1
        else:
            print("CORRECT")
    results.append("Test set. " + str(datagen.test_tot) + "samples, " + 
        str(test_errs) + " incorrect.")

for s in results:
    print(s)
