
import tensorflow.keras as tfk
import classical as c
import data_seq as ds

genr = ds.TrainDataSeq()
gene = ds.TestDataSeq()

resultsr = c.model.predict(genr)
resultse = c.model.predict(gene)
correctr = 0
correcte = 0
for i in range(len(resultsr)):
    l = resultsr[i].tolist()
    pt = l.index(max(l))
    tt = int(i / 1201)
    if pt == tt:
        print("Sample ", i, " correct")
        correctr += 1
    else:
        print("Sample ", i, " incorrect")
for i in range(len(resultse)):
    l = resultse[i].tolist()
    pt = l.index(max(l))
    tt = int(i / 1201)
    if pt == tt:
        print("Sample ", i, " correct")
        correcte += 1
    else:
        print("Sample ", i, " incorrect")

print("Total correct (training): ", correctr)
print("Total samples (training): ", len(resultsr))

print("Total correct (testing): ", correcte)
print("Total samples (testing): ", len(resultse))



