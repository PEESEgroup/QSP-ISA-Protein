
#For this task, make a transformer that memorizes sequences of numbers
#The numbers will be integers from 0 to 9 inclusive, plus a special number for
#start and a special number for end.
#Numbers will be one-hot encoded, as a length 12 vector, where only the [n]th
# entry is 1 when representing [n]. Start is treated as 10 and end is treated
# as 11.

import random
import torch
import math

def one_hot(i, d):
    return torch.tensor([int(j == i) for j in range(d)]).float()

def concat(t):
    return torch.cat([x.reshape(1, -1) for x in t], dim=0)

#On input xl to be encoded, and dl as the partially decoded, train the
# transformer to output do
data_xl = []
data_dl = []
data_do = []

random.seed(42)

def smooth(v):
    return v * 0.9 + 1.0 / 120

for _ in range(2000):
    x = [one_hot(10, 12)]
    x.extend([one_hot(random.randint(0, 9), 12) 
    for _ in range(random.randint(2, 10))])
    x.append(one_hot(11, 12))
    data_xl.append(concat(x[1:-1]))
    data_dl.append(concat(x[:-1]))
    data_do.append(smooth(concat(x[1:])))

def round(ls):
    if isinstance(ls, list):
        return [round(x) for x in ls]
    return int(ls * 1000) / 1000

def test(model, x):
    enc = concat([one_hot(xx, 12) for xx in x])
    dls = [one_hot(10, 12).reshape(1, -1)]
    out = []
    confidence = 1
    for _ in range(len(x)):
        y = model(enc, concat(dls), log=True)
        print(round(y.tolist()))
        dls.append(y[-1])
        yls = y.tolist()
        m = max(yls[-1])
        confidence *= m
        out.append(yls[-1].index(m))
    print("On input: " + str(x))
    print("Predicted: " + str(out))
    print("Confidence: " + str(confidence))

def total_loss(model, loss):
    output = 0
    for xl, dl, do in zip(data_xl, data_dl, data_do):
        pred = model(xl, dl)
        l = loss(pred, do)
        output += l
    return output

from transformer import Transformer
import torch.nn as nn

model = Transformer(12, 4, 48, 12)
model.load_state_dict(torch.load("save.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

print("Before training loss: " + str(total_loss(model, loss)))

for epoch in range(0):
    i = 0
    loss_accumulated = 0
    for x, d, y in zip(data_xl, data_dl, data_do):
        i += 1
        pred = model(x, d)
        l = loss(pred, y)
        loss_accumulated += l
        if i % 60 == 0:
            optimizer.zero_grad()
            loss_accumulated.backward()
            optimizer.step()
            loss_accumulated = 0
    print("After epoch " + str(epoch) + ": " + str(total_loss(model, loss)))

torch.save(model.state_dict(), "save.pth")
test(model, [1, 2, 3, 4, 5, 6, 7, 8])
