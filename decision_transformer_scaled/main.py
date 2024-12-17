from data import BatchStream, Gate
from decision_transformer import DecisionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

save_file = "save.pth"

num_qubits = 5
model = DecisionTransformer(128, num_qubits, 4, n_layers=4, n_heads=4)
if save_file in os.listdir():
    model.load_state_dict(torch.load(save_file))
else:
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss()
    def forward(self, x, y, requires_grad=True):
        if type(x) is list: x = torch.tensor(x).float()
        if type(y) is list: y = torch.tensor(y).float()
        xt, xi, xa = x.split([len(Gate.types), num_qubits, 1], dim=-1)
        yt, yi, ya = y.split([len(Gate.types), num_qubits, 1], dim=-1)
        cosa = torch.mean(1 - torch.cos(xa - ya))
        output = self.mseloss(xt, yt) + self.mseloss(xi, yi) + cosa
#        output = self.mseloss(x, y)
        if requires_grad: return output
        else: return float(output)

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = Loss()

def compute_test_loss():
    model.eval()
    total = 0
    test_stream = BatchStream("testing_data.json")
    i = 0
    while test_stream.hasNext():
        i += 1
        print("Testing sample: ", i)
        x, y = test_stream.next()
        output = model(x)
        loss = criterion(output, y, requires_grad=False)
        total += loss
    test_stream.close()
    model.train()
    return total

losses = []
losses.append(compute_test_loss())
print("Before training loss: ", losses[-1])
for epoch in range(50):
    train_stream = BatchStream("training_data.json")
    i = 0
    while train_stream.hasNext():
        i += 1
        print("Epoch ", epoch, "training sample", i)
        x, y = train_stream.next()
        output = model(x)
        optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    train_stream.close()
    losses.append(compute_test_loss())
    if losses[-1] == min(losses):
        print("Save!")
        torch.save(model.state_dict(), save_file)
    print("After epoch", epoch, "loss: ", losses[-1])

out_file = "result.csv"
while out_file in os.listdir(): out_file = "$" + out_file
with open(out_file, "w") as f:
    f.write("Epoch,Training loss\n")
    for i, l in enumerate(losses):
        f.write(str(i) + "," + str(l) + "\n")

