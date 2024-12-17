import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w1 = nn.Linear(2, 2)
        self.w2 = nn.Linear(2, 2)
        self.w3 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.w1(x))
        x = torch.sigmoid(self.w2(x))
        x = torch.sigmoid(self.w3(x))
        return x

net = Net()
net.load_state_dict(torch.load("load_save.pth"))
print(net(torch.tensor([0, 1]).float()))
torch.save(net.state_dict(), "load_save.pth")
print(list(net.parameters()))
