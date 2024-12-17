import torch
import torch.nn as nn
import torch.nn.functional as F

#Simple MNIST classifier, densely connected. 784 -> 16 -> 10 -> 10.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w1 = torch.rand((16, 784)) - 0.5
        self.b1 = torch.rand((16,)) - 0.5
        self.w2 = torch.rand((10, 16)) - 0.5
        self.b2 = torch.rand((10,)) - 0.5
        self.w3 = torch.rand((10, 10)) - 0.5
        self.b3 = torch.rand((10,)) - 0.5
        self.w1.requires_grad = True
        self.w2.requires_grad = True
        self.w3.requires_grad = True
        self.b1.requires_grad = True
        self.b2.requires_grad = True
        self.b3.requires_grad = True
    def forward(self, x):
        x = self.w1.matmul(x)
        x = x + self.b1
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.w2.matmul(x) + self.b2)
        x = torch.sigmoid(self.w3.matmul(x) + self.b3)
        return x
    def get_extra_state(self):
        return (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
    def set_extra_state(self, state):
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = state

net = Net()

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = [torch.from_numpy(xx.flatten() / 256.0).float()
for xx in x_train]
x_test = [torch.from_numpy(xx.flatten() / 256.0).float()
for xx in x_test]
y_train = [torch.tensor([int(y == i) for i in range(10)]).float() 
for y in y_train]
y_test = [torch.tensor([int(y == i) for i in range(10)]).float() 
for y in y_test]
print("Loaded data")

import torch.optim as optim
optimizer = optim.SGD([net.w1, net.b1, net.w2, net.b2, net.w3, net.b3], lr=0.1)
criterion = nn.MSELoss()

def test_accuracy(network):
    correct = 0
    total = 0
    for x, y in zip(x_test, y_test):
        output = network(x)
        if output.tolist().index(max(output)) == y.tolist().index(max(y)): 
            correct += 1
        total += 1
    return correct / total

print("Pre-training accuracy: ", test_accuracy(net))
for epoch in range(10):
    i = 0
    for x, y in zip(x_train, y_train):
        i += 1
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print("After epoch ", epoch, " accuracy: ", test_accuracy(net))

torch.save(net.state_dict(), "save.pth")

net2 = Net()
net2.load_state_dict(torch.load("save.pth"))

print("loaded model accuracy: ", test_accuracy(net2))
