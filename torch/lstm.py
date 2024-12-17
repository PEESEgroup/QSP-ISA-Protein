
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#On the forward pass, returns the pair, (c, h), where [c] is the long term
#memory and [h] is the short term memory
class LSTM(nn.Module):
    def __init__(self, x_dim, c_dim):
        super().__init__()
        self._x_dim = x_dim
        self._c_dim = c_dim
        i_dim = x_dim + c_dim
        self._wf = nn.Linear(i_dim, c_dim)
        self._wi = nn.Linear(i_dim, c_dim)
        self._wo = nn.Linear(i_dim, c_dim)
        self._wc = nn.Linear(i_dim, c_dim)
    def forward(self, x_list):
        c = torch.zeros((self._c_dim,))
        h = torch.zeros((self._c_dim,))
        for x in x_list:
            i = torch.cat((x, h))
            ff = torch.sigmoid(self._wf(i))
            ii = torch.sigmoid(self._wi(i))
            cc = torch.tanh(self._wc(i))
            oo = torch.sigmoid(self._wo(i))
            c = ff * c + ii * cc
            h = torch.tanh(c) * oo
        return (c, h)
    
    def get_extra_state(self):
        return (self._x_dim, self._c_dim)
    def set_extra_state(self, state):
        self._x_dim, self._c_dim = state

if __name__ == "__main__":
    import lstm2
    net = lstm2.LSTMwNN(1, 5, 1)
    net.load_state_dict(torch.load("lstm_save.pth"))
    import math
    data = [[math.sin(x / 10)] for x in range(1000)]
    optimizer = optim.SGD(net.parameters(), lr = 0.1)
    criterion = nn.MSELoss()
    
    #use first 400 for training, next 400 for testing
    
    def test_loss():
        total = 0
        for i in range(400, 800):
            x = torch.tensor(data[i:i + 4])
            y = torch.tensor(data[i + 4])
            o = net(x)
            total += (y - o) ** 2
        return total
    
    print(test_loss())
    for _ in range(10):
        for i in range(400):
            optimizer.zero_grad()
            x = torch.tensor(data[i:i + 4])
            y = torch.tensor(data[i + 4])
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        print(test_loss())
    torch.save(net.state_dict(), "lstm_save.pth")
