
#another attempt to make lstm

import torch
import torch.nn as nn

class LSTMUnit(nn.Module):
    def __init__(self, x_dim, c_dim):
        super(LSTMUnit, self).__init__()
        self._x_dim = x_dim
        self._c_dim = c_dim
        i_dim = x_dim + c_dim
        self._wf = nn.Linear(i_dim, c_dim)
        self._wi = nn.Linear(i_dim, c_dim)
        self._wo = nn.Linear(i_dim, c_dim)
        self._wc = nn.Linear(i_dim, c_dim)

    def forward(self, h, c, x):
        i = torch.cat((x, h))
        ff = torch.sigmoid(self._wf(i))
        ii = torch.sigmoid(self._wi(i))
        cc = torch.tanh(self._wc(i))
        oo = torch.sigmoid(self._wo(i))
        c = ff * c + ii * cc
        h = torch.tanh(c) * oo
        return (c, h)

class LSTM(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(LSTM, self).__init__()
        self._unit = LSTMUnit(x_dim, h_dim)

    def forward(self, x_list):
        h = torch.zeros((self._unit._c_dim,))
        c = torch.zeros((self._unit._c_dim,))
        for x in x_list:
            c, h = self._unit(h, c, x)
        return (c, h)

class LSTMwTruncate(nn.Module):
    def __init__(self, x_dim, h_dim, o_dim):
        super(LSTMwTruncate, self).__init__()
        self._o_dim = o_dim
        self._lstm = LSTM(x_dim, h_dim)
    def forward(self, x_list):
        c, h = self._lstm(x_list)
        return h[0:self._o_dim]

class LSTMwNN(nn.Module):
    def __init__(self, x_dim, h_dim, o_dim):
        super(LSTMwNN, self).__init__()
        self._out = nn.Linear(h_dim, o_dim)
        self._lstm = LSTM(x_dim, h_dim)
    def forward(self, x_list):
        c, h = self._lstm(x_list)
        return torch.tanh(self._out(h))


