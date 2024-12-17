import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class SubLinear(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(SubLinear, self).__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.weight = Parameter(torch.rand((out_dims, in_dims)))

    def forward(self, x):
        return self.weight.data.matmul(x)

class CompLinear(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(CompLinear, self).__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.w1 = SubLinear(in_dims, out_dims)
        self.w2 = SubLinear(in_dims, out_dims)

    def forward(self, x):
        return self.w1(x) + self.w2(x)

class BigLinear(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(BigLinear, self).__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.w1 = CompLinear(in_dims, out_dims)
        self.w2 = CompLinear(in_dims, out_dims)

    def forward(self, x):
        return self.w1(x) + self.w2(x)

m = SubLinear(2, 1)
x = torch.tensor([1, 1]).float()
print(m(torch.tensor([1, 1]).float()))
print(list(m.parameters()))

m2 = CompLinear(2, 1)
print(m2(x))
print(list(m2.parameters()))

m3 = BigLinear(2, 1)
print(m3(x))
print(list(m3.parameters()))
print(m3.state_dict())
