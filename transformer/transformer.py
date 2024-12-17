
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, mask):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.mask = mask
    #qx is a torch 2d tensor, first index is the position, second index is the
    # element. kx, vx are key and value torch 2d tensors with the same format
    def forward(self, qx, kx, vx):
        q = self.query(qx)
        k = self.key(kx)
        v = self.value(vx)
        scores = q @ k.transpose(0, 1) / math.sqrt(self.out_dim)
        n = len(qx)
        if(self.mask):
            maskmat = [[int(j > i) for j in range(n)] for i in range(n)]
            maskmat = -10000 * torch.tensor(maskmat).float()
            scores = scores + maskmat
        scores = self.softmax(scores)
        return scores @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, heads, mask):
        super().__init__()
        assert(in_dim % heads == 0)
        out_dim = int(in_dim / heads)
        self.attn = nn.ModuleList(
          [Attention(in_dim, out_dim, mask) for _ in range(heads)])
        self.combine = nn.Linear(in_dim, in_dim)
    def forward(self, qx, kx, vx):
        results = [attn(qx, kx, vx) for attn in self.attn]
        return self.combine(torch.cat(results, dim=1))

class Residual(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
    def forward(self, x, fx):
        x = x + fx
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class FeedForward(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, h_dim)
        self.l2 = nn.Linear(h_dim, in_dim)
    def forward(self, x):
        return self.l2(self.l1(x).relu())

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h_dim, heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, False)
        self.residual1 = Residual(d_model)
        self.residual2 = Residual(d_model)
        self.ffnn = FeedForward(d_model, h_dim)
    def forward(self, x):
        fx = self.attn(x, x, x)
        x = self.residual1(x, fx)
        fx = self.ffnn(x)
        x = self.residual2(x, fx)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h_dim, heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, True)
        self.edattn = MultiHeadAttention(d_model, heads, False)
        self.residual1 = Residual(d_model)
        self.residual2 = Residual(d_model)
        self.residual3 = Residual(d_model)
        self.ffnn = FeedForward(d_model, h_dim)
    def forward(self, x, memory):
        fx = self.attn(x, x, x)
        x = self.residual1(x, fx)
        fx = self.edattn(x, memory, memory)
        x = self.residual2(x, fx)
        fx = self.ffnn(x)
        return self.residual3(x, fx)

#main model: embed the words, add positional encoding, go through six encoder
# layers, go through six decoder layers, use a linear and softmax at the end
# to determine the word probabilities

#Computes the positional encoding for position [pos], for a vector of dimension
# d_model. d_model must be even
def positional_encoding(pos, d_model):
    scale = 1000 ** (2 / d_model)
    output = []
    for i in range(d_model >> 1):
        output.append(math.sin(pos / (scale ** i)))
        output.append(math.cos(pos / (scale ** i)))
    return torch.tensor(output).float()

def positional_encoding_matrix(d_model, length):
    output = [positional_encoding(pos, d_model).reshape(1, -1) 
        for pos in range(length)]
    return torch.cat(output, dim=0)

class Transformer(nn.Module):
    def __init__(self, d_model, heads, h_dim, max_len):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderLayer(d_model, h_dim, heads) 
            for _ in range(6)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, h_dim, heads)
            for _ in range(6)])
        self.max_len = max_len
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        self.pos_enc = positional_encoding_matrix(d_model, max_len)
        self.pos_enc.requires_grad = False
    def forward(self, xl, dl, log=False):
        if log: 
            print("input: ", xl)
            print("decoded so far: ", dl)
        m = xl + self.pos_enc[0:len(xl)]
        if log:
            print("input w positional encoding: ", m)
        for encoder in self.encoders:
            m = encoder(m)
            if log:
                print("encoded: ", m)
        out = dl + self.pos_enc[0:len(dl)]
        if log:
            print("decoded so far w positional encoding: ", out)
        for decoder in self.decoders:
            out = decoder(out, m)
            if log:
                print("decoded: ", out)
        out = F.softmax(out, dim=1)
        return out

