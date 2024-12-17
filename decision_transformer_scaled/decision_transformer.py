import torch
import torch.nn as nn
from data import Gate, BatchStream
import math

class SigmoidNN(nn.Module):
    def __init__(self, dims_list):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dims_list[i], dims_list[i + 1]) 
        for i in range(len(dims_list) - 1)])
    def forward(self, x):
        for l in self.layers: 
            x = torch.sigmoid(l(x))
        return x

class ReluNN(nn.Module):
    def __init__(self, dims_list):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dims_list[i], dims_list[i + 1])
        for i in range(len(dims_list) - 1)])
        self.activation = nn.LeakyReLU()
        assert(len(list(self.parameters())) > 0)
    def forward(self, x):
        for l in self.layers:
            x = self.activation(l(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, d_model, n_qubits, previews, n_layers=4, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.s_dim = 1 << (n_qubits + 1)
        self.a_dim = n_qubits + len(Gate.types) + 1
        self.embed_reward = nn.Linear(2, d_model)
        self.embed_state = nn.Linear(self.s_dim, d_model)
        self.embed_action = nn.Linear(self.a_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_layers, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_heads)
        self.position_matrix = nn.Parameter(torch.randn(previews + 1, d_model))
        self.project_type = nn.Linear(d_model, len(Gate.types), bias=True)
        self.project_index = nn.Linear(d_model, n_qubits, bias=True)
        self.project_angle = nn.Linear(d_model, 1, bias=True)

    def forward(self, batch):
        batch_encoded = []
        for seq in batch:
            embedded = []
            for i in range(0, len(seq) - 2, 3):
                pos_enc = self.position_matrix[int(i / 3)]
                r, s, a = seq[i:i + 3]
                r = torch.tensor(r).float()
                s = torch.tensor(s).float()
                a = torch.tensor(a).float()
                embedded.append(self.embed_reward(r) + pos_enc)
                embedded.append(self.embed_state(s) + pos_enc)
                embedded.append(self.embed_action(a) + pos_enc)
            pos_enc = self.position_matrix[int(len(seq) / 3)]
            embedded.append(self.embed_reward(torch.tensor(seq[-2]).float()) + pos_enc)
            embedded.append(self.embed_state(torch.tensor(seq[-1]).float()) + pos_enc)
            batch_encoded.append(torch.cat([v.reshape(1, -1) for v in embedded], dim=0))
        batch_encoded = torch.cat([v.reshape(1, -1, self.d_model) for v in batch_encoded], dim=0)
        l = batch_encoded.shape[1]
        src_mask = -10000 * torch.tensor([[int(j > i) for j in range(l)] for i in range(l)])
        src_mask = src_mask.float()
        processed = self.encoder(batch_encoded, mask=src_mask)
        #processed = self.encoder(batch_encoded)
        output = []
        for seq in processed:
            a_seq = []
            for i in range(1, len(seq), 3):
                a_seq.append(self.project(seq[i]))
            output.append(torch.cat([v.reshape(1, -1) for v in a_seq], dim=0))
            #output.append(a_seq[-1])
        return torch.cat([v.reshape(1, -1, self.a_dim) for v in output], dim=0)
    def project(self, v):
        output = []
        output.append(torch.sigmoid(self.project_type(v)))
        output.append(torch.sigmoid(self.project_index(v)))
        output.append(self.project_angle(v))
        return torch.cat(output, dim=0)

