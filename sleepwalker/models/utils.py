import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        num_timescales = (dim + 1) // 2
        div_term = torch.exp(torch.arange(0, num_timescales).float() * (-math.log(10000.0) / dim))
        angle_rates = position * div_term

        pe[:, 0::2] = torch.sin(angle_rates[:, : dim // 2 + dim % 2])
        pe[:, 1::2] = torch.cos(angle_rates[:, : dim // 2])

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)

class AttentionPooling(nn.Module):
    def __init__(self, dim, attn_size, along_dimension=1):
        super().__init__()
        self.Wa = nn.Linear(dim, attn_size)
        self.ae = nn.Parameter(torch.randn(attn_size))
        self.ba = nn.Parameter(torch.zeros(attn_size))
        self.along_dimension = along_dimension

    def forward(self, x):
        a_t = torch.tanh(self.Wa(x) + self.ba)
        e_t = torch.matmul(a_t, self.ae)
        alpha = torch.softmax(e_t, dim=self.along_dimension)
        out = torch.sum(x * alpha.unsqueeze(-1), dim=self.along_dimension)
        return out, alpha
