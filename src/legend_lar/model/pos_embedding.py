import math
import torch
from torch import Tensor
import torch.nn as nn

class SinPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int, device=None):
        super(SinPositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, emb_dim, dtype=torch.float32, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)

        # inv_freq[i] = 1/10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2, dtype=torch.float32, device=device) *
            -(math.log(10000.0) / emb_dim)
        ) # [emb_dim/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) # (L, D)
    
    def forward(self, x: Tensor, ids: Tensor):
        return x + self.pe[ids]
