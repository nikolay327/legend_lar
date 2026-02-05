from torch import Tensor
import torch.nn as nn

class DiscreteEmbedder(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int):
        super(DiscreteEmbedder, self).__init__()
        self.emb = nn.Embedding(codebook_size, emb_dim)

    def forward(self, x: Tensor):
        return self.emb(x)

class ContinuousEmbedder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(ContinuousEmbedder, self).__init__()
        
        self.emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x: Tensor):
        if x.dim() == 1:
            x = x[:, None]
        return self.emb(x)
