import torch
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

class JointHPGeEmbedder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hpge_codebook_size: int,
        hidden_dim: int
    ):
        super(JointHPGeEmbedder, self).__init__()
        self.hpge_emb = DiscreteEmbedder(
            codebook_size=hpge_codebook_size,
            emb_dim=emb_dim
        )
        self.hpge_energy_emb = ContinuousEmbedder(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim
        )

        self.joint_emb = nn.Sequential(
            nn.Linear(2 * emb_dim, 2 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * hidden_dim, emb_dim)
        )
    
    def forward(self, g: Tensor, E: Tensor):
        e_g = self.hpge_emb(g)
        e_E = self.hpge_energy_emb(E)
        e = torch.cat((e_g, e_E), dim=-1)
        e = self.joint_emb(e)
        return e
