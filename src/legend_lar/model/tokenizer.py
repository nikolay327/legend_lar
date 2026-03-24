import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from legend_lar.model.cls import _create_mha_cls, _create_mlp_cls
from legend_lar.model.block import UnconditionalBlock

class DiscreteEmbedder(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int):
        super(DiscreteEmbedder, self).__init__()
        self.emb = nn.Embedding(codebook_size, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
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

class ParallelContinuousEmbedder:
    def __init__(
        self,
        num_features: int,
        emb_dim: int
    ):
        self.weight = nn.Parameter(
            torch.empty((num_features, emb_dim, 1))
        )
        self.bias = nn.Parameter(torch.empty(num_features, emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
            x: (B, num_features, 1, 1)
            return: (B, num_features, D)
        """
        return torch.matmul(self.weight, x).squeeze(-1) + self.bias


class HPGewithPSD(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hpge_codebook_size: int,
        global_partitioning_size: int,
        hidden_dim: int,
        num_of_features: int,
        num_attn_heads: int,
        num_layers: int
    ):
        super(HPGewithPSD, self).__init__()
        self.num_of_features = num_of_features
        self.hpge_emb = DiscreteEmbedder(
            codebook_size=hpge_codebook_size,
            emb_dim=emb_dim
        )
        self.partitioning_emb = DiscreteEmbedder(
            codebook_size=global_partitioning_size,
            emb_dim=emb_dim
        )

        self.features_emb = ParallelContinuousEmbedder(
            num_features=num_of_features,
            emb_dim=emb_dim
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, emb_dim))

        self.norm_in = nn.LayerNorm(emb_dim)
        self.mixer = nn.ModuleList([
            UnconditionalBlock(
                emb_dim=emb_dim,
                mixer_cls=_create_mha_cls(num_attention_heads=num_attn_heads, causal=False),
                mlp_cls=_create_mlp_cls(intermediate_size=hidden_dim),
                resid_dropout1=0.,
                resid_dropout2=0.
            ) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(emb_dim)

    def forward(self, gid: Tensor, pid: Tensor, features: Tensor):
        """
            gid: (B,)
            features: (B, num_of_features, 1)
        """
        e_gid = self.hpge_emb(gid).unsqueeze(1) # (B, 1, D)
        e_pid = self.partitioning_emb(pid).unsqueeze(1) # (B, 1, D)

        features = features.reshape(-1, self.num_of_features, 1, 1)
        e_feat = self.features_emb(features) # (B, num_of_features, D)

        emb = torch.cat((self.cls_token, e_gid, e_pid, e_feat), dim=1) # (B, num_of_features+3, D)
        emb = self.norm_in(emb)
        residual = None
        for block in self.mixer:
            emb, residual = block(
                hidden_states=emb,
                residual=residual,
                cu_seqlens=None,
                max_seqlen=None
            )

        emb = self.norm_out(emb[:, 0, :] + residual[:, 0, :])
        return emb
