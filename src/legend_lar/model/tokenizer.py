import math
from numpy import angle
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

class FourierScalarEmbedding(nn.Module):
    def __init__(
        self,
        num_bands: int = 16,
        max_freq_log2: float = 8.0,
        include_raw: bool = True,
        base_freq: float = math.pi,
        dtype: torch.dtype = torch.float32
    ):
        super(FourierScalarEmbedding, self).__init__()

        self.num_bands = int(num_bands)
        self.max_freq_log2 = float(max_freq_log2)
        self.include_raw = bool(include_raw)
        self.base_freq = float(base_freq)

        exponents = torch.linspace(0.0, self.max_freq_log2, steps=self.num_bands, dtype=dtype)
        freqs = self.base_freq * torch.pow(2.0, exponents) # (num_bands,)

        self.register_buffer("freqs", freqs, persistent=True)
    
    @property
    def out_dim(self) -> int:
        return (1 if self.include_raw else 0) + 2 * self.num_bands
    
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        angles = x.unsqueeze(-1) * self.freqs
        parts = []
        if self.include_raw:
            parts.append(x.unsqueeze(-1))
        parts.append(torch.sin(angles))
        parts.append(torch.cos(angles))

        return torch.cat(parts, dim=-1).contiguous()

class ContinuousFourierTokenizer(nn.Module):
    def __init__(
        self,
        n_cont: int,
        emb_dim: int,
        mlp_hidden_dim: int,
        num_bands: int = 16,
        max_freq_log2: float = 8.0,
        include_raw: bool = True,
        base_freq: float = math.pi,
        dtype: torch.dtype = torch.float32
    ):
        super(ContinuousFourierTokenizer, self).__init__()
        
        self.n_cont = int(n_cont)
        self.emb_dim = int(emb_dim)

        self.scalar_emb = FourierScalarEmbedding(
            num_bands=num_bands,
            max_freq_log2=max_freq_log2,
            include_raw=include_raw,
            base_freq=base_freq,
            dtype=dtype
        )

        self.proj = nn.Sequential(
            nn.Linear(self.scalar_emb.out_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, emb_dim)
        )

        self.var_id_emb = nn.Parameter(torch.randn(self.n_cont, emb_dim, dtype=dtype))
    
    def forward(self, x: Tensor):
        """
            x: (B, n_cont)
            return: (B, n_cont, D)
        """
        tokens = self.scalar_emb(x)
        tokens = self.proj(tokens) # (B, n_cont, D)
        tokens = tokens + self.var_id_emb
        return tokens # (B, n_cont, D)

class DetectorTokenizer(nn.Module):
    def __init__(
        self,
        coords: Tensor,
        emb_dim: int,
        mlp_hidden_dim: int,
        num_rz_bands: int = 4,
        max_freq_log2_rz: float = 4.0,
        num_phi_harmonics: int = 4,
        base_freq: float = math.pi,
        dtype: torch.dtype = torch.float32
    ):
        super(DetectorTokenizer, self).__init__()

        self.num_detectors = int(coords.shape[0])
        self.emb_dim = int(emb_dim)
        self.num_rz_bands = int(num_rz_bands)
        self.num_phi_harmonics = int(num_phi_harmonics)
        self.max_freq_log2_rz = float(max_freq_log2_rz)
        self.base_freq = float(base_freq)

        coords = coords.to(dtype)
        self.register_buffer("coords", coords, persistent=True)

        exponents = torch.linspace(0.0, self.max_freq_log2_rz, steps=self.num_rz_bands, dtype=dtype)
        freqs_rz = self.base_freq * torch.pow(2.0, exponents) # (num_rz_bands,)
        self.register_buffer("freqs_rz", freqs_rz, persistent=True)

        geom_features = self._build_geometry_features(coords, freqs_rz)
        self.register_buffer("geom_features", geom_features, persistent=True)

        self.proj = nn.Sequential(
            nn.Linear(self.geom_features.shape[1], mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, emb_dim)
        )

        self.det_id_emb = nn.Embedding(self.num_detectors, emb_dim)

    @torch.no_grad()
    def _build_geometry_features(
        self,
        coords: Tensor,
        freqs_rz: Tensor
    ) -> Tensor:
        r = coords[:, 0]
        phi = coords[:, 1]
        z = coords[:, 2]

        parts = [
            r.unsqueeze(-1),
            z.unsqueeze(-1)
        ]

        m = torch.arange(
            1, self.num_phi_harmonics + 1,
            dtype=coords.dtype,
            device=coords.device
        )
        angles = phi.unsqueeze(-1) * m.unsqueeze(0)
        parts.append(torch.sin(angles))
        parts.append(torch.cos(angles))

        r_angles = r.unsqueeze(-1) * freqs_rz.unsqueeze(0)
        z_angles = z.unsqueeze(-1) * freqs_rz.unsqueeze(0)

        parts.append(torch.sin(r_angles))
        parts.append(torch.cos(r_angles))
        parts.append(torch.sin(z_angles))
        parts.append(torch.cos(z_angles))

        return torch.cat(parts, dim=-1)

    @property
    def geom_feature_dim(self) -> int:
        return int(self.geom_features.shape[1])
    
    def forward(self, x: Tensor):
        if x.dtype != torch.long:
            x = x.to(dtype=torch.long)

        tokens = self.geom_features[x]
        tokens = self.proj(tokens)
        tokens = tokens + self.det_id_emb(x)

        return tokens # (B, D)
