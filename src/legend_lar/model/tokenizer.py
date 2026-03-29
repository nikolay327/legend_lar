import math
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn

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
        emb_dim: int,
        mlp_hidden_dim: int,
        num_bands: int = 16,
        max_freq_log2: float = 8.0,
        include_raw: bool = True,
        base_freq: float = math.pi,
        dtype: torch.dtype = torch.float32
    ):
        super(ContinuousFourierTokenizer, self).__init__()
        
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
    
    def forward(self, x: Tensor):
        """
            x: (B, )
            return: (B, D)
        """
        tokens = self.scalar_emb(x)
        tokens = self.proj(tokens) # (B, D)
        return tokens # (B, D)
    
class DetectorGeometryFeatureTable(nn.Module):
    """
        Fixed geometry-feature lookup table for detector IDs using cylindrical coordinates (r, phi, z).
        det_ids -> (r_feat, phi_feat, z_feat)
    """
    def __init__(
        self,
        detector_coords: Tensor, # (num_detectors, 3): [r, phi, z]
        num_rz_bands: int = 4,
        max_freq_log2_rz: float = 4.0,
        num_phi_harmonics: int = 4,
        base_freq: float = math.pi,
        include_raw_rz: bool = True
    ):
        super(DetectorGeometryFeatureTable, self).__init__()

        self.num_detectors = int(detector_coords.shape[0])
        self.num_rz_bands = int(num_rz_bands)
        self.num_phi_harmonics = int(num_phi_harmonics)
        self.max_freq_log2_rz = float(max_freq_log2_rz)
        self.base_freq = float(base_freq)
        self.include_raw_rz = bool(include_raw_rz)

        coords = detector_coords.to(torch.float32)
        self.register_buffer("detector_coords", coords, persistent=True)

        # Frequencies for r and z Fourier features
        exponents = torch.linspace(0.0, self.max_freq_log2_rz, steps=self.num_rz_bands, dtype=torch.float32)
        freqs_rz = self.base_freq * torch.pow(2.0, exponents)
        self.register_buffer("freqs_rz", freqs_rz, persistent=True)

        # Precompute geometry feature tables
        r_feat, phi_feat, z_feat = self._build_geometry_features(coords, freqs_rz)
        self.register_buffer("r_features", r_feat, persistent=True)
        self.register_buffer("phi_features", phi_feat, persistent=True)
        self.register_buffer("z_features", z_feat, persistent=True)
    
    def _build_geometry_features(
        self,
        coords: Tensor, # (N, 3)
        freqs_rz: Tensor # (K,)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build detector geometry features:

            r-token:
                [r, sin(w_k r), cos(w_k r)] if include_raw_rz==True
                [sin(w_k r), cos(w_k r)] otherwise

            phi-token:
                [sin(m phi), cos(m phi)]

            z-token:
                [z, sin(w_k z), cos(w_k z)] if include_raw_rz==True
                [sin(w_k z), cos(w_k z)] otherwise
        """
        r = coords[:, 0] # (N,)
        phi = coords[:, 1] # (N,)
        z = coords[:, 2] # (N,)

        # r features
        r_parts = []
        if self.include_raw_rz:
            r_parts.append(r.unsqueeze(-1))
        r_angles = r.unsqueeze(-1) * freqs_rz.unsqueeze(0) # (N, K)
        r_parts.append(torch.sin(r_angles))
        r_parts.append(torch.cos(r_angles))
        r_feat = torch.cat(r_parts, dim=-1)

        # phi features
        phi_parts = []
        m = torch.arange(
            1, self.num_phi_harmonics + 1,
            dtype=coords.dtype,
            device=coords.device
        ) # (M,)
        angles = phi.unsqueeze(-1) * m.unsqueeze(0) # (N, M)
        phi_parts.append(torch.sin(angles))
        phi_parts.append(torch.cos(angles))
        phi_feat = torch.cat(phi_parts, dim=-1)

        # z features
        z_parts = []
        if self.include_raw_rz:
            z_parts.append(z.unsqueeze(-1))
        z_angles = z.unsqueeze(-1) * freqs_rz.unsqueeze(0) # (N, K)
        z_parts.append(torch.sin(z_angles))
        z_parts.append(torch.cos(z_angles))
        z_feat = torch.cat(z_parts, dim=-1)
    
        return r_feat, phi_feat, z_feat

    @property
    def r_feature_dim(self) -> int:
        return int(self.r_features.shape[1])

    @property
    def phi_feature_dim(self) -> int:
        return int(self.phi_features.shape[1])

    @property
    def z_feature_dim(self) -> int:
        return int(self.z_features.shape[1])

    def forward(self, det_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            det_ids: LongTensor of shape (B,) or (B, T)

            returns:
                r_feat: (..., N_feat_r)
                phi_feat: (..., N_feat_phi)
                z_feat: (..., N_feat_z)
        """
        if det_ids.dtype != torch.long:
            det_ids = det_ids.to(torch.long)

        r_feat = self.r_features[det_ids]
        phi_feat = self.phi_features[det_ids]
        z_feat = self.z_features[det_ids]

        return r_feat, phi_feat, z_feat

class GeometryTokenizer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidded_size: int,
        r_dim: int,
        phi_dim: int,
        z_dim: int
    ):
        super(GeometryTokenizer, self).__init__()

        self.r_proj = nn.Sequential(
            nn.Linear(r_dim, hidded_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidded_size, emb_dim)
        )
        self.phi_proj = nn.Sequential(
            nn.Linear(phi_dim, hidded_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidded_size, emb_dim)
        )
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, hidded_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidded_size, emb_dim)
        )

    def forward(self, r_feat: Tensor, phi_feat: Tensor, z_feat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            return:
                r_feat: (..., D)
                phi_feat: (..., D)
                z_feat: (..., D)
        """
        r_feat = self.r_proj(r_feat)
        phi_feat = self.phi_proj(phi_feat)
        z_feat = self.z_proj(z_feat)

        return r_feat, phi_feat, z_feat
