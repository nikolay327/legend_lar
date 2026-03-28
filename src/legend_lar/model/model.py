import torch.nn as nn
from torch import Tensor

from legend_lar.utils import NRECConfig
from legend_lar.model.encoder import LArEncoder, HPGeEncoder


class NREC(nn.Module):
    def __init__(
        self,
        lar_detector_coords: Tensor,
        hpge_detector_coords: Tensor,
        config: NRECConfig,
        device
    ):
        super(NREC, self).__init__()

        self.lar_encoder = LArEncoder(
            detector_coords=lar_detector_coords,
            config=config,
            device=device
        )

        self.hpge_encoder = HPGeEncoder(
            detector_coords=hpge_detector_coords,
            config=config,
            device=device
        )

    def forward(
        self,
        f_idx: Tensor, # (N_valid,)
        f_vals: Tensor, # (N_valid,)
        ge_cu_seqlens: Tensor, # (B/2+1,)
        ge_max_seqlen: int,
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int
    ):
        e_lar = self.lar_encoder(
            t_idx=t_idx,
            s_idx=s_idx,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        ) # (B, D)

        e_hpge = self.hpge_encoder(
            f_idx=f_idx,
            f_vals=f_vals,
            cu_seqlens=ge_cu_seqlens,
            max_seqlen=ge_max_seqlen
        ) # (B/2, D)

        return e_lar, e_hpge
