import torch.nn as nn
from torch import Tensor

from legend_lar.utils import NRECConfig
from legend_lar.model.tokenizer import GeometryTokenizer
from legend_lar.model.encoder import LArEncoder, HPGeEncoder, CausalHPGeEncoder, UnbinnedLArEncoder


class NREC(nn.Module):
    def __init__(
        self,
        lar_detector_coords: Tensor,
        hpge_detector_coords: Tensor,
        config: NRECConfig,
        device
    ):
        super(NREC, self).__init__()

        if config.sipm_unbinned_pe == 1:
            self.lar_encoder = UnbinnedLArEncoder(
                detector_coords=lar_detector_coords,
                config=config,
                device=device
            )
        else:
            self.lar_encoder = LArEncoder(
                detector_coords=lar_detector_coords,
                config=config,
                device=device
            )

        if config.deep_supervision == 1:
            self.hpge_encoder = CausalHPGeEncoder(
                detector_coords=hpge_detector_coords,
                config=config,
                device=device
            )
        else:
            self.hpge_encoder = HPGeEncoder(
                detector_coords=hpge_detector_coords,
                config=config,
                device=device
            )

        self.geom_tokenizer = GeometryTokenizer(
            emb_dim=config.hidden_size,
            hidded_size=config.intermediate_size,
            r_dim=self.lar_encoder.geometry_table.r_feature_dim,
            phi_dim=self.lar_encoder.geometry_table.phi_feature_dim,
            z_dim=self.lar_encoder.geometry_table.z_feature_dim
        )

    def forward(
        self,
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        v_val: Tensor, # (N,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int,
        f_idx: Tensor = None, # (N_valid,)
        f_vals: Tensor = None, # (N_valid,)
        ge_cu_seqlens: Tensor = None, # (B/2+1,)
        ge_max_seqlen: int = None,
        pre_cumsum: bool = False
    ):
        e_lar = self.lar_encoder(
            t_idx=t_idx,
            s_idx=s_idx,
            v_val=v_val,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            geom_tokenizer=self.geom_tokenizer
        ) # (B, D)

        if ge_cu_seqlens is None:
            e_hpge = None
        else:
            e_hpge = self.hpge_encoder(
                f_idx=f_idx,
                f_vals=f_vals,
                cu_seqlens=ge_cu_seqlens,
                max_seqlen=ge_max_seqlen,
                geom_tokenizer=self.geom_tokenizer,
                pre_cumsum=pre_cumsum
            ) # (B/2, D)

        return e_lar, e_hpge
