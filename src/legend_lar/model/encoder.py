from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from legend_lar.utils import NRECConfig
from legend_lar.model.cls import create_unconditional_block
from legend_lar.model.tokenizer import DetectorTokenizer, ContinuousFourierTokenizer
from legend_lar.model.pos_embedding import SinPositionalEmbedding

class LArEncoder(nn.Module):
    def __init__(
        self,
        detector_coords: Tensor,
        config: NRECConfig,
        device
    ):
        super(LArEncoder, self).__init__()

        self.config = config
        self.device = device
        self.cls_placeholder_id = self.config.sipm_cls_placeholder_id

        self.tokenizer = DetectorTokenizer(
            coords=detector_coords,
            emb_dim=self.config.hidden_size,
            mlp_hidden_dim=self.config.intermediate_size,
            num_rz_bands=self.config.sipm_num_rz_bands,
            max_freq_log2_rz=self.config.sipm_max_freq_log2_rz,
            num_phi_harmonics=self.config.sipm_num_phi_harmonics
        )

        self.time_emb = SinPositionalEmbedding(
            emb_dim=self.config.hidden_size,
            max_len=self.config.num_sipm_t_bins,
            device=self.device
        )

        self.cls_token = nn.Parameter(torch.empty(self.config.hidden_size))

        self.blocks = nn.ModuleList([
            create_unconditional_block(self.config) for _ in range(self.config.sipm_num_layers)
        ])

    def forward(
        self,
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (N+1,)
        max_seqlen: int
    ) -> Tensor:
        cls_mask = (s_idx == self.cls_placeholder_id) # (N,)
        non_cls_mask = ~cls_mask

        tokens_raw = torch.empty(
            s_idx.size(0),
            self.config.hidden_size,
            device=s_idx.device,
            dtype=self.cls_token.dtype
        ) # (N, D)

        if non_cls_mask.any():
            tokens_raw[non_cls_mask] = self.tokenizer(s_idx[non_cls_mask])

        if cls_mask.any():
            tokens_raw[cls_mask] = self.cls_token.unsqueeze(0).expand(cls_mask.sum(), -1)

        tokens = self.time_emb(tokens_raw, t_idx)

        if cls_mask.any():
            tokens = tokens.clone()
            tokens[cls_mask] = tokens_raw[cls_mask]
        
        tokens = tokens.contiguous()

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        tokens = residual + tokens # (N, D)

        # CLS pooling: first token in each packed sequence
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        pooled = tokens[cls_pos]

        return pooled # (B, D)

class HPGeEncoder(nn.Module):
    def __init__(
        self,
        detector_coords: Tensor,
        config: NRECConfig,
        device
    ):
        super(HPGeEncoder, self).__init__()

        self.config = config
        self.device = device

        self.detector_tokenizer = DetectorTokenizer(
            coords=detector_coords,
            emb_dim=self.config.hidden_size,
            mlp_hidden_dim=self.config.intermediate_size,
            num_rz_bands=self.config.hpge_num_rz_bands,
            max_freq_log2_rz=self.config.hpge_max_freq_log2_rz,
            num_phi_harmonics=self.config.hpge_num_phi_harmonics
        )

        self.partitioning_tokenizer = nn.Embedding(self.config.hpge_global_partitioning_size, self.config.hidden_size)

        self.features_tokenizer = ContinuousFourierTokenizer(
            n_cont=self.config.hpge_num_features,
            emb_dim=self.config.hidden_size,
            mlp_hidden_dim=self.config.intermediate_size,
            num_bands=self.config.hpge_num_feat_bands,
            max_freq_log2=self.config.hpge_feat_max_freq_log2
        )

        self.cls_token = nn.Parameter(torch.empty(self.config.hidden_size))

        self.blocks = nn.ModuleList([
            create_unconditional_block(self.config) for _ in range(self.config.hpge_num_layers)
        ])

    def forward(self, gid: Tensor, pid: Tensor, features: Tensor):
        """
            gid: (B,)
            pid: (B,)
            features: (B, N_feats)

            return: (B, D)
        """
        tokens = torch.empty(
            gid.size(0),
            self.config.hpge_num_features + 3, # num_feats + 1 cls + 1 det + 1 partitioning
            self.config.hidden_size,
            device=gid.device,
            dtype=self.cls_token.dtype
        ) # (N, D)
        tokens[:, 0] = self.cls_token
        tokens[:, 1] = self.detector_tokenizer(gid)
        tokens[:, 2] = self.partitioning_tokenizer(pid)
        tokens[:, 3:] = self.features_tokenizer(features)
        tokens = tokens.contiguous()

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=None,
                max_seqlen=None
            )

        pooled = tokens[:, 0, :] + residual[:, 0, :]

        return pooled # (B, D)
