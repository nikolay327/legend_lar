import torch
import torch.nn as nn
from torch import Tensor

from legend_lar.utils import NRECConfig
from legend_lar.model.cls import create_block
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
            create_block(self.config) for _ in range(self.config.sipm_num_layers)
        ])

        self.norm = nn.LayerNorm(self.config.hidden_size)

        self.proj_out = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        )

    def forward(
        self,
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int
    ) -> Tensor:
        cls_mask = (s_idx == self.cls_placeholder_id) # (N,)
        non_cls_mask = ~cls_mask

        tokens = torch.empty(
            s_idx.size(0),
            self.config.hidden_size,
            device=s_idx.device,
            dtype=self.cls_token.dtype
        ) # (N, D)

        tokens[cls_mask] = self.cls_token
        if non_cls_mask.any():
            tokens[non_cls_mask] = self.time_emb(self.tokenizer(s_idx[non_cls_mask]), t_idx[non_cls_mask])

        tokens = tokens.contiguous()

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

        # CLS pooling: first token in each packed sequence
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        tokens = self.norm(residual[cls_pos] + tokens[cls_pos]) # (B, D)
        tokens = self.proj_out(tokens)

        return tokens # (B, D)

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
        self.cls_placeholder_id = self.config.hpge_cls_placeholder_id

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
            emb_dim=self.config.hidden_size,
            mlp_hidden_dim=self.config.intermediate_size,
            num_bands=self.config.hpge_num_feat_bands,
            max_freq_log2=self.config.hpge_feat_max_freq_log2
        )

        self.pos_emb = nn.Embedding(self.config.hpge_num_features, self.config.hidden_size)
        self.cls_token = nn.Parameter(torch.empty(self.config.hidden_size))

        self.blocks = nn.ModuleList([
            create_block(self.config) for _ in range(self.config.hpge_num_layers)
        ])

        self.norm = nn.LayerNorm(self.config.hidden_size)

        self.proj_out = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        )

    def forward(
        self,
        f_idx: Tensor, # (N_valid,)
        f_vals: Tensor, # (N_valid,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int
    ) -> Tensor:
        tokens = torch.empty(
            f_vals.size(0),
            self.config.hidden_size,
            device=f_vals.device,
            dtype=self.cls_token.dtype
        ) # (N_valid, D)

        cls_mask = f_idx == self.cls_placeholder_id
        gid_mask = f_idx == 0
        pid_mask = f_idx == 1
        feat_mask = ~(cls_mask | gid_mask | pid_mask)

        tokens[cls_mask] = self.cls_token

        if gid_mask.any():
            tokens[gid_mask] = self.detector_tokenizer(f_vals[gid_mask].to(torch.long))
        
        if pid_mask.any():
            tokens[pid_mask] = self.partitioning_tokenizer(f_vals[pid_mask].to(torch.long))
        
        if feat_mask.any():
            tokens[feat_mask] = self.features_tokenizer(f_vals[feat_mask]) + self.pos_emb(f_idx[feat_mask] - 2)
        
        tokens = tokens.contiguous()

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        
        # CLS pooling: first token in each packed sequence
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        tokens = self.norm(residual[cls_pos] + tokens[cls_pos]) # (B, D)
        tokens = self.proj_out(tokens)

        return tokens # (B, D)
