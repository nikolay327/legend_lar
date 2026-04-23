import torch
import torch.nn as nn
from torch import Tensor

from legend_lar.utils import NRECConfig
from legend_lar.model.cls import create_block
from legend_lar.model.tokenizer import ContinuousFourierTokenizer, DetectorGeometryFeatureTable, GeometryTokenizer
from legend_lar.model.pos_embedding import SinPositionalEmbedding

class LArEncoder(nn.Module):
    def __init__(
        self,
        detector_coords: Tensor,
        config: NRECConfig,
        device = None
    ):
        super(LArEncoder, self).__init__()

        self.config = config
        self.cls_placeholder_id = self.config.cls_placeholder_id

        self.geometry_table = DetectorGeometryFeatureTable(
            detector_coords=detector_coords,
            num_rz_bands=self.config.num_rz_bands,
            max_freq_log2_rz=self.config.max_freq_log2_rz,
            num_phi_harmonics=self.config.num_phi_harmonics
        )
        self.det_emb = nn.Embedding(self.config.num_sipms, self.config.hidden_size)

        self.time_emb = SinPositionalEmbedding(
            emb_dim=self.config.hidden_size,
            max_len=self.config.num_sipm_t_bins,
            device=device
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
        max_seqlen: int,
        geom_tokenizer: GeometryTokenizer,
        **kwargs
    ) -> Tensor:
        device = self.cls_token.device
        dtype = self.cls_token.dtype

        cls_mask = (s_idx == self.cls_placeholder_id) # (N,)
        non_cls_mask = ~cls_mask

        non_cls_det_hits = s_idx[non_cls_mask]
        non_cls_time_hits = t_idx[non_cls_mask]

        # detector hits tokenizer
        r_tokens, phi_tokens, z_tokens = self.geometry_table(non_cls_det_hits)
        tokens = geom_tokenizer(r_tokens, phi_tokens, z_tokens)
        # residual detectorwise-information
        tokens = tokens + self.det_emb(non_cls_det_hits)
        # time information
        tokens = self.time_emb(tokens, non_cls_time_hits)

        new_tokens = torch.empty(
            s_idx.shape[0],
            self.config.hidden_size,
            device=device,
            dtype=dtype
        )

        # write cls tokens
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        cls_src = self.cls_token.unsqueeze(0).expand(cls_pos.numel(), -1)
        new_tokens.index_copy_(0, cls_pos, cls_src)

        # write the rest of the tokens
        non_cls_pos = non_cls_mask.nonzero(as_tuple=False).squeeze(-1)
        new_tokens.index_copy_(0, non_cls_pos, tokens)

        new_tokens = new_tokens.contiguous()

        residual = None
        for block in self.blocks:
            new_tokens, residual = block(
                hidden_states=new_tokens,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

        # CLS pooling: first token in each packed sequence
        new_tokens = self.norm(residual[cls_pos].float() + new_tokens[cls_pos].float()) # (B, D)
        new_tokens = self.proj_out(new_tokens)

        return new_tokens # (B, D)
    
class UnbinnedLArEncoder(nn.Module):
    def __init__(
        self,
        detector_coords: Tensor,
        config: NRECConfig,
        device = None
    ):
        super(UnbinnedLArEncoder, self).__init__()

        self.config = config
        self.cls_placeholder_id = self.config.cls_placeholder_id

        self.geometry_table = DetectorGeometryFeatureTable(
            detector_coords=detector_coords,
            num_rz_bands=self.config.num_rz_bands,
            max_freq_log2_rz=self.config.max_freq_log2_rz,
            num_phi_harmonics=self.config.num_phi_harmonics
        )
        self.det_emb = nn.Embedding(self.config.num_sipms, self.config.hidden_size)

        self.time_emb = SinPositionalEmbedding(
            emb_dim=self.config.hidden_size,
            max_len=self.config.num_sipm_t_bins,
            device=device
        )

        self.pe_emb = ContinuousFourierTokenizer(
            emb_dim=self.config.hidden_size,
            mlp_hidden_dim=self.config.intermediate_size,
            num_bands=self.config.sipm_num_feat_bands,
            max_freq_log2=self.config.sipm_feat_max_freq_log2
        )

        self.norm_in = nn.LayerNorm(self.config.hidden_size)
        self.proj_in = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size)
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
        v_val: Tensor, # (N,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int,
        geom_tokenizer: GeometryTokenizer
    ) -> Tensor:
        device = self.cls_token.device
        dtype = self.cls_token.dtype

        cls_mask = (s_idx == self.cls_placeholder_id) # (N,)
        non_cls_mask = ~cls_mask

        non_cls_det_hits = s_idx[non_cls_mask]
        non_cls_time_hits = t_idx[non_cls_mask]
        non_cls_num_pes = v_val[non_cls_mask]

        # detector hits tokenizer
        r_tokens, phi_tokens, z_tokens = self.geometry_table(non_cls_det_hits)
        tokens = geom_tokenizer(r_tokens, phi_tokens, z_tokens)
        # residual detectorwise-information
        tokens = tokens + self.det_emb(non_cls_det_hits)
        # time information
        tokens = self.time_emb(tokens, non_cls_time_hits)
        # num of pes
        tokens = tokens + self.pe_emb(non_cls_num_pes)

        new_tokens = torch.empty(
            s_idx.shape[0],
            self.config.hidden_size,
            device=device,
            dtype=dtype
        )

        # write cls tokens
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        cls_src = self.cls_token.unsqueeze(0).expand(cls_pos.numel(), -1)
        new_tokens.index_copy_(0, cls_pos, cls_src)

        # write the rest of the tokens
        non_cls_pos = non_cls_mask.nonzero(as_tuple=False).squeeze(-1)
        new_tokens.index_copy_(0, non_cls_pos, tokens)

        new_tokens = new_tokens.contiguous()
        new_tokens = self.proj_in(self.norm_in(new_tokens))

        residual = None
        for block in self.blocks:
            new_tokens, residual = block(
                hidden_states=new_tokens,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

        # CLS pooling: first token in each packed sequence
        new_tokens = self.norm(residual[cls_pos].float() + new_tokens[cls_pos].float()) # (B, D)
        new_tokens = self.proj_out(new_tokens)

        return new_tokens # (B, D)

class HPGeEncoder(nn.Module):
    def __init__(
        self,
        detector_coords: Tensor,
        config: NRECConfig,
        device = None
    ):
        super(HPGeEncoder, self).__init__()

        self.config = config
        self.cls_placeholder_id = self.config.cls_placeholder_id

        self.geometry_table = DetectorGeometryFeatureTable(
            detector_coords=detector_coords,
            num_rz_bands=self.config.num_rz_bands,
            max_freq_log2_rz=self.config.max_freq_log2_rz,
            num_phi_harmonics=self.config.num_phi_harmonics
        )
        self.det_emb = nn.Embedding(self.config.num_hpges, self.config.hidden_size)
        if self.config.subpartition_hpge_feats == 1:
            self.partitioning_emb = nn.Embedding(self.config.hpge_global_partitioning_size, self.config.hidden_size)
        else:
            self.partitioning_emb = None

        self.features_tokenizer = ContinuousFourierTokenizer(
            emb_dim=self.config.hidden_size,
            mlp_hidden_dim=self.config.intermediate_size,
            num_bands=self.config.hpge_num_feat_bands,
            max_freq_log2=self.config.hpge_feat_max_freq_log2
        )

        self.features_id_emb = nn.Embedding(self.config.hpge_num_features, self.config.hidden_size)
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
        max_seqlen: int,
        geom_tokenizer: GeometryTokenizer
    ) -> Tensor:
        device = self.cls_token.device
        dtype = self.cls_token.dtype

        cls_mask = f_idx == self.cls_placeholder_id
        gid_mask = f_idx == 0
        if self.config.subpartition_hpge_feats == 1:
            pid_mask = f_idx == 1
            feat_mask = ~(cls_mask | gid_mask | pid_mask)
            feats_start = 2

            partitioning_emb = self.partitioning_emb(f_vals[pid_mask].to(torch.long))
        else:
            feat_mask = ~(cls_mask | gid_mask)
            feats_start = 1
            partitioning_emb = None

        # detector hit tokenizer
        gid = f_vals[gid_mask].to(torch.long)
        r_tokens, phi_tokens, z_tokens = self.geometry_table(gid)
        geom_tokens = geom_tokenizer(r_tokens, phi_tokens, z_tokens)
        # residual detectorwise-information
        geom_tokens = geom_tokens + self.det_emb(gid)

        # feature tokens
        feat_emb = self.features_tokenizer(f_vals[feat_mask]) + self.features_id_emb(f_idx[feat_mask] - feats_start)

        tokens = torch.empty(
            f_idx.shape[0],
            self.config.hidden_size,
            device=device,
            dtype=dtype
        )

        # cls token
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        cls_src = self.cls_token.unsqueeze(0).expand(cls_pos.numel(), -1)
        tokens.index_copy_(0, cls_pos, cls_src)

        # gid token
        gid_pos = gid_mask.nonzero(as_tuple=False).squeeze(-1)
        tokens.index_copy_(0, gid_pos, geom_tokens)

        # partitioning tokens
        if self.config.subpartition_hpge_feats == 1:
            pid_pos = pid_mask.nonzero(as_tuple=False).squeeze(-1)
            tokens.index_copy_(0, pid_pos, partitioning_emb)

        # feature tokens
        feat_pos = feat_mask.nonzero(as_tuple=False).squeeze(-1)
        tokens.index_copy_(0, feat_pos, feat_emb)

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
        tokens = self.norm(residual[cls_pos].float() + tokens[cls_pos].float()) # (B, D)
        tokens = self.proj_out(tokens)

        return tokens # (B, D)

class CausalHPGeEncoder(nn.Module):
    def __init__(
        self,
        detector_coords: Tensor,
        config: NRECConfig,
        device = None
    ):
        super(CausalHPGeEncoder, self).__init__()

        self.config = config

        self.geometry_table = DetectorGeometryFeatureTable(
            detector_coords=detector_coords,
            num_rz_bands=self.config.num_rz_bands,
            max_freq_log2_rz=self.config.max_freq_log2_rz,
            num_phi_harmonics=self.config.num_phi_harmonics
        )
        self.det_emb = nn.Embedding(self.config.num_hpges, self.config.hidden_size)
        if self.config.subpartition_hpge_feats == 1:
            self.partitioning_emb = nn.Embedding(self.config.hpge_global_partitioning_size, self.config.hidden_size)
        else:
            self.partitioning_emb = None

        self.features_tokenizers = nn.ModuleList([
            ContinuousFourierTokenizer(
                emb_dim=self.config.hidden_size,
                mlp_hidden_dim=self.config.intermediate_size,
                num_bands=self.config.hpge_num_feat_bands,
                max_freq_log2=self.config.hpge_feat_max_freq_log2
            )
            for _ in range(self.config.hpge_num_features)
        ])

        self.norm_in = nn.LayerNorm(self.config.hidden_size)
        self.proj_in = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        )

        self.blocks = nn.ModuleList([
            create_block(self.config, True) for _ in range(self.config.hpge_num_layers)
        ])

        self.norm = nn.LayerNorm(self.config.hidden_size)

        self.proj_out = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        )

    def _tokenize_continuous_features(
        self,
        feat_vals: Tensor,
        feat_local_idx: Tensor,
        device,
        dtype
    ) -> Tensor:
        feat_emb = torch.empty(
            (feat_vals.shape[0], self.config.hidden_size),
            device=device,
            dtype=dtype
        )

        for j, tokenizer in enumerate(self.features_tokenizers):
            pos = (feat_local_idx == j).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue

            vals_j = feat_vals.index_select(0, pos)
            emb_j = tokenizer(vals_j)
            feat_emb.index_copy_(0, pos, emb_j)

        return feat_emb

    def forward(
        self,
        f_idx: Tensor, # (N_valid,)
        f_vals: Tensor, # (N_valid,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int,
        geom_tokenizer: GeometryTokenizer
    ) -> Tensor:
        device = self.det_emb.weight.device
        dtype = self.det_emb.weight.dtype

        gid_mask = f_idx == 0
        if self.config.subpartition_hpge_feats == 1:
            pid_mask = f_idx == 1
            feat_mask = ~(gid_mask | pid_mask)
            feats_start = 2

            partitioning_emb = self.partitioning_emb(f_vals[pid_mask].to(torch.long))
        else:
            feat_mask = ~gid_mask
            feats_start = 1
            partitioning_emb = None

        # detector hit tokenizer
        gid = f_vals[gid_mask].to(torch.long)
        r_tokens, phi_tokens, z_tokens = self.geometry_table(gid)
        geom_tokens = geom_tokenizer(r_tokens, phi_tokens, z_tokens)
        # residual detectorwise-information
        geom_tokens = geom_tokens + self.det_emb(gid)

        # feature tokens
        feat_vals = f_vals[feat_mask]
        feat_local_idx = f_idx[feat_mask] - feats_start

        feat_emb = self._tokenize_continuous_features(
            feat_vals=feat_vals,
            feat_local_idx=feat_local_idx,
            device=device,
            dtype=dtype
        )

        tokens = torch.empty(
            f_idx.shape[0],
            self.config.hidden_size,
            device=device,
            dtype=dtype
        )

        # gid token
        gid_pos = gid_mask.nonzero(as_tuple=False).squeeze(-1)
        tokens.index_copy_(0, gid_pos, geom_tokens)

        # partitioning tokens
        if self.config.subpartition_hpge_feats == 1:
            pid_pos = pid_mask.nonzero(as_tuple=False).squeeze(-1)
            tokens.index_copy_(0, pid_pos, partitioning_emb)

        # feature tokens
        feat_pos = feat_mask.nonzero(as_tuple=False).squeeze(-1)
        tokens.index_copy_(0, feat_pos, feat_emb)

        tokens = tokens.contiguous()
        tokens = self.proj_in(self.norm_in(tokens))

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        tokens = self.norm(residual.float() + tokens.float())
        tokens = self.proj_out(tokens)

        if self.config.deep_supervision == 0:
            last_pos = cu_seqlens[1:].to(torch.long) - 1 # (B,)
            tokens = tokens.index_select(0, last_pos) # (B, D)

        return tokens
