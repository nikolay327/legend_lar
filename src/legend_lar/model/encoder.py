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
        device
    ):
        super(LArEncoder, self).__init__()

        self.config = config
        self.device = device
        self.cls_placeholder_id = self.config.sipm_cls_placeholder_id

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
        max_seqlen: int,
        geom_tokenizer: GeometryTokenizer
    ) -> Tensor:
        device = self.device
        dtype = self.cls_token.dtype

        cls_mask = (s_idx == self.cls_placeholder_id) # (N,)
        non_cls_mask = ~cls_mask

        non_cls_det_hits = s_idx[non_cls_mask]
        non_cls_time_hits = t_idx[non_cls_mask]

        # detector hits tokenizer
        r_tokens, phi_tokens, z_tokens = self.geometry_table(non_cls_det_hits)
        r_tokens, phi_tokens, z_tokens = geom_tokenizer(r_tokens, phi_tokens, z_tokens)

        # residual detectorwise-information
        det_emb = self.det_emb(non_cls_det_hits)
        r_tokens = r_tokens + det_emb
        phi_tokens = phi_tokens + det_emb
        z_tokens = z_tokens + det_emb

        # time information
        r_tokens = self.time_emb(r_tokens, non_cls_time_hits)
        phi_tokens = self.time_emb(phi_tokens, non_cls_time_hits)
        z_tokens = self.time_emb(z_tokens, non_cls_time_hits)

        # compute the new packed layout
        B = cu_seqlens.numel() - 1
        orig_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long) # (B,)
        n_hits_per_seq = orig_lens - 1
        new_lens = 1 + 3 * n_hits_per_seq

        new_cu_seqlens = torch.zeros(
            B + 1,
            dtype=cu_seqlens.dtype,
            device=device
        )
        new_cu_seqlens[1:] = torch.cumsum(new_lens, dim=0)

        new_max_seqlen = int(1 + 3 * (max_seqlen - 1))
        total_new_tokens = int(new_cu_seqlens[-1].item())

        # find the position of each hit token in the new packed layout
        token_indices = torch.arange(s_idx.shape[0], device=device, dtype=torch.long)
        seq_indices = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.long),
            orig_lens
        )
        
        # old hit and cls indices
        seq_starts_old = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), orig_lens)
        local_pos_old = token_indices - seq_starts_old # (N_hits + B,)
        hit_idx_in_seq = local_pos_old[non_cls_mask] - 1 # (N_hits,)

        # new sequence starts
        seq_starts_new = new_cu_seqlens[:-1].to(torch.long) # (B,) this is also the new CLS postions
        hit_seq_ids = seq_indices[non_cls_mask] # (N_hits,)

        triplet_base = seq_starts_new[hit_seq_ids] + 1 + 3 * hit_idx_in_seq # (N_hits,)

        # materialize the new packed token tensor
        tokens = torch.empty(
            total_new_tokens,
            self.config.hidden_size,
            device=device,
            dtype=dtype
        ) # (N_new, D)

        tokens[seq_starts_new] = self.cls_token
        tokens[triplet_base] = r_tokens
        tokens[triplet_base + 1] = phi_tokens
        tokens[triplet_base + 2] = z_tokens
        tokens = tokens.contiguous()

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=new_cu_seqlens,
                max_seqlen=new_max_seqlen
            )

        # CLS pooling: first token in each packed sequence
        cls_pos = cu_seqlens[:-1].to(torch.long) # (B,)
        tokens = self.norm(residual[cls_pos].float() + tokens[cls_pos].float()) # (B, D)
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

        self.geometry_table = DetectorGeometryFeatureTable(
            detector_coords=detector_coords,
            num_rz_bands=self.config.num_rz_bands,
            max_freq_log2_rz=self.config.max_freq_log2_rz,
            num_phi_harmonics=self.config.num_phi_harmonics
        )
        self.det_emb = nn.Embedding(self.config.num_hpges, self.config.hidden_size)
        self.partitioning_emb = nn.Embedding(self.config.hpge_global_partitioning_size, self.config.hidden_size)

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
        device = self.device
        dtype = self.cls_token.dtype

        cls_mask = f_idx == self.cls_placeholder_id
        gid_mask = f_idx == 0
        pid_mask = f_idx == 1
        feat_mask = ~(cls_mask | gid_mask | pid_mask)

        # detector hit tokenizer
        gid = f_vals[gid_mask].to(torch.long())
        r_tokens, phi_tokens, z_tokens = self.geometry_table(gid)
        r_tokens, phi_tokens, z_tokens = geom_tokenizer(r_tokens, phi_tokens, z_tokens)

        # residual detectorwise-information
        det_emb = self.det_emb(gid)
        r_tokens = r_tokens + det_emb
        phi_tokens = phi_tokens + det_emb
        z_tokens = z_tokens + det_emb

        # partitioning emb
        partitioning_emb = self.partitioning_emb(f_vals[pid_mask].to(torch.long))

        # feature tokens
        feat_emb = self.features_tokenizer(f_vals[feat_mask]) + self.features_id_emb(f_idx[feat_mask] - 2)

        # compute the new packed layout
        N_old = f_vals.shape[0]

        exp_sizes = torch.ones(N_old, dtype=torch.long, device=device)
        exp_sizes[gid_mask] = 3

        # global start position of each original token in the NEW packed layout
        start_pos = torch.cumsum(exp_sizes, dim=0) - exp_sizes  # (N_old,)

        total_new_tokens = int(exp_sizes.sum().item())

        # the start of each new sequence is the expanded start position of the corresponding OLD sequence start.
        orig_seq_starts = cu_seqlens[:-1].to(torch.long)  # (B,)
        new_cu_seqlens = torch.empty_like(cu_seqlens)
        new_cu_seqlens[:-1] = start_pos[orig_seq_starts].to(cu_seqlens.dtype)
        new_cu_seqlens[-1] = torch.tensor(total_new_tokens, dtype=cu_seqlens.dtype, device=device)

        new_max_seqlen = int(max_seqlen + 2)

        # materialize the new packed token tensor
        tokens = torch.empty(
            total_new_tokens,
            self.config.hidden_size,
            device=device,
            dtype=dtype
        ) # (N_new, D)

        # cls token
        tokens[start_pos[cls_mask]] = self.cls_token

        # gid -> 3 coordinate tokens
        gid_start = start_pos[gid_mask]
        tokens[gid_start] = r_tokens
        tokens[gid_start + 1] = phi_tokens
        tokens[gid_start + 2] = z_tokens

        # partitioning tokens
        tokens[start_pos[pid_mask]] = partitioning_emb

        # feature tokens
        tokens[start_pos[feat_mask]] = feat_emb

        tokens = tokens.contiguous()

        residual = None
        for block in self.blocks:
            tokens, residual = block(
                hidden_states=tokens,
                residual=residual,
                cu_seqlens=new_cu_seqlens,
                max_seqlen=new_max_seqlen
            )
        
        # CLS pooling: first token in each packed sequence
        cls_pos = new_cu_seqlens[:-1].to(torch.long) # (B,)
        tokens = self.norm(residual[cls_pos].float() + tokens[cls_pos].float()) # (B, D)
        tokens = self.proj_out(tokens)

        return tokens # (B, D)
