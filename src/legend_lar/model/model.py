import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from legend_lar.utils import ModelConfig, BootstrappedKFoldConfig
from legend_lar.model.cls import create_conditional_block, create_unconditional_block
from legend_lar.model.tokenizer import DiscreteEmbedder, ContinuousEmbedder, JointHPGeEmbedder
from legend_lar.model.pos_embedding import SinPositionalEmbedding
from legend_lar.utils import pack_data

class BCERatioEstimator(nn.Module):
    def __init__(
        self,
        config: ModelConfig | BootstrappedKFoldConfig,
        device
    ):
        super(BCERatioEstimator, self).__init__()

        self.config = config
        self.device = device

        self.hpge_emb = DiscreteEmbedder(
            codebook_size=self.config.num_hpges,
            emb_dim=self.config.hidden_size
        )
        self.hpge_energy_emb = ContinuousEmbedder(
            emb_dim=self.config.hidden_size,
            hidden_dim=self.config.intermediate_size
        )
        self.sipm_emb = DiscreteEmbedder(
            codebook_size=self.config.num_sipms + 1, # +1 due to the last index = event with zero pe in total
            emb_dim=self.config.hidden_size
        )
        self.time_emb = SinPositionalEmbedding(
            emb_dim=self.config.hidden_size,
            max_len=self.config.num_sipm_t_bins,
            device=self.device
        )

        self.blocks = nn.ModuleList([
            create_conditional_block(self.config) for _ in range(self.config.num_layers)
        ])
        self.norm = nn.LayerNorm(self.config.hidden_size)

        self.Wlogit = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, 1)
        )

    def forward(
        self,
        g: Tensor, # (N,)
        E: Tensor, # (N,)
        b_idx: Tensor, # (N,)
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (N+1,)
        max_seqlen: int,
        lengths: Tensor # (N,)
    ) -> Tensor:
        e_g = self.hpge_emb(g) # (N, D)
        e_E = self.hpge_energy_emb(E) # (N, D)

        x = self.sipm_emb(s_idx) # (N, D)
        x = self.time_emb(x, t_idx)

        residual = None
        g2_ = None
        for block in self.blocks:
            x, residual, g2_ = block(
                hidden_states=x,
                residual=residual,
                g2_=g2_,
                e_g=e_g,
                e_E=e_E,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        x = self.norm(residual + g2_ * x) # (N, D)

        # Mean pooling, so that the pooled token is permutation-invariant within any multiset
        B = b_idx.max().detach().cpu().item() + 1
        pooled = x.new_zeros((B, self.config.hidden_size)) # (B, D) zero-tensor to accumulate the sum
        pooled.index_add_(0, b_idx, x) # For each i, x[i] is added into pooled[b_idx[i]]
        num_pe = lengths.to(x.dtype).clamp_min(1).unsqueeze(1)  # (B,1) the total number of pe in a batch entry
        pooled = pooled / num_pe # (B, D)

        return self.Wlogit(pooled)

    def tokenize_then_forward(self, x: Tensor, gE: Tensor):
        g, E, b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths = pack_data(x, gE, zero_token_id=self.config.num_sipms)
        return self.forward(g.to(torch.long), E.to(torch.float32), b_all.to(torch.long), t_all.to(torch.long), k_all.to(torch.long), cu_seqlens.to(torch.int32), max_seqlen, lengths.to(torch.long))

class UnconditionalRatioEstimator(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device
    ):
        super(UnconditionalRatioEstimator, self).__init__()

        self.config = config
        self.device = device

        self.sipm_emb = DiscreteEmbedder(
            codebook_size=self.config.num_sipms + 1, # +1 due to the last index = event with zero pe in total
            emb_dim=self.config.hidden_size
        )
        self.time_emb = SinPositionalEmbedding(
            emb_dim=self.config.hidden_size,
            max_len=self.config.num_sipm_t_bins,
            device=self.device
        )

        self.blocks = nn.ModuleList([
            create_unconditional_block(self.config) for _ in range(self.config.num_layers)
        ])
        self.norm = nn.LayerNorm(self.config.hidden_size)

        self.Wlogit = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.intermediate_size, 1)
        )

    def forward(
        self,
        b_idx: Tensor, # (N,)
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (N+1,)
        max_seqlen: int,
        lengths: Tensor # (N,)
    ) -> Tensor:
        x = self.sipm_emb(s_idx) # (N, D)
        x = self.time_emb(x, t_idx)

        residual = None
        for block in self.blocks:
            x, residual = block(
                hidden_states=x,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        x = self.norm(residual + x) # (N, D)

        # Mean pooling, so that the pooled token is permutation-invariant within any multiset
        B = int(b_idx.max().item()) + 1
        pooled = x.new_zeros((B, self.config.hidden_size)) # (B, D) zero-tensor to accumulate the sum
        pooled.index_add_(0, b_idx, x) # For each i, x[i] is added into pooled[b_idx[i]]
        num_pe = lengths.to(x.dtype).clamp_min(1).unsqueeze(1)  # (B,1) the total number of pe in a batch entry
        pooled = pooled / num_pe # (B, D)

        return self.Wlogit(pooled)

    def tokenize_then_forward(self, x: Tensor, gE: Tensor):
        _, _, b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths = pack_data(x, gE, zero_token_id=self.config.num_sipms)
        return self.forward(b_all.to(torch.long), t_all.to(torch.long), k_all.to(torch.long), cu_seqlens.to(torch.int32), max_seqlen, lengths.to(torch.long))

class ConditionalRatioEstimator(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device
    ):
        super(ConditionalRatioEstimator, self).__init__()

        self.config = config
        self.device = device

        self.joint_hpge_emb = JointHPGeEmbedder(
            emb_dim=config.hidden_size,
            hpge_codebook_size=config.num_hpges,
            hidden_dim=config.hidden_size
        )
        self.sipm_emb = DiscreteEmbedder(
            codebook_size=self.config.num_sipms + 1, # +1 due to the last index = event with zero pe in total
            emb_dim=self.config.hidden_size
        )
        self.time_emb = SinPositionalEmbedding(
            emb_dim=self.config.hidden_size,
            max_len=self.config.num_sipm_t_bins,
            device=self.device
        )

        self.blocks = nn.ModuleList([
            create_unconditional_block(self.config) for _ in range(self.config.num_layers)
        ])
        self.norm = nn.LayerNorm(self.config.hidden_size)

    def forward(
        self,
        g: Tensor, # (B,)
        E: Tensor, # (B,)
        b_idx: Tensor, # (N,)
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (N+1,)
        max_seqlen: int,
        lengths: Tensor, # (N,)
        e_hpge_: Tensor = None # (B, D)
    ) -> Tensor:
        if e_hpge_ is None:
            e_hpge = self.joint_hpge_emb(g, E) # (B, D)

        x = self.sipm_emb(s_idx) # (N, D)
        x = self.time_emb(x, t_idx)
        residual = None
        for block in self.blocks:
            x, residual = block(
                hidden_states=x,
                residual=residual,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        x = self.norm(residual + x) # (N, D)

        # Mean pooling, so that the pooled token is permutation-invariant within any multiset
        B = int(b_idx.max().item()) + 1
        anchors = x.new_zeros((B, self.config.hidden_size)) # (B, D) zero-tensor to accumulate the sum
        anchors.index_add_(0, b_idx, x) # For each i, x[i] is added into anchors[b_idx[i]]
        num_pe = lengths.to(x.dtype).clamp_min(1).unsqueeze(1) # (B,1) the total number of pe in a batch entry
        anchors = anchors / num_pe # (B, D)

        # Contrastive loss
        anchors = F.normalize(anchors, p=2, dim=-1)
        e_hpge = F.normalize(e_hpge, p=2, dim=-1) if e_hpge_ is None else e_hpge_
        logits = (anchors * e_hpge).sum(dim=-1, keepdim=True) / self.config.temperature # (B, 1)

        return logits, (anchors, e_hpge)

    def training_forward(
        self,
        g: Tensor, # (B,)
        E: Tensor, # (B,)
        b_idx: Tensor, # (N,)
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (N+1,)
        max_seqlen: int,
        lengths: Tensor # (N,)
    ):
        _, (anchors, e_hpge) = self.forward(
            g=g,
            E=E,
            b_idx=b_idx,
            t_idx=t_idx,
            s_idx=s_idx,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            lengths=lengths
        )
        logits = (anchors @ e_hpge.t()) / self.config.temperature # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device) # (B,)
        return logits, labels
