import torch
import torch.nn as nn
from torch import Tensor

from legend_lar.utils import ModelConfig
from legend_lar.model.cls import create_block
from legend_lar.model.tokenizer import DiscreteEmbedder, ContinuousEmbedder
from legend_lar.model.pos_embedding import SinPositionalEmbedding
from legend_lar.utils import pack_data

class NRatioEstimator(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device
    ):
        super(NRatioEstimator, self).__init__()

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
            create_block(self.config) for _ in range(self.config.num_layers)
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
    ):
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
        pooled = x.new_zeros((self.config.local_batch_size, self.config.hidden_size)) # (B, D) zero-tensor to accumulate the sum
        pooled.index_add_(0, b_idx, x) # For each i, x[i] is added into pooled[b_idx[i]]
        num_pe = lengths.to(x.dtype).clamp_min(1).unsqueeze(1)  # (B,1) the total number of pe in a batch entry
        pooled = pooled / num_pe # (B, D)

        return self.Wlogit(pooled)

    def tokenize_then_forward(self, x: Tensor, gE: Tensor):
        g, E, b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths = pack_data(x, gE, zero_token_id=self.config.num_sipms)
        return self.forward(g.to(torch.long), E.to(torch.float32), b_all.to(torch.long), t_all.to(torch.long), k_all.to(torch.long), cu_seqlens.to(torch.int32), max_seqlen, lengths.to(torch.long))
