from typing import Tuple

from torch import Tensor
import torch.nn as nn
from legend_lar.model.layers import AdaLN

class ConditionalBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        mixer_cls,
        mlp_cls,
        norm_modulator_cls,
        resid_dropout1,
        resid_dropout2
    ):
        super(ConditionalBlock, self).__init__()
        
        self.mixer = mixer_cls(emb_dim)
        self.mlp = mlp_cls(emb_dim)

        self.norm_modulator = norm_modulator_cls(emb_dim)
        self.norm_attn = AdaLN(emb_dim)
        self.norm_mlp = AdaLN(emb_dim)
        
        self.dropout1 = nn.Dropout(resid_dropout1)
        self.dropout2 = nn.Dropout(resid_dropout2)

        self.cached_mod_g = None # used for inference, if needed

    def set_cached_mod_g(self, e_g: Tensor):
        self.cached_mod_g = self.norm_modulator.cache_mod_g(e_g)
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        g2_: Tensor,
        e_g: Tensor,
        e_E: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # AdaLN modulator
        (s1, b1, g1), (s2, b2, g2) = self.norm_modulator(e_g, e_E, self.cached_mod_g)

        # prenorm path
        dropped = self.dropout1(hidden_states)
        residual = (residual + g2_ * dropped) if residual is not None else dropped

        hidden_states = self.norm_attn(residual, s1, b1)
        hidden_states = self.mixer(hidden_states, cu_seqlens, max_seqlen)
        residual = residual + g1 * self.dropout2(hidden_states)

        hidden_states = self.norm_mlp(residual, s2, b2)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, g2

class UnconditionalBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        mixer_cls,
        mlp_cls,
        resid_dropout1,
        resid_dropout2
    ):
        super(UnconditionalBlock, self).__init__()
        
        self.mixer = mixer_cls(emb_dim)
        self.mlp = mlp_cls(emb_dim)

        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim)
        
        self.dropout1 = nn.Dropout(resid_dropout1)
        self.dropout2 = nn.Dropout(resid_dropout2)
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # prenorm path
        dropped = self.dropout1(hidden_states)
        residual = (residual + dropped) if residual is not None else dropped

        hidden_states = self.norm_attn(residual)
        hidden_states = self.mixer(hidden_states, cu_seqlens, max_seqlen)
        residual = residual + self.dropout2(hidden_states)

        hidden_states = self.norm_mlp(residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
