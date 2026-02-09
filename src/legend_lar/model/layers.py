from typing import Optional, Tuple

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ResidualAdaLNModulator(nn.Module):
    """
    Produces residual-AdaLN parameters, with:
      - per-channel scale/shift (vectors length D)
      - scalar residual gates (one per branch)
      - separable conditioning: mod = mod_E(e_E) + mod_g(e_g) for energy and HPGe conditioning

    Recommended usage:
      params = mod(e_E, e_g) or params = mod(e_E, e_g, cached_mod_g=...)
      x = (1+params.s1)*LN(h) + params.b1
      h = h + params.g1 * Attn(x)
      ...
    """
    def __init__(self, emb_dim: int, gate_tanh_scale: float = 0., zero_init = True):
        super(ResidualAdaLNModulator, self).__init__()

        self.emb_dim = emb_dim # D
        self.gate_tanh_scale = gate_tanh_scale

        # Each branch (attn and MLP) has D scales, D shifts, and 1 gate
        out_dim = 4 * self.emb_dim + 2
        self.to_mod_E = nn.Linear(self.emb_dim, out_dim)
        self.to_mod_g = nn.Linear(self.emb_dim, out_dim)

        if zero_init:
            nn.init.zeros_(self.to_mod_E.weight)
            nn.init.zeros_(self.to_mod_E.bias)
            nn.init.zeros_(self.to_mod_g.weight)
            nn.init.zeros_(self.to_mod_g.bias)
    
    def cache_mod_g(self, e_g: Tensor):
        return self.to_mod_g(e_g) # intended out: (1, 4 * D + 2)

    def _gate(self, gate_raw: Tensor):
        if self.gate_tanh_scale > 0.:
            gate = 1.0 + self.gate_tanh_scale * F.tanh(gate_raw)
        else:
            gate = 1.0 + gate_raw
        return gate # (B, 1)

    def _split(self, mod: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # Attn branch modulation parameters
        s1 = mod[:, :self.emb_dim]
        b1 = mod[:, self.emb_dim:2*self.emb_dim]
        g1_raw = mod[:, 2*self.emb_dim:2*self.emb_dim+1]

        # MLP branch modulation parameters
        base = 2*self.emb_dim + 1
        s2 = mod[:, base:base+self.emb_dim]
        b2 = mod[:, base+self.emb_dim:base+2*self.emb_dim]
        g2_raw = mod[:, base+2*self.emb_dim:base+2*self.emb_dim+1]

        return (s1, b1, g1_raw), (s2, b2, g2_raw)

    def forward(
        self,
        e_g: Tensor,
        e_E: Tensor,
        cached_mod_g: Optional[Tensor] = None
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        mod_g = self.to_mod_g(e_g) if cached_mod_g is None else cached_mod_g
        mod_E = self.to_mod_E(e_E)
        mod = mod_g + mod_E

        (s1, b1, g1_raw), (s2, b2, g2_raw) = self._split(mod)

        # s1 scale for attn LN output
        # b1 shift for attn LN output
        # s2 scale for mlp LN output
        # b2 scale for mlp LN output

        # Scalar gates to (B, 1, 1)
        g1 = self._gate(g1_raw) # scalar gate for attn residual branch
        g2 = self._gate(g2_raw) # scalar gate for mlp residual branch

        return (s1, b1, g1), (s2, b2, g2)

class AdaLN(nn.LayerNorm):
    def __init__(self, emb_dim: int):
        super(AdaLN, self).__init__(emb_dim)

    def forward(self, x: Tensor, s: Tensor, b: Tensor) -> Tensor:
        return (1 + s) * super().forward(x) + b
