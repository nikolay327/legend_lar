
from torch import Tensor
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

class MHA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        causal: bool = False
    ):
        super(MHA, self).__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = self.emb_dim // self.num_heads
        self.causal = causal

        self.attn = FlashSelfAttention(causal=self.causal, attention_dropout=attn_dropout)
        self.Wqkv = nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim).contiguous()
        out = self.attn(qkv, causal=self.causal, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).reshape(-1, self.emb_dim)
        out = self.out_proj(out)
        return out
