from functools import partial

from legend_lar.model.mha import MHA
from legend_lar.model.mlp import MLP
from legend_lar.model.block import Block
from legend_lar.utils.configs import NRECConfig

def _create_mha_cls(num_attention_heads: int, attn_dropout: float, causal: bool):
    return partial(
        MHA,
        num_heads=num_attention_heads,
        attn_dropout=attn_dropout,
        causal=causal
    )

def _create_mlp_cls(intermediate_size: int):
    return partial(
        MLP,
        hidden_dim=intermediate_size
    )

def create_block(config: NRECConfig, causal: bool = False):
    mixer_cls = _create_mha_cls(
        num_attention_heads=config.num_attention_heads,
        attn_dropout=0.0 if config.attn_dropout is None else config.attn_dropout,
        causal=causal
    )
    mlp_cls = _create_mlp_cls(config.intermediate_size)

    return Block(
        emb_dim=config.hidden_size,
        mixer_cls=mixer_cls,
        mlp_cls=mlp_cls,
        resid_dropout1=config.block_resid_dropout1 if config.block_resid_dropout1 is not None else 0.0,
        resid_dropout2=config.block_resid_dropout2 if config.block_resid_dropout2 is not None else 0.0
    )
