from functools import partial

from legend_lar.model.mha import MHA
from legend_lar.model.mlp import MLP
from legend_lar.model.block import Block
from legend_lar.model.layers import ResidualAdaLNModulator
from legend_lar.utils.configs import ModelConfig

def _create_mha_cls(num_attention_heads: int, causal: bool):
    return partial(
        MHA,
        num_heads=num_attention_heads,
        causal=causal
    )

def _create_mlp_cls(intermediate_size: int):
    return partial(
        MLP,
        hidden_dim=intermediate_size
    )

def _create_norm_modulator_cls(norm_gate_tanh_scale: float, norm_zero_init: bool):
    return partial(
        ResidualAdaLNModulator,
        gate_tanh_scale=norm_gate_tanh_scale,
        zero_init=norm_zero_init
    )

def create_block(config: ModelConfig):
    mixer_cls = _create_mha_cls(
        num_attention_heads=config.num_attention_heads,
        causal=False if config.causal is None else config.causal==1,
    )
    mlp_cls = _create_mlp_cls(config.intermediate_size)
    norm_modulator_cls = _create_norm_modulator_cls(
        norm_gate_tanh_scale=config.norm_gate_tanh_scale,
        norm_zero_init=config.norm_zero_init==1
    )

    return Block(
        emb_dim=config.hidden_size,
        mixer_cls=mixer_cls,
        mlp_cls=mlp_cls,
        norm_modulator_cls=norm_modulator_cls,
        resid_dropout1=config.block_resid_dropout1 if config.block_resid_dropout1 is not None else 0.0,
        resid_dropout2=config.block_resid_dropout2 if config.block_resid_dropout2 is not None else 0.0
    )
