"""
    Custom implementation of a bidirectional/causal transformer block with fixed/variable sequence length input
    using optimized GPU kernels, flash_attn.flash_attn_qkvpacked_func and flash_attn.flash_attn_varlen_qkvpacked_func.
    See https://github.com/dao-ailab/flash-attention

    Input, weights, and biases are casted to bf16, while layer_norm computations are kept in fp32 for stability.

    This implementation is opaque to torch.compile, and thus fully compatible with it.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func

def _to_dtype(x: Tensor, dtype: torch.dtype):
    return x.to(dtype) if x.is_floating_point() and x.dtype != dtype else x

def _bf16_block_func(
    hidden_states: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,

    # norm_attn
    norm_attn_weight: Tensor,
    norm_attn_bias: Tensor,

    # mixer: Wqkv + out_proj
    Wqkv_weight: Tensor,
    Wqkv_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,

    # norm_mlp
    norm_mlp_weight: Tensor,
    norm_mlp_bias: Tensor,

    # mlp: fc1 + fc2
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,

    # metadata
    num_heads: int,
    head_dim: int,
    causal: bool,

    attn_dropout_p: float,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    deterministic: bool,

    resid_dropout1_p: float,
    resid_dropout2_p: float,
    training: bool,

    norm_attn_eps: float,
    norm_mlp_eps: float,
    is_varlen: bool

) -> Tuple[Tensor, Tensor]:
    block_dtype = torch.bfloat16

    hs = _to_dtype(hidden_states, block_dtype)
    res = _to_dtype(residual, block_dtype)

    na_w = _to_dtype(norm_attn_weight, block_dtype)
    na_b = _to_dtype(norm_attn_bias, block_dtype)

    wqkv_w = _to_dtype(Wqkv_weight, block_dtype)
    wqkv_b = _to_dtype(Wqkv_bias, block_dtype)
    out_w = _to_dtype(out_proj_weight, block_dtype)
    out_b = _to_dtype(out_proj_bias, block_dtype)

    nm_w = _to_dtype(norm_mlp_weight, block_dtype)
    nm_b = _to_dtype(norm_mlp_bias, block_dtype)

    fc1_w = _to_dtype(fc1_weight, block_dtype)
    fc1_b = _to_dtype(fc1_bias, block_dtype)
    fc2_w = _to_dtype(fc2_weight, block_dtype)
    fc2_b = _to_dtype(fc2_bias, block_dtype)

    dropped = F.dropout(hs, p=resid_dropout1_p, training=training)
    res1 = res + dropped

    # Layernorm in FP32
    x = F.layer_norm(
        input=res1.float(),
        normalized_shape=(res1.shape[-1],),
        weight=na_w.float(),
        bias=na_b.float(),
        eps=norm_attn_eps
    )
    x = _to_dtype(x, block_dtype)

    qkv = F.linear(x, wqkv_w, wqkv_b)
    qkv = qkv.view(*x.shape[:-1], 3, num_heads, head_dim).contiguous()

    attn_p = attn_dropout_p if training else 0.0

    if is_varlen:
        x = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            attn_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(window_left, window_right),
            deterministic=deterministic
        )

    else:
        x = flash_attn_qkvpacked_func(
            qkv,
            attn_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(window_left, window_right),
            deterministic=deterministic
        )
    
    x = x.reshape(*x.shape[:-2], num_heads * head_dim)
    x = F.linear(x, out_w, out_b)

    res2 = res1 + F.dropout(x, p=resid_dropout2_p, training=training)
    
    # Layernorm in FP32
    x = F.layer_norm(
        input=res2.float(),
        normalized_shape=(res2.shape[-1],),
        weight=nm_w.float(),
        bias=nm_b.float(),
        eps=norm_mlp_eps
    )
    x = _to_dtype(x, block_dtype)

    x = F.linear(x, fc1_w, fc1_b)
    x = F.gelu(x, approximate="tanh")
    out = F.linear(x, fc2_w, fc2_b)

    return out, res2

@torch.library.custom_op("larlib::bf16_block_forward", mutates_args=(), device_types="cuda")
def bf16_block_forward(
    hidden_states: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,

    # norm_attn
    norm_attn_weight: Tensor,
    norm_attn_bias: Tensor,

    # mixer: Wqkv + out_proj
    Wqkv_weight: Tensor,
    Wqkv_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,

    # norm_mlp
    norm_mlp_weight: Tensor,
    norm_mlp_bias: Tensor,

    # mlp: fc1 + fc2
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,

    # metadata
    num_heads: int,
    head_dim: int,
    causal: bool,

    attn_dropout_p: float,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    deterministic: bool,

    resid_dropout1_p: float,
    resid_dropout2_p: float,
    training: bool,

    norm_attn_eps: float,
    norm_mlp_eps: float,
    is_varlen: bool
) -> Tuple[Tensor, Tensor]:
    return _bf16_block_func(
        hidden_states,
        residual,
        cu_seqlens,
        max_seqlen,
        norm_attn_weight,
        norm_attn_bias,
        Wqkv_weight,
        Wqkv_bias,
        out_proj_weight,
        out_proj_bias,
        norm_mlp_weight,
        norm_mlp_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        num_heads,
        head_dim,
        causal,
        attn_dropout_p,
        softmax_scale,
        window_left,
        window_right,
        deterministic,
        resid_dropout1_p,
        resid_dropout2_p,
        training,
        norm_attn_eps,
        norm_mlp_eps,
        is_varlen,
    )

@bf16_block_forward.register_fake
def _(
    hidden_states: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,

    norm_attn_weight: Tensor,
    norm_attn_bias: Tensor,

    Wqkv_weight: Tensor,
    Wqkv_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,

    norm_mlp_weight: Tensor,
    norm_mlp_bias: Tensor,

    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,

    num_heads: int,
    head_dim: int,
    causal: bool,

    attn_dropout_p: float,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    deterministic: bool,

    resid_dropout1_p: float,
    resid_dropout2_p: float,
    training: bool,

    norm_attn_eps: float,
    norm_mlp_eps: float,
    is_varlen: bool
) -> Tuple[Tensor, Tensor]:
    out = hidden_states.new_empty(hidden_states.shape, dtype=torch.float32)
    res = hidden_states.new_empty(hidden_states.shape, dtype=torch.float32)
    return out, res

@torch.library.custom_op("larlib::bf16_block_backward", mutates_args=(), device_types="cuda")
def bf16_block_backward(
    grad_out: Tensor,
    grad_residual_out: Tensor,

    hidden_states: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,

    norm_attn_weight: Tensor,
    norm_attn_bias: Tensor,

    Wqkv_weight: Tensor,
    Wqkv_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,

    norm_mlp_weight: Tensor,
    norm_mlp_bias: Tensor,

    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,

    num_heads: int,
    head_dim: int,
    causal: bool,

    attn_dropout_p: float,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    deterministic: bool,

    resid_dropout1_p: float,
    resid_dropout2_p: float,
    training: bool,

    norm_attn_eps: float,
    norm_mlp_eps: float,
    is_varlen: bool
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor
]:
    with torch.enable_grad():
        hs = hidden_states.detach().requires_grad_(hidden_states.requires_grad)
        res = residual.detach().requires_grad_(residual.requires_grad)

        na_w = norm_attn_weight.detach(norm_attn_weight.requires_grad)
        na_b = norm_attn_bias.detach(norm_attn_bias.requires_grad)

        wqkv_w = Wqkv_weight.detach().requires_grad_(Wqkv_weight.requires_grad)
        wqkv_b = Wqkv_bias.detach().requires_grad_(Wqkv_bias.requires_grad)
        out_w = out_proj_weight.detach().requires_grad_(out_proj_weight.requires_grad)
        out_b = out_proj_bias.detach().requires_grad_(out_proj_bias.requires_grad)

        nm_w = norm_mlp_weight.detach().requires_grad_(norm_mlp_weight.requires_grad)
        nm_b = norm_mlp_bias.detach().requires_grad_(norm_mlp_bias.requires_grad)

        f1_w = fc1_weight.detach().requires_grad_(fc1_weight.requires_grad)
        f1_b = fc1_bias.detach().requires_grad_(fc1_bias.requires_grad)
        f2_w = fc2_weight.detach().requires_grad_(fc2_weight.requires_grad)
        f2_b = fc2_bias.detach().requires_grad_(fc2_bias.requires_grad)

        out, res_out = _bf16_block_func(
            hs,
            res,
            cu_seqlens,
            max_seqlen,
            na_w,
            na_b,
            wqkv_w,
            wqkv_b,
            out_w,
            out_b,
            nm_w,
            nm_b,
            f1_w,
            f1_b,
            f2_w,
            f2_b,
            num_heads,
            head_dim,
            causal,
            attn_dropout_p,
            softmax_scale,
            window_left,
            window_right,
            deterministic,
            resid_dropout1_p,
            resid_dropout2_p,
            training,
            norm_attn_eps,
            norm_mlp_eps,
            is_varlen
        )

        grads = torch.autograd.grad(
            outputs=(out, res_out),
            inputs=(
                hs,
                res,
                na_w,
                na_b,
                wqkv_w,
                wqkv_b,
                out_w,
                out_b,
                nm_w,
                nm_b,
                f1_w,
                f1_b,
                f2_w,
                f2_b
            ),
            grad_outputs=(grad_out, grad_residual_out),
            allow_unused=True,
            retain_graph=False,
            create_graph=False
        )

    refs = (
        hidden_states,
        residual,
        norm_attn_weight,
        norm_attn_bias,
        Wqkv_weight,
        Wqkv_bias,
        out_proj_weight,
        out_proj_bias,
        norm_mlp_weight,
        norm_mlp_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias
    )

    fixed = []
    for g, ref in zip(grads, refs):
        fixed.append(torch.zeros_like(ref) if g is None else g.to(dtype=ref.dtype))

    return tuple(fixed)

@bf16_block_backward.register_fake
def _(grad_out: Tensor,
    grad_residual_out: Tensor,

    hidden_states: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,

    norm_attn_weight: Tensor,
    norm_attn_bias: Tensor,

    Wqkv_weight: Tensor,
    Wqkv_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,

    norm_mlp_weight: Tensor,
    norm_mlp_bias: Tensor,

    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,

    num_heads: int,
    head_dim: int,
    causal: bool,

    attn_dropout_p: float,
    softmax_scale: float,
    window_left: int,
    window_right: int,
    deterministic: bool,

    resid_dropout1_p: float,
    resid_dropout2_p: float,
    training: bool,

    norm_attn_eps: float,
    norm_mlp_eps: float,
    is_varlen: bool
):
    return (
        torch.empty_like(hidden_states),
        torch.empty_like(residual),
        torch.empty_like(norm_attn_weight),
        torch.empty_like(norm_attn_bias),
        torch.empty_like(Wqkv_weight),
        torch.empty_like(Wqkv_bias),
        torch.empty_like(out_proj_weight),
        torch.empty_like(out_proj_bias),
        torch.empty_like(norm_mlp_weight),
        torch.empty_like(norm_mlp_bias),
        torch.empty_like(fc1_weight),
        torch.empty_like(fc1_bias),
        torch.empty_like(fc2_weight),
        torch.empty_like(fc2_bias)
    )

def _bf16_block_setup_context(ctx, inputs, output):
    (
        hidden_states,
        residual,
        cu_seqlens,
        max_seqlen,

        norm_attn_weight,
        norm_attn_bias,

        Wqkv_weight,
        Wqkv_bias,
        out_proj_weight,
        out_proj_bias,

        norm_mlp_weight,
        norm_mlp_bias,

        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,

        num_heads,
        head_dim,
        causal,

        attn_dropout_p,
        softmax_scale,
        window_left,
        window_right,
        deterministic,

        resid_dropout1_p,
        resid_dropout2_p,
        training,

        norm_attn_eps,
        norm_mlp_eps,
        is_varlen
    ) = inputs

    ctx.save_for_backward(
        hidden_states,
        residual,
        cu_seqlens,
        norm_attn_weight,
        norm_attn_bias,
        Wqkv_weight,
        Wqkv_bias,
        out_proj_weight,
        out_proj_bias,
        norm_mlp_weight,
        norm_mlp_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias
    )

    ctx.max_seqlen = max_seqlen
    ctx.num_heads = num_heads
    ctx.head_dim = head_dim
    ctx.causal = causal

    ctx.attn_dropout_p = attn_dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.window_left = window_left
    ctx.window_right = window_right
    ctx.deterministic = deterministic

    ctx.resid_dropout1_p = resid_dropout1_p
    ctx.resid_dropout2_p = resid_dropout2_p
    ctx.training = training

    ctx.norm_attn_eps = norm_attn_eps
    ctx.norm_mlp_eps = norm_mlp_eps
    ctx.is_varlen = is_varlen

def _bf16_block_backward_autograd(ctx, grad_out, grad_residual_out):
    (
        hidden_states,
        residual,
        cu_seqlens,
        norm_attn_weight,
        norm_attn_bias,
        Wqkv_weight,
        Wqkv_bias,
        out_proj_weight,
        out_proj_bias,
        norm_mlp_weight,
        norm_mlp_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias
    ) = ctx.saved_tensors

    grads = torch.ops.larlib.bf16_block_backward(
        grad_out,
        grad_residual_out,

        hidden_states,
        residual,
        cu_seqlens,
        ctx.max_seqlen,

        norm_attn_weight,
        norm_attn_bias,

        Wqkv_weight,
        Wqkv_bias,
        out_proj_weight,
        out_proj_bias,

        norm_mlp_weight,
        norm_mlp_bias,

        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,

        ctx.num_heads,
        ctx.head_dim,
        ctx.causal,

        ctx.attn_dropout_p,
        ctx.softmax_scale,
        ctx.window_left,
        ctx.window_right,
        ctx.deterministic,

        ctx.resid_dropout1_p,
        ctx.resid_dropout2_p,
        ctx.training,

        ctx.norm_attn_eps,
        ctx.norm_mlp_eps,
        ctx.is_varlen
    )

    (
        grad_hidden_states,
        grad_residual,
        grad_norm_attn_weight,
        grad_norm_attn_bias,
        grad_Wqkv_weight,
        grad_Wqkv_bias,
        grad_out_proj_weight,
        grad_out_proj_bias,
        grad_norm_mlp_weight,
        grad_norm_mlp_bias,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias
    ) = grads

    return (
        grad_hidden_states,
        grad_residual,
        None,
        None,

        grad_norm_attn_weight,
        grad_norm_attn_bias,

        grad_Wqkv_weight,
        grad_Wqkv_bias,
        grad_out_proj_weight,
        grad_out_proj_bias,

        grad_norm_mlp_weight,
        grad_norm_mlp_bias,

        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,

        None,
        None,
        None,

        None,
        None,
        None,
        None,
        None,

        None,
        None,
        None,

        None,
        None,
        None,
    )

bf16_block_forward.register_autograd(
    _bf16_block_backward_autograd,
    setup_context=_bf16_block_setup_context,
)


class Block(nn.Module):
    def __init__(
        self,
        emb_dim,
        mixer_cls,
        mlp_cls,
        resid_dropout1,
        resid_dropout2,
    ):
        super(Block, self).__init__()

        self.mixer = mixer_cls(emb_dim)
        self.mlp = mlp_cls(emb_dim)

        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(resid_dropout1)
        self.dropout2 = nn.Dropout(resid_dropout2)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None,
        cu_seqlens: Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[Tensor, Tensor]:
        residual_in = residual if residual is not None else torch.zeros_like(hidden_states)

        attn = self.mixer.attn
        softmax_scale = (
            attn.softmax_scale
            if attn.softmax_scale is not None
            else 1.0 / math.sqrt(self.mixer.head_dim)
        )

        is_varlen = cu_seqlens is not None
        if is_varlen:
            cu_seqlens_in = cu_seqlens
            max_seqlen_in = int(max_seqlen)
        else:
            cu_seqlens_in = torch.empty(0, device=hidden_states.device, dtype=torch.int32)
            max_seqlen_in = 0

        hidden_states, residual_out = torch.ops.larlib.bf16_block_forward(
            hidden_states,
            residual_in,
            cu_seqlens_in,
            max_seqlen_in,

            self.norm_attn.weight,
            self.norm_attn.bias,

            self.mixer.Wqkv.weight,
            self.mixer.Wqkv.bias,
            self.mixer.out_proj.weight,
            self.mixer.out_proj.bias,

            self.norm_mlp.weight,
            self.norm_mlp.bias,

            self.mlp.fc1.weight,
            self.mlp.fc1.bias,
            self.mlp.fc2.weight,
            self.mlp.fc2.bias,

            self.mixer.num_heads,
            self.mixer.head_dim,
            self.mixer.causal,

            attn.drop.p,
            softmax_scale,
            attn.window_size[0],
            attn.window_size[1],
            attn.deterministic,

            self.dropout1.p,
            self.dropout2.p,
            self.training,

            self.norm_attn.eps,
            self.norm_mlp.eps,
            is_varlen
        )

        return hidden_states, residual_out
