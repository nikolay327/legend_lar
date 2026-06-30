"""
Autograd integration for packed segmented cumulative sums.

Forward:
y = segmented_cumsum(x)

Backward:
dx = reverse_segmented_cumsum(dy)
"""

from __future__ import annotations

from typing import Optional

import torch

from .ops import (
    SegmentCumsumConfig,
    segment_cumsum_fwd,
    segment_cumsum_bwd,
)


class PackedSegmentCumsumFunction(torch.autograd.Function):
    """
    Differentiable packed segmented cumulative sum.

    Inputs:
    x: [total_tokens, D]
    cu_seqlens: [B + 1]
    max_seq_len: Python int
    config: SegmentCumsumConfig or None

    Output:
    y: [total_tokens, D]
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
        config: Optional[SegmentCumsumConfig],
    ) -> torch.Tensor:
        y = segment_cumsum_fwd(
            x,
            cu_seqlens,
            max_seq_len,
            config=config,
        )

        ctx.save_for_backward(cu_seqlens)
        ctx.max_seq_len = max_seq_len
        ctx.config = config

        return y

    @staticmethod
    def backward(
        ctx,
        grad_y: torch.Tensor,
    ):
        (cu_seqlens,) = ctx.saved_tensors

        if grad_y is None:
            grad_x = None
        else:
            grad_x = segment_cumsum_bwd(
                grad_y.contiguous(),
                cu_seqlens,
                ctx.max_seq_len,
                config=ctx.config,
            )

        return grad_x, None, None, None


def packed_segment_cumsum(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    *,
    config: Optional[SegmentCumsumConfig] = None,
) -> torch.Tensor:
    """
    Differentiable packed segmented cumulative sum.

    Inputs:
    x: [total_tokens, D]
    cu_seqlens: [B + 1]
    max_seq_len: Python int

    Output:
    y: [total_tokens, D]

    For each sequence b:
    y[start + t] = sum_{k=0}^{t} x[start + k]

    The backward computes:
    grad_x[start + t] = sum_{k=t}^{L - 1} grad_y[start + k]
    """
    return PackedSegmentCumsumFunction.apply(
        x,
        cu_seqlens,
        max_seq_len,
        config,
    )


__all__ = [
    "PackedSegmentCumsumFunction",
    "packed_segment_cumsum",
]