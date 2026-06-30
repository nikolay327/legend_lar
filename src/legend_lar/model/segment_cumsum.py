from __future__ import annotations

import torch
import torch.nn as nn

from ..kernels import packed_segment_cumsum, SegmentCumsumConfig

class PackedSegmentCumsum(nn.Module):
    def __init__(self, config: SegmentCumsumConfig | None = None):
        super().__init__()
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        return packed_segment_cumsum(
            x,
            cu_seqlens,
            max_seq_len,
            config=self.config
        )
