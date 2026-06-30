"""
Custom Triton kernels for packed segmented cumulative sums.

Packed layout:
X: [total_tokens, D]
Y: [total_tokens, D]
CU: [B + 1]

Sequence b occupies packed token indices:
CU[b] : CU[b + 1]

Forward:
Y[start + t] = sum_{k=0}^{t} X[start + k]

Backward:
DX[start + t] = sum_{k=t}^{L - 1} DY[start + k]
"""

import triton
import triton.language as tl

# Oneblock kernels

@triton.jit
def _segmented_cumsum_oneblock_fwd_kernel(
    X,
    Y,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Oneblock forward segmented cumulative sum.

    Grid:
    (B, ceil(D / BLOCK_D))

    Inputs:
    X: [total_tokens, D]
    CU: [B + 1]

    Output:
    Y: [total_tokens, D]

    Each Triton program handles one sequence and one block of embedding
    dimensions.

    Requirement:
    BLOCK_M >= max_seq_len for the current batch.

    Accumulation is performed in fp32. The store casts to Y's dtype.
    """
    b = tl.program_id(0)
    d_block = tl.program_id(1)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    d = d_block * BLOCK_D + offs_d

    ptrs = X + (start + offs_m[:, None]) * D + d[None, :]
    mask = (offs_m[:, None] < L) & (d[None, :] < D)

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    out = tl.cumsum(vals, axis=0)

    out_ptrs = Y + (start + offs_m[:, None]) * D + d[None, :]
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _segmented_cumsum_oneblock_bwd_kernel(
    DY,
    DX,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Oneblock backward segmented cumulative sum.

    Grid:
    (B, ceil(D / BLOCK_D))

    Inputs:
    DY: [total_tokens, D]
    CU: [B + 1]

    Output:
    DX: [total_tokens, D]

    Computes the reverse segmented cumulative sum:
    DX[start + t] = sum_{k=t}^{L - 1} DY[start + k]

    Requirement:
    BLOCK_M >= max_seq_len for the current batch.

    Accumulation is performed in fp32. The store casts to DX's dtype.
    """
    b = tl.program_id(0)
    d_block = tl.program_id(1)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    d = d_block * BLOCK_D + offs_d

    ptrs = DY + (start + offs_m[:, None]) * D + d[None, :]
    mask = (offs_m[:, None] < L) & (d[None, :] < D)

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    out = tl.cumsum(vals, axis=0, reverse=True)

    out_ptrs = DX + (start + offs_m[:, None]) * D + d[None, :]
    tl.store(out_ptrs, out, mask=mask)


# Hierarchical forward kernels

@triton.jit
def _segmented_cumsum_hier_fwd_local_kernel(
    X,
    LOCAL_Y,
    TILE_SUMS,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    """
    Hierarchical forward stage 1.

    Computes local cumulative sums within each tile and writes one tile sum
    per sequence tile.

    Grid:
    (B, MAX_TILES, ceil(D / BLOCK_D))

    Inputs:
    X: [total_tokens, D]
    CU: [B + 1]

    Outputs:
    LOCAL_Y: [total_tokens, D]
    TILE_SUMS: [B, MAX_TILES, D]

    Requirements:
    MAX_TILES >= ceil(max_seq_len / BLOCK_M)

    Accumulation is performed in fp32. Stores cast to the output tensor dtype.
    """
    b = tl.program_id(0)
    tile_id = tl.program_id(1)
    d_block = tl.program_id(2)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    tile_start = tile_id * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    token_offsets = tile_start + offs_m
    d = d_block * BLOCK_D + offs_d

    ptrs = X + (start + token_offsets[:, None]) * D + d[None, :]
    mask = (token_offsets[:, None] < L) & (d[None, :] < D)

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    local = tl.cumsum(vals, axis=0)

    out_ptrs = LOCAL_Y + (start + token_offsets[:, None]) * D + d[None, :]
    tl.store(out_ptrs, local, mask=mask)

    tile_sum = tl.sum(vals, axis=0)

    sum_ptrs = TILE_SUMS + (b * MAX_TILES + tile_id) * D + d
    tl.store(sum_ptrs, tile_sum, mask=d < D)


@triton.jit
def _segmented_cumsum_hier_fwd_carry_kernel(
    TILE_SUMS,
    TILE_CARRIES,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    """
    Hierarchical forward stage 2.

    Scans tile sums across tiles to produce exclusive tile carries.

    For tile t:
    TILE_CARRIES[t] = sum_{p < t} TILE_SUMS[p]

    Grid:
    (B, ceil(D / BLOCK_D))

    Inputs:
    TILE_SUMS: [B, MAX_TILES, D]
    CU: [B + 1]

    Output:
    TILE_CARRIES: [B, MAX_TILES, D]

    Requirements:
    BLOCK_TILES >= MAX_TILES

    Accumulation is performed in fp32. Stores cast to TILE_CARRIES' dtype.
    """
    b = tl.program_id(0)
    d_block = tl.program_id(1)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    num_tiles = tl.cdiv(L, BLOCK_M)

    offs_t = tl.arange(0, BLOCK_TILES)
    offs_d = tl.arange(0, BLOCK_D)

    d = d_block * BLOCK_D + offs_d

    ptrs = TILE_SUMS + (b * MAX_TILES + offs_t[:, None]) * D + d[None, :]
    mask = (
        (offs_t[:, None] < num_tiles)
        & (offs_t[:, None] < MAX_TILES)
        & (d[None, :] < D)
    )

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    inclusive = tl.cumsum(vals, axis=0)
    exclusive = inclusive - vals

    out_ptrs = TILE_CARRIES + (b * MAX_TILES + offs_t[:, None]) * D + d[None, :]
    store_mask = (offs_t[:, None] < MAX_TILES) & (d[None, :] < D)

    tl.store(out_ptrs, exclusive, mask=store_mask)


@triton.jit
def _segmented_cumsum_hier_fwd_add_kernel(
    LOCAL_Y,
    Y,
    TILE_CARRIES,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    """
    Hierarchical forward stage 3.

    Adds each tile's exclusive carry to the local cumulative sum.

    Grid:
    (B, MAX_TILES, ceil(D / BLOCK_D))

    Inputs:
    LOCAL_Y: [total_tokens, D]
    TILE_CARRIES: [B, MAX_TILES, D]
    CU: [B + 1]

    Output:
    Y: [total_tokens, D]

    LOCAL_Y and Y may alias if the wrapper wants in-place finalization.

    Accumulation is performed in fp32. Stores cast to Y's dtype.
    """
    b = tl.program_id(0)
    tile_id = tl.program_id(1)
    d_block = tl.program_id(2)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    tile_start = tile_id * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    token_offsets = tile_start + offs_m
    d = d_block * BLOCK_D + offs_d

    carry_ptrs = TILE_CARRIES + (b * MAX_TILES + tile_id) * D + d
    carry = tl.load(carry_ptrs, mask=d < D, other=0.0).to(tl.float32)

    local_ptrs = LOCAL_Y + (start + token_offsets[:, None]) * D + d[None, :]
    mask = (token_offsets[:, None] < L) & (d[None, :] < D)

    local = tl.load(local_ptrs, mask=mask, other=0.0).to(tl.float32)

    out = local + carry[None, :]

    out_ptrs = Y + (start + token_offsets[:, None]) * D + d[None, :]
    tl.store(out_ptrs, out, mask=mask)


# Hierarchical backward kernels

@triton.jit
def _segmented_cumsum_hier_bwd_local_kernel(
    DY,
    LOCAL_DX,
    TILE_SUMS,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    """
    Hierarchical backward stage 1.

    Computes local reverse cumulative sums within each tile and writes one
    tile sum per sequence tile.

    Grid:
    (B, MAX_TILES, ceil(D / BLOCK_D))

    Inputs:
    DY: [total_tokens, D]
    CU: [B + 1]

    Outputs:
    LOCAL_DX: [total_tokens, D]
    TILE_SUMS: [B, MAX_TILES, D]

    Requirements:
    MAX_TILES >= ceil(max_seq_len / BLOCK_M)

    Accumulation is performed in fp32. Stores cast to the output tensor dtype.
    """
    b = tl.program_id(0)
    tile_id = tl.program_id(1)
    d_block = tl.program_id(2)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    tile_start = tile_id * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    token_offsets = tile_start + offs_m
    d = d_block * BLOCK_D + offs_d

    ptrs = DY + (start + token_offsets[:, None]) * D + d[None, :]
    mask = (token_offsets[:, None] < L) & (d[None, :] < D)

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    local = tl.cumsum(vals, axis=0, reverse=True)

    out_ptrs = LOCAL_DX + (start + token_offsets[:, None]) * D + d[None, :]
    tl.store(out_ptrs, local, mask=mask)

    tile_sum = tl.sum(vals, axis=0)

    sum_ptrs = TILE_SUMS + (b * MAX_TILES + tile_id) * D + d
    tl.store(sum_ptrs, tile_sum, mask=d < D)


@triton.jit
def _segmented_cumsum_hier_bwd_carry_kernel(
    TILE_SUMS,
    TILE_CARRIES,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    """
    Hierarchical backward stage 2.

    Scans tile sums in reverse to produce exclusive reverse tile carries.

    For tile t:
    TILE_CARRIES[t] = sum_{p > t} TILE_SUMS[p]

    Grid:
    (B, ceil(D / BLOCK_D))

    Inputs:
    TILE_SUMS: [B, MAX_TILES, D]
    CU: [B + 1]

    Output:
    TILE_CARRIES: [B, MAX_TILES, D]

    Requirements:
    BLOCK_TILES >= MAX_TILES

    Accumulation is performed in fp32. Stores cast to TILE_CARRIES' dtype.
    """
    b = tl.program_id(0)
    d_block = tl.program_id(1)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    num_tiles = tl.cdiv(L, BLOCK_M)

    offs_t = tl.arange(0, BLOCK_TILES)
    offs_d = tl.arange(0, BLOCK_D)

    d = d_block * BLOCK_D + offs_d

    ptrs = TILE_SUMS + (b * MAX_TILES + offs_t[:, None]) * D + d[None, :]
    mask = (
        (offs_t[:, None] < num_tiles)
        & (offs_t[:, None] < MAX_TILES)
        & (d[None, :] < D)
    )

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    inclusive = tl.cumsum(vals, axis=0, reverse=True)
    exclusive = inclusive - vals

    out_ptrs = TILE_CARRIES + (b * MAX_TILES + offs_t[:, None]) * D + d[None, :]
    store_mask = (offs_t[:, None] < MAX_TILES) & (d[None, :] < D)

    tl.store(out_ptrs, exclusive, mask=store_mask)


@triton.jit
def _segmented_cumsum_hier_bwd_add_kernel(
    LOCAL_DX,
    DX,
    TILE_CARRIES,
    CU,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    """
    Hierarchical backward stage 3.

    Adds each tile's exclusive reverse carry to the local reverse cumulative sum.

    Grid:
    (B, MAX_TILES, ceil(D / BLOCK_D))

    Inputs:
    LOCAL_DX: [total_tokens, D]
    TILE_CARRIES: [B, MAX_TILES, D]
    CU: [B + 1]

    Output:
    DX: [total_tokens, D]

    LOCAL_DX and DX may alias if the wrapper wants in-place finalization.

    Accumulation is performed in fp32. Stores cast to DX's dtype.
    """
    b = tl.program_id(0)
    tile_id = tl.program_id(1)
    d_block = tl.program_id(2)

    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    L = end - start

    tile_start = tile_id * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    token_offsets = tile_start + offs_m
    d = d_block * BLOCK_D + offs_d

    carry_ptrs = TILE_CARRIES + (b * MAX_TILES + tile_id) * D + d
    carry = tl.load(carry_ptrs, mask=d < D, other=0.0).to(tl.float32)

    local_ptrs = LOCAL_DX + (start + token_offsets[:, None]) * D + d[None, :]
    mask = (token_offsets[:, None] < L) & (d[None, :] < D)

    local = tl.load(local_ptrs, mask=mask, other=0.0).to(tl.float32)

    out = local + carry[None, :]

    out_ptrs = DX + (start + token_offsets[:, None]) * D + d[None, :]
    tl.store(out_ptrs, out, mask=mask)


__all__ = [
    "_segmented_cumsum_oneblock_fwd_kernel",
    "_segmented_cumsum_oneblock_bwd_kernel",
    "_segmented_cumsum_hier_fwd_local_kernel",
    "_segmented_cumsum_hier_fwd_carry_kernel",
    "_segmented_cumsum_hier_fwd_add_kernel",
    "_segmented_cumsum_hier_bwd_local_kernel",
    "_segmented_cumsum_hier_bwd_carry_kernel",
    "_segmented_cumsum_hier_bwd_add_kernel",
]
