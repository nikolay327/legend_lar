from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

try:
    from triton.runtime.errors import OutOfResources
except Exception:
    OutOfResources = None

from .kernels import (
    _segmented_cumsum_oneblock_fwd_kernel,
    _segmented_cumsum_oneblock_bwd_kernel,
    _segmented_cumsum_hier_fwd_local_kernel,
    _segmented_cumsum_hier_fwd_carry_kernel,
    _segmented_cumsum_hier_fwd_add_kernel,
    _segmented_cumsum_hier_bwd_local_kernel,
    _segmented_cumsum_hier_bwd_carry_kernel,
    _segmented_cumsum_hier_bwd_add_kernel,
)


Algorithm = Literal["oneblock", "hierarchical"]
Direction = Literal["fwd", "bwd"]


@dataclass(frozen=True)
class SegmentCumsumConfig:
    oneblock_threshold: int = 512
    oneblock_block_m_buckets: tuple[int, ...] = (128, 256, 512)
    oneblock_block_d: int = 64

    hier_block_m: int = 128
    hier_block_d: int = 64
    max_block_tiles: int = 1024

    autotune: bool = False
    autotune_warmup: int = 2
    autotune_iters: int = 5
    autotune_cache: bool = True

    autotune_oneblock_block_m_buckets: tuple[int, ...] = (
        8,
        16,
        32,
        64,
        128,
        256,
    )
    autotune_oneblock_block_d_candidates: tuple[int, ...] = (
        32,
        64,
        128,
        256,
    )
    autotune_hier_block_m_candidates: tuple[int, ...] = (
        64,
        128,
        256,
        512,
    )
    autotune_hier_block_d_candidates: tuple[int, ...] = (
        32,
        64,
        128,
    )


@dataclass(frozen=True)
class ResolvedSegmentCumsumConfig:
    algorithm: Algorithm
    block_m: int
    block_d: int
    max_tiles: int | None = None
    block_tiles: int | None = None


_AUTOTUNE_CACHE: dict[tuple[object, ...], ResolvedSegmentCumsumConfig] = {}


def clear_segment_cumsum_autotune_cache() -> None:
    _AUTOTUNE_CACHE.clear()


def get_segment_cumsum_autotune_cache_size() -> int:
    return len(_AUTOTUNE_CACHE)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _smallest_bucket_geq(x: int, buckets: tuple[int, ...]) -> int | None:
    for bucket in buckets:
        if bucket >= x:
            return bucket
    return None


def _validate_positive_sorted(name: str, values: tuple[int, ...]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")

    if any(v <= 0 for v in values):
        raise ValueError(f"All values in {name} must be positive.")

    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted ascending.")


def _validate_positive_candidates(name: str, values: tuple[int, ...]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")

    if any(v <= 0 for v in values):
        raise ValueError(f"All values in {name} must be positive.")


def _validate_config(config: SegmentCumsumConfig) -> None:
    if config.oneblock_threshold < 0:
        raise ValueError("oneblock_threshold must be non-negative.")

    _validate_positive_sorted(
        "oneblock_block_m_buckets",
        config.oneblock_block_m_buckets,
    )

    if config.oneblock_block_d <= 0:
        raise ValueError("oneblock_block_d must be positive.")

    if config.hier_block_m <= 0:
        raise ValueError("hier_block_m must be positive.")

    if config.hier_block_d <= 0:
        raise ValueError("hier_block_d must be positive.")

    if config.max_block_tiles <= 0:
        raise ValueError("max_block_tiles must be positive.")

    if config.autotune_warmup < 0:
        raise ValueError("autotune_warmup must be non-negative.")

    if config.autotune_iters <= 0:
        raise ValueError("autotune_iters must be positive.")

    _validate_positive_sorted(
        "autotune_oneblock_block_m_buckets",
        config.autotune_oneblock_block_m_buckets,
    )

    _validate_positive_candidates(
        "autotune_oneblock_block_d_candidates",
        config.autotune_oneblock_block_d_candidates,
    )

    _validate_positive_candidates(
        "autotune_hier_block_m_candidates",
        config.autotune_hier_block_m_candidates,
    )

    _validate_positive_candidates(
        "autotune_hier_block_d_candidates",
        config.autotune_hier_block_d_candidates,
    )


def _validate_packed_tensor(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    *,
    tensor_name: str,
) -> None:
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise TypeError("max_seq_len must be a Python int.")

    if max_seq_len < 0:
        raise ValueError("max_seq_len must be non-negative.")

    if not x.is_cuda:
        raise ValueError(f"{tensor_name} must be a CUDA tensor.")

    if x.ndim != 2:
        raise ValueError(f"{tensor_name} must have shape [total_tokens, D].")

    if not x.is_contiguous():
        raise ValueError(f"{tensor_name} must be contiguous.")

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"{tensor_name} must have dtype float16, bfloat16, or float32."
        )

    if not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be a CUDA tensor.")

    if cu_seqlens.device != x.device:
        raise ValueError("cu_seqlens must be on the same device as the input.")

    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must have shape [B + 1].")

    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous.")

    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype int32 or int64.")

    if cu_seqlens.numel() == 0:
        raise ValueError("cu_seqlens must have at least one element.")

    B = cu_seqlens.numel() - 1
    T = x.shape[0]

    if B == 0 and T != 0:
        raise ValueError("cu_seqlens describes an empty batch but input is non-empty.")

    if T != 0 and max_seq_len == 0:
        raise ValueError("max_seq_len cannot be zero when input has tokens.")


def resolve_segment_cumsum_config(
    max_seq_len: int,
    *,
    config: SegmentCumsumConfig | None = None,
) -> ResolvedSegmentCumsumConfig:
    if config is None:
        config = SegmentCumsumConfig()

    _validate_config(config)

    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise TypeError("max_seq_len must be a Python int.")

    if max_seq_len < 0:
        raise ValueError("max_seq_len must be non-negative.")

    if max_seq_len <= config.oneblock_threshold:
        block_m = _smallest_bucket_geq(
            max_seq_len,
            config.oneblock_block_m_buckets,
        )

        if block_m is not None:
            return ResolvedSegmentCumsumConfig(
                algorithm="oneblock",
                block_m=block_m,
                block_d=config.oneblock_block_d,
            )

    num_tiles_needed = _ceil_div(max_seq_len, config.hier_block_m)
    max_tiles = _next_power_of_2(num_tiles_needed)
    block_tiles = max_tiles

    if block_tiles > config.max_block_tiles:
        raise NotImplementedError(
            "The current hierarchical implementation supports only one "
            "carry-scan level. Increase max_block_tiles or add another "
            "hierarchical carry-scan level."
        )

    return ResolvedSegmentCumsumConfig(
        algorithm="hierarchical",
        block_m=config.hier_block_m,
        block_d=config.hier_block_d,
        max_tiles=max_tiles,
        block_tiles=block_tiles,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _is_triton_out_of_resources(exc: BaseException) -> bool:
    if OutOfResources is not None and isinstance(exc, OutOfResources):
        return True

    msg = str(exc).lower()
    type_name = type(exc).__name__.lower()

    return (
        "outofresources" in type_name
        or "out of resource" in msg
        or "shared memory" in msg
        or "hardware limit" in msg
    )


def _is_invalid_candidate_exception(exc: BaseException) -> bool:
    return _is_cuda_oom(exc) or _is_triton_out_of_resources(exc)


def _recover_after_candidate_error() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _nearby_geq_buckets(
    x: int,
    buckets: tuple[int, ...],
    *,
    max_extra_buckets: int,
) -> tuple[int, ...]:
    valid = [b for b in buckets if b >= x]
    return tuple(valid[: 1 + max_extra_buckets])


def _shape_aware_block_d_candidates(
    D: int,
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    if D <= 32:
        max_block_d = 32
    elif D <= 64:
        max_block_d = 64
    elif D <= 128:
        max_block_d = 128
    else:
        max_block_d = max(candidates)

    filtered = tuple(b for b in candidates if b <= max_block_d)

    if len(filtered) == 0:
        return (min(candidates),)

    return filtered


def _shape_aware_hier_block_m_candidates(
    max_seq_len: int,
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    filtered = tuple(b for b in candidates if b <= max_seq_len)

    if len(filtered) > 0:
        return filtered

    return (min(candidates),)


def _make_autotune_candidates(
    max_seq_len: int,
    D: int,
    config: SegmentCumsumConfig,
) -> list[ResolvedSegmentCumsumConfig]:
    candidates: list[ResolvedSegmentCumsumConfig] = []

    oneblock_block_d_candidates = _shape_aware_block_d_candidates(
        D,
        config.autotune_oneblock_block_d_candidates,
    )

    oneblock_block_m_candidates = _nearby_geq_buckets(
        max_seq_len,
        config.autotune_oneblock_block_m_buckets,
        max_extra_buckets=1,
    )

    for block_m in oneblock_block_m_candidates:
        for block_d in oneblock_block_d_candidates:
            candidates.append(
                ResolvedSegmentCumsumConfig(
                    algorithm="oneblock",
                    block_m=block_m,
                    block_d=block_d,
                )
            )

    hier_block_m_candidates = _shape_aware_hier_block_m_candidates(
        max_seq_len,
        config.autotune_hier_block_m_candidates,
    )

    hier_block_d_candidates = _shape_aware_block_d_candidates(
        D,
        config.autotune_hier_block_d_candidates,
    )

    for block_m in hier_block_m_candidates:
        num_tiles_needed = _ceil_div(max_seq_len, block_m)
        max_tiles = _next_power_of_2(num_tiles_needed)
        block_tiles = max_tiles

        if block_tiles > config.max_block_tiles:
            continue

        for block_d in hier_block_d_candidates:
            candidates.append(
                ResolvedSegmentCumsumConfig(
                    algorithm="hierarchical",
                    block_m=block_m,
                    block_d=block_d,
                    max_tiles=max_tiles,
                    block_tiles=block_tiles,
                )
            )

    return candidates


def _autotune_config_signature(
    config: SegmentCumsumConfig,
) -> tuple[object, ...]:
    return (
        config.autotune_oneblock_block_m_buckets,
        config.autotune_oneblock_block_d_candidates,
        config.autotune_hier_block_m_candidates,
        config.autotune_hier_block_d_candidates,
        config.max_block_tiles,
    )


def _autotune_cache_key(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    *,
    direction: Direction,
    config: SegmentCumsumConfig,
) -> tuple[object, ...]:
    device = x.device
    capability = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)

    T, D = x.shape
    B = cu_seqlens.numel() - 1

    max_seq_len_bucket = _next_power_of_2(max(1, max_seq_len))
    total_tokens_bucket = _next_power_of_2(max(1, T))

    return (
        direction,
        capability,
        props.name,
        str(x.dtype),
        D,
        B,
        max_seq_len_bucket,
        total_tokens_bucket,
        _autotune_config_signature(config),
    )


def _launch_with_resolved_config(
    inp: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch_config: ResolvedSegmentCumsumConfig,
    *,
    direction: Direction,
) -> None:
    if launch_config.algorithm == "oneblock":
        if direction == "fwd":
            _launch_oneblock_fwd(inp, out, cu_seqlens, launch_config)
        else:
            _launch_oneblock_bwd(inp, out, cu_seqlens, launch_config)
    else:
        if direction == "fwd":
            _launch_hierarchical_fwd(inp, out, cu_seqlens, launch_config)
        else:
            _launch_hierarchical_bwd(inp, out, cu_seqlens, launch_config)


def _time_candidate_ms(
    inp: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch_config: ResolvedSegmentCumsumConfig,
    *,
    direction: Direction,
    warmup: int,
    iters: int,
) -> float:
    out = torch.empty_like(inp)

    def run_once() -> None:
        _launch_with_resolved_config(
            inp,
            out,
            cu_seqlens,
            launch_config,
            direction=direction,
        )

    torch.cuda.synchronize()

    for _ in range(warmup):
        run_once()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        run_once()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def autotune_segment_cumsum_config(
    inp: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    *,
    direction: Direction,
    config: SegmentCumsumConfig,
) -> ResolvedSegmentCumsumConfig:
    _validate_config(config)

    key = _autotune_cache_key(
        inp,
        cu_seqlens,
        max_seq_len,
        direction=direction,
        config=config,
    )

    if config.autotune_cache and key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[key]

    candidates = _make_autotune_candidates(
        max_seq_len,
        inp.shape[1],
        config,
    )

    best_config: ResolvedSegmentCumsumConfig | None = None
    best_ms = float("inf")

    for candidate in candidates:
        try:
            ms = _time_candidate_ms(
                inp,
                cu_seqlens,
                candidate,
                direction=direction,
                warmup=config.autotune_warmup,
                iters=config.autotune_iters,
            )
        except Exception as exc:
            _recover_after_candidate_error()

            if _is_invalid_candidate_exception(exc):
                continue

            raise

        if ms < best_ms:
            best_ms = ms
            best_config = candidate

    if best_config is None:
        best_config = resolve_segment_cumsum_config(
            max_seq_len,
            config=config,
        )

    if config.autotune_cache:
        _AUTOTUNE_CACHE[key] = best_config

    return best_config


def segment_cumsum_fwd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    *,
    config: SegmentCumsumConfig | None = None,
    return_launch_config: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ResolvedSegmentCumsumConfig]:
    _validate_packed_tensor(
        x,
        cu_seqlens,
        max_seq_len,
        tensor_name="x",
    )

    if config is None:
        config = SegmentCumsumConfig()

    y = torch.empty_like(x)

    T, D = x.shape

    if T == 0 or D == 0:
        launch_config = resolve_segment_cumsum_config(
            max_seq_len,
            config=config,
        )

        if return_launch_config:
            return y, launch_config

        return y

    if config.autotune:
        launch_config = autotune_segment_cumsum_config(
            x,
            cu_seqlens,
            max_seq_len,
            direction="fwd",
            config=config,
        )
    else:
        launch_config = resolve_segment_cumsum_config(
            max_seq_len,
            config=config,
        )

    _launch_with_resolved_config(
        x,
        y,
        cu_seqlens,
        launch_config,
        direction="fwd",
    )

    if return_launch_config:
        return y, launch_config

    return y


def segment_cumsum_bwd(
    dy: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    *,
    config: SegmentCumsumConfig | None = None,
    return_launch_config: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ResolvedSegmentCumsumConfig]:
    _validate_packed_tensor(
        dy,
        cu_seqlens,
        max_seq_len,
        tensor_name="dy",
    )

    if config is None:
        config = SegmentCumsumConfig()

    dx = torch.empty_like(dy)

    T, D = dy.shape

    if T == 0 or D == 0:
        launch_config = resolve_segment_cumsum_config(
            max_seq_len,
            config=config,
        )

        if return_launch_config:
            return dx, launch_config

        return dx

    if config.autotune:
        launch_config = autotune_segment_cumsum_config(
            dy,
            cu_seqlens,
            max_seq_len,
            direction="bwd",
            config=config,
        )
    else:
        launch_config = resolve_segment_cumsum_config(
            max_seq_len,
            config=config,
        )

    _launch_with_resolved_config(
        dy,
        dx,
        cu_seqlens,
        launch_config,
        direction="bwd",
    )

    if return_launch_config:
        return dx, launch_config

    return dx


def _launch_oneblock_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch_config: ResolvedSegmentCumsumConfig,
) -> None:
    B = cu_seqlens.numel() - 1
    D = x.shape[1]

    grid = (B, _ceil_div(D, launch_config.block_d))

    _segmented_cumsum_oneblock_fwd_kernel[grid](
        x,
        y,
        cu_seqlens,
        D=D,
        BLOCK_M=launch_config.block_m,
        BLOCK_D=launch_config.block_d,
    )


def _launch_oneblock_bwd(
    dy: torch.Tensor,
    dx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch_config: ResolvedSegmentCumsumConfig,
) -> None:
    B = cu_seqlens.numel() - 1
    D = dy.shape[1]

    grid = (B, _ceil_div(D, launch_config.block_d))

    _segmented_cumsum_oneblock_bwd_kernel[grid](
        dy,
        dx,
        cu_seqlens,
        D=D,
        BLOCK_M=launch_config.block_m,
        BLOCK_D=launch_config.block_d,
    )


def _launch_hierarchical_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch_config: ResolvedSegmentCumsumConfig,
) -> None:
    if launch_config.max_tiles is None or launch_config.block_tiles is None:
        raise ValueError("Hierarchical launch requires max_tiles and block_tiles.")

    B = cu_seqlens.numel() - 1
    T, D = x.shape

    block_m = launch_config.block_m
    block_d = launch_config.block_d
    max_tiles = launch_config.max_tiles
    block_tiles = launch_config.block_tiles

    local_y = torch.empty(
        (T, D),
        device=x.device,
        dtype=torch.float32,
    )

    tile_sums = torch.empty(
        (B, max_tiles, D),
        device=x.device,
        dtype=torch.float32,
    )

    tile_carries = torch.empty_like(tile_sums)

    tile_grid = (B, max_tiles, _ceil_div(D, block_d))
    carry_grid = (B, _ceil_div(D, block_d))

    _segmented_cumsum_hier_fwd_local_kernel[tile_grid](
        x,
        local_y,
        tile_sums,
        cu_seqlens,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MAX_TILES=max_tiles,
    )

    _segmented_cumsum_hier_fwd_carry_kernel[carry_grid](
        tile_sums,
        tile_carries,
        cu_seqlens,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MAX_TILES=max_tiles,
        BLOCK_TILES=block_tiles,
    )

    _segmented_cumsum_hier_fwd_add_kernel[tile_grid](
        local_y,
        y,
        tile_carries,
        cu_seqlens,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MAX_TILES=max_tiles,
    )


def _launch_hierarchical_bwd(
    dy: torch.Tensor,
    dx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch_config: ResolvedSegmentCumsumConfig,
) -> None:
    if launch_config.max_tiles is None or launch_config.block_tiles is None:
        raise ValueError("Hierarchical launch requires max_tiles and block_tiles.")

    B = cu_seqlens.numel() - 1
    T, D = dy.shape

    block_m = launch_config.block_m
    block_d = launch_config.block_d
    max_tiles = launch_config.max_tiles
    block_tiles = launch_config.block_tiles

    local_dx = torch.empty(
        (T, D),
        device=dy.device,
        dtype=torch.float32,
    )

    tile_sums = torch.empty(
        (B, max_tiles, D),
        device=dy.device,
        dtype=torch.float32,
    )

    tile_carries = torch.empty_like(tile_sums)

    tile_grid = (B, max_tiles, _ceil_div(D, block_d))
    carry_grid = (B, _ceil_div(D, block_d))

    _segmented_cumsum_hier_bwd_local_kernel[tile_grid](
        dy,
        local_dx,
        tile_sums,
        cu_seqlens,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MAX_TILES=max_tiles,
    )

    _segmented_cumsum_hier_bwd_carry_kernel[carry_grid](
        tile_sums,
        tile_carries,
        cu_seqlens,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MAX_TILES=max_tiles,
        BLOCK_TILES=block_tiles,
    )

    _segmented_cumsum_hier_bwd_add_kernel[tile_grid](
        local_dx,
        dx,
        tile_carries,
        cu_seqlens,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MAX_TILES=max_tiles,
    )


__all__ = [
    "SegmentCumsumConfig",
    "ResolvedSegmentCumsumConfig",
    "resolve_segment_cumsum_config",
    "autotune_segment_cumsum_config",
    "clear_segment_cumsum_autotune_cache",
    "get_segment_cumsum_autotune_cache_size",
    "segment_cumsum_fwd",
    "segment_cumsum_bwd",
]
