import torch
from torch import Tensor

@torch.no_grad()
def pack_nrec_data(x: Tensor, cls_placeholder_id: int, return_meta: bool = False):
    """
    x: (B, T, S) int counts
        T-dimesion stores the time information, and the S-dimension stores the sipm id information.
        Each entry in x[b] is an integer count of hits in sipm s at time t.
    cls_placeholder_id: placeholder id used in the packed s_idx tensor for the CLS token
    """
    
    assert x.dim() == 3, f"Expected x to be (B, T, S), got {tuple(x.shape)}"
    B, T, S = x.shape
    device = x.device

    # Indices of nonzero entries (b,t,k) where count > 0
    nz = (x > 0).nonzero(as_tuple=False) # (M, 3)
    
    if nz.numel() > 0:
        counts = x[nz[:, 0], nz[:, 1], nz[:, 2]].to(torch.long) # (M,)
        # Expand each (b,t,k) row counts[m] times
        rep = torch.repeat_interleave(torch.arange(nz.shape[0], device=device), counts) # (N_rep,)
        b_rep = nz[rep, 0] # (N_rep,)
        t_rep = nz[rep, 1] # (N_rep,)
        k_rep = nz[rep, 2] # (N_rep,)
    else: # no nonzeros anywhere
        b_rep = t_rep = k_rep = torch.empty((0,), device=device, dtype=torch.long)

    # Add one CLS placeholder token to every batch entry
    b_cls = torch.arange(B, device=device, dtype=torch.long) # (B,)
    t_cls = torch.zeros_like(b_cls) # placeholder only
    k_cls = torch.full_like(b_cls, fill_value=cls_placeholder_id) # (B,)

    b_all = torch.cat([b_rep, b_cls], dim=0)
    t_all = torch.cat([t_rep, t_cls], dim=0)
    k_all = torch.cat([k_rep, k_cls], dim=0)
    # b_all, t_all, k_all now contains the b, t, k indices of sipm hits + CLS token placeholder

    if b_all.numel() == 0:
        raise RuntimeError("No tokens produced (unexpected).")
    
    # Sort by (b, is_not_cls, t, k) so CLS is first within each batch entry, and the remaining tokens are time-ordered.
    is_not_cls = (k_all != cls_placeholder_id).to(torch.long)
    k_sort = torch.where(is_not_cls.bool(), k_all, torch.zeros_like(k_all))
    key = (((b_all * 2 + is_not_cls) * (T + 1) + t_all) * S + k_sort)
    order = torch.argsort(key)

    b_all = b_all[order]
    t_all = t_all[order]
    k_all = k_all[order]
    # (b_all, t_all, k_all) now contains the absolute (b, t, k) indices of unique 1 pe hits, with CLS first in every packed sequence

    # cu_seqlens and max_seqlen for FlashAttention varlen
    lengths = torch.bincount(b_all, minlength=B)  # (B,)
    cu_seqlens = torch.zeros(B + 1, device=device, dtype=torch.long)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item())

    if return_meta:
        return b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths, order
    else:
        return b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths

@torch.no_grad()
def pack_hpge_nrec_data(x: Tensor, cls_placeholder_id: int, return_meta: bool = False):
    """
    x: (B, H) HPGe feature vectors
        H-dimesion stores HPGe observables: global partition id, detector id, energy, pulse shapes features like A/E, LQ, drift_time, etc.
        Some of those features can be nan for some detectors. This class handle those cases by packing the data into variable-length input for
        downstream transformer blocks.

        global partition id and detector id are integers that will never be nan.

    cls_placeholder_id: placeholder id used in the packed s_idx tensor for the CLS token
    """
    
    assert x.dim() == 2, f"Expected x to be (B, H), got {tuple(x.shape)}"
    B, H = x.shape
    device = x.device

    # Valid feature entries are the non-nan entries
    valid = ~torch.isnan(x) # (B, H)

    # Indices of valid feature entries (b, h)
    nz = valid.nonzero(as_tuple=False) # (N_valid, 2)
    
    if nz.numel() > 0:
        b_rep = nz[:, 0].to(torch.long) # (N_valid,)
        f_rep = nz[:, 1].to(torch.long) # (N_valid,)
        v_rep = x[b_rep, f_rep] # (N_valid,)
    else: # no valid features anywhere
        b_rep = torch.empty((0,), device=device, dtype=torch.long)
        f_rep = torch.empty((0,), device=device, dtype=torch.long)
        v_rep = torch.empty((0,), device=device, dtype=x.dtype)

    # Add one CLS placeholder token to every batch entry
    b_cls = torch.arange(B, device=device, dtype=torch.long) # (B,)
    f_cls = torch.full((B,), fill_value=cls_placeholder_id, device=device, dtype=torch.long) # (B,)
    v_cls = torch.zeros(B, device=device, dtype=x.dtype) # placeholder only

    b_all = torch.cat([b_rep, b_cls], dim=0)
    f_all = torch.cat([f_rep, f_cls], dim=0)
    v_all = torch.cat([v_rep, v_cls], dim=0)
    # b_all, f_all, v_all now contains the batch index, feature index, and feature value
    # for all valid HPGe features + one CLS placeholder token per batch entry

    if b_all.numel() == 0:
        raise RuntimeError("No tokens produced (unexpected).")
    
    # Sort by (b, is_not_cls, f) so CLS is first within each batch entry, and the remaining feature tokens are ordered by feature index.
    is_not_cls = (f_all != cls_placeholder_id).to(torch.long)
    f_sort = torch.where(is_not_cls.bool(), f_all, torch.zeros_like(f_all))
    key = ((b_all * 2 + is_not_cls) * H) + f_sort
    order = torch.argsort(key)
    # Packed tokens are now contiguous by batch, with CLS first in every sequence

    # cu_seqlens and max_seqlen for FlashAttention varlen
    lengths = torch.bincount(b_all, minlength=B) # (B,)
    cu_seqlens = torch.zeros(B + 1, device=device, dtype=torch.long)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item())

    if return_meta:
        return b_all, f_all, v_all, cu_seqlens, max_seqlen, lengths, order
    else:
        return b_all, f_all, v_all, cu_seqlens, max_seqlen, lengths
