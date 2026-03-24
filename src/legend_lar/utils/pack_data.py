import torch
from torch import Tensor

@torch.no_grad()
def pack_data(x: Tensor, gE: Tensor, zero_token_id: int, return_meta: bool = False):
    """
    x: (B, T, S) int counts
        T-dimesion stores the time information, and the S-dimension stores the sipm id information.
        Each entry in x[b] is an integer count of hits in sipm s at time t.
    gE: (B, n_feats) Tensor of [gedet_id, partition_id, energy, ...]
    zero_token_id: token id in the sipm codebook assigned for a zero-pe event
    """
    
    assert x.dim() == 3, f"Expected x to be (B, T, S), got {tuple(x.shape)}"
    B, T, S = x.shape
    device = x.device

    # Detect zero-pe events
    has_any = (x > 0).any(dim=(1, 2)) # (B,)
    empty_b = (~has_any).nonzero(as_tuple=False).squeeze(1) # (E,)

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

    # Concatenate zero-pe events to the sequence. Assign t = 0 by default for these events
    if empty_b.numel() > 0:
        b_z = empty_b.to(dtype=torch.long)
        t_z = torch.zeros_like(b_z) # choose t = 0
        k_z = torch.full_like(b_z, fill_value=zero_token_id) # zero-pe token id

        b_all = torch.cat([b_rep, b_z], dim=0)
        t_all = torch.cat([t_rep, t_z], dim=0)
        k_all = torch.cat([k_rep, k_z], dim=0)
    else:
        b_all, t_all, k_all = b_rep, t_rep, k_rep
    # b_all, t_all, k_all now contains the b, t, k indices of sipm hits within a batch

    if b_all.numel() == 0:
        raise RuntimeError("No tokens produced (unexpected).")

    # Sort by (b, t, k) so tokens per batch are contiguous and time-ordered
    key = (b_all * T + t_all) * S + k_all
    order = torch.argsort(key)
    b_all = b_all[order]
    t_all = t_all[order]
    k_all = k_all[order]
    # (b_all, t_all, k_all) now contains the absolute (b, t, k) indices of unique 1 pe hits, ordered contigiously

    # Extract the corresponding HPGe id and energy based on the batch index (all 1 pe hits for each batch id b has the same (g, E))
    g = gE[b_all, 0] # (N_rep,)
    if gE.size(1) == 2:
        E = gE[b_all, 1] # (N_rep,)
    else:
        E = gE[b_all, 1:] # (N_rep, n_feats)

    # cu_seqlens and max_seqlen for FlashAttention varlen
    lengths = torch.bincount(b_all, minlength=B) # (B,)
    cu_seqlens = torch.zeros(B + 1, device=device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item())

    if return_meta:
        return g, E, b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths, order
    else:
        return g, E, b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths

@torch.no_grad()
def pack_nrec_data(x: Tensor, zero_token_id: int, return_meta: bool = False):
    """
    x: (B, T, S) int counts
        T-dimesion stores the time information, and the S-dimension stores the sipm id information.
        Each entry in x[b] is an integer count of hits in sipm s at time t.
    zero_token_id: token id in the sipm codebook assigned for a zero-pe event
    """
    
    assert x.dim() == 3, f"Expected x to be (B, T, S), got {tuple(x.shape)}"
    B, T, S = x.shape
    device = x.device

    # Detect zero-pe events
    has_any = (x > 0).any(dim=(1, 2)) # (B,)
    empty_b = (~has_any).nonzero(as_tuple=False).squeeze(1) # (E,)

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

    # Concatenate zero-pe events to the sequence. Assign t = 0 by default for these events
    if empty_b.numel() > 0:
        b_z = empty_b.to(dtype=torch.long)
        t_z = torch.zeros_like(b_z) # choose t = 0
        k_z = torch.full_like(b_z, fill_value=zero_token_id) # zero-pe token id

        b_all = torch.cat([b_rep, b_z], dim=0)
        t_all = torch.cat([t_rep, t_z], dim=0)
        k_all = torch.cat([k_rep, k_z], dim=0)
    else:
        b_all, t_all, k_all = b_rep, t_rep, k_rep
    # b_all, t_all, k_all now contains the b, t, k indices of sipm hits within a batch

    if b_all.numel() == 0:
        raise RuntimeError("No tokens produced (unexpected).")

    # Sort by (b, t, k) so tokens per batch are contiguous and time-ordered
    key = (b_all * T + t_all) * S + k_all
    order = torch.argsort(key)
    b_all = b_all[order]
    t_all = t_all[order]
    k_all = k_all[order]
    # (b_all, t_all, k_all) now contains the absolute (b, t, k) indices of unique 1 pe hits, ordered contigiously

    # cu_seqlens and max_seqlen for FlashAttention varlen
    lengths = torch.bincount(b_all, minlength=B) # (B,)
    cu_seqlens = torch.zeros(B + 1, device=device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item())

    if return_meta:
        return b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths, order
    else:
        return b_all, t_all, k_all, cu_seqlens, max_seqlen, lengths
