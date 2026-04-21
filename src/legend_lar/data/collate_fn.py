from typing import Tuple
import torch
from torch import Tensor
from legend_lar.utils import pack_nrec_data, pack_hpge_nrec_data, pack_continuous_nrec_data

class NRECCollateFn:
    def __init__(
        self,
        cls_placeholder_id: int,
        has_cls: bool = True,
        sipm_unbinned_pe: bool = False,
        cnre_group_size: int | None = None,
        hpge_prefix_count: int | None = None,
        cuda_device: str = "cpu",
        **kwargs
    ):
        """
        NOTE: LAr encoder always have a cls token. The has_cls arg only affects the HPGe encoder.
        """
        self.cls_placeholder_id = cls_placeholder_id
        self.has_cls = has_cls
        self.sipm_unbinned_pe = sipm_unbinned_pe
        self.cnre_group_size = cnre_group_size
        self.hpge_prefix_count = hpge_prefix_count
        self.device = "cpu"
        self.cuda_device = cuda_device
        self._device_set = False

    def _set_worker_cuda(self):
        if not self._device_set:
            torch.cuda.set_device(self.cuda_device)
            self._device_set = True

    @torch.no_grad()
    def _build_hpge_prefix_groups_padded(
        self,
        ge_b_idx: Tensor,
        ge_f_idx: Tensor,
        ge_cu_seqlens: Tensor
    ):
        if ge_b_idx is None or ge_f_idx is None or ge_cu_seqlens is None:
            return None

        if self.has_cls:
            return None

        if self.cnre_group_size is None or self.hpge_prefix_count is None:
            return None

        K = int(self.cnre_group_size)
        Tprefix = int(self.hpge_prefix_count)
        B_hpge = int(ge_cu_seqlens.numel() - 1)
        Gmax = B_hpge // K
        device = ge_b_idx.device

        ge_b_idx = ge_b_idx.to(torch.long)
        ge_f_idx = ge_f_idx.to(torch.long)

        group_ex_idx = torch.zeros((Tprefix, Gmax, K), device=device, dtype=torch.long)
        group_hpge_pos = torch.zeros((Tprefix, Gmax, K), device=device, dtype=torch.long)
        group_valid = torch.zeros((Tprefix, Gmax), device=device, dtype=torch.bool)

        if Gmax == 0:
            return group_ex_idx, group_hpge_pos, group_valid

        for t in range(Tprefix):
            pos = (ge_f_idx == t).nonzero(as_tuple=True)[0]
            n_valid = (pos.numel() // K) * K
            if n_valid == 0:
                continue

            pos = pos[:n_valid]
            ex_idx = ge_b_idx.index_select(0, pos)

            G_t = n_valid // K
            group_ex_idx[t, :G_t] = ex_idx.view(G_t, K)
            group_hpge_pos[t, :G_t] = pos.view(G_t, K)
            group_valid[t, :G_t] = True

        return group_ex_idx, group_hpge_pos, group_valid

    @torch.no_grad()
    def preprocess(self, x, gE, indices):
        x = torch.from_numpy(x).to(dtype=torch.float32)
        gE = torch.from_numpy(gE).to(dtype=torch.float32) if gE is not None else None

        if self.sipm_unbinned_pe:
            b_idx, t_idx, s_idx, v_vals, cu_seqlens, max_seqlen, lengths = pack_continuous_nrec_data(
                x, cls_placeholder_id=self.cls_placeholder_id
            )
        else:
            b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths = pack_nrec_data(
                x, cls_placeholder_id=self.cls_placeholder_id
            )
            v_vals = None

        ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths = pack_hpge_nrec_data(
            gE,
            cls_placeholder_id=self.cls_placeholder_id,
            has_cls=self.has_cls
        )

        ge_group_meta = self._build_hpge_prefix_groups_padded(
            ge_b_idx=ge_b_idx,
            ge_f_idx=ge_f_idx,
            ge_cu_seqlens=ge_cu_seqlens
        )

        return (
            (
                b_idx.to(dtype=torch.float32),
                t_idx.to(dtype=torch.float32),
                s_idx.to(dtype=torch.float32),
                v_vals.to(dtype=torch.float32) if v_vals is not None else None,
                cu_seqlens.to(dtype=torch.float32),
                int(max_seqlen),
                lengths.to(dtype=torch.float32)
            ),
            (
                ge_b_idx.to(dtype=torch.float32) if ge_b_idx is not None else None,
                ge_f_idx.to(dtype=torch.float32) if ge_f_idx is not None else None,
                ge_f_vals.to(dtype=torch.float32) if ge_f_vals is not None else None,
                ge_cu_seqlens.to(dtype=torch.float32) if ge_cu_seqlens is not None else None,
                int(ge_max_seqlen) if ge_max_seqlen is not None else None,
                ge_lengths.to(dtype=torch.float32) if ge_lengths is not None else None,
                ge_group_meta
            )
        ), indices

    def __call__(self, batch: Tuple):
        self._set_worker_cuda()

        (
            (b_idx, t_idx, s_idx, v_vals, cu_seqlens, max_seqlen, lengths),
            (ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths, ge_group_meta)
        ), indices = self.preprocess(*batch)

        b_idx = b_idx.pin_memory()
        t_idx = t_idx.pin_memory()
        s_idx = s_idx.pin_memory()
        v_vals = v_vals.pin_memory() if v_vals is not None else None
        cu_seqlens = cu_seqlens.pin_memory()
        lengths = lengths.pin_memory()

        if ge_b_idx is not None:
            ge_b_idx = ge_b_idx.pin_memory()
            ge_f_idx = ge_f_idx.pin_memory()
            ge_f_vals = ge_f_vals.pin_memory()
            ge_cu_seqlens = ge_cu_seqlens.pin_memory()
            ge_lengths = ge_lengths.pin_memory()

        if ge_group_meta is not None:
            ge_group_ex_idx, ge_group_hpge_pos, ge_group_valid = ge_group_meta
            ge_group_meta = (
                ge_group_ex_idx.pin_memory(),
                ge_group_hpge_pos.pin_memory(),
                ge_group_valid.pin_memory()
            )

        return (
            (b_idx, t_idx, s_idx, v_vals, cu_seqlens, max_seqlen, lengths),
            (ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths, ge_group_meta)
        ), indices
