from typing import Tuple
import torch
from legend_lar.utils import pack_nrec_data, pack_hpge_nrec_data

class NRECCollateFn:
    def __init__(
        self,
        cuda_device: str = "cpu",
        **kwargs
    ):
        self.device = "cpu"
        self.cuda_device = cuda_device
        self._device_set = False

    def _set_worker_cuda(self):
        if not self._device_set:
            torch.cuda.set_device(self.cuda_device)
            self._device_set = True

    @torch.no_grad()
    def preprocess(self, x, gE, indices):
        x = torch.from_numpy(x).to(dtype=torch.float32)
        gE = torch.from_numpy(gE).to(dtype=torch.float32)

        b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths = pack_nrec_data(x, cls_placeholder_id=-99)
        ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths = pack_hpge_nrec_data(gE, cls_placeholder_id=-99)
        return (
            (
                b_idx.to(dtype=torch.float32), t_idx.to(dtype=torch.float32), s_idx.to(dtype=torch.float32),
                cu_seqlens.to(dtype=torch.float32), int(max_seqlen), lengths.to(dtype=torch.float32)
            ),
            (
                ge_b_idx.to(dtype=torch.float32), ge_f_idx.to(dtype=torch.float32), ge_f_vals.to(dtype=torch.float32),
                ge_cu_seqlens.to(dtype=torch.float32), int(ge_max_seqlen), ge_lengths.to(dtype=torch.float32)
            )
        ), indices

    def __call__(self, batch: Tuple):
        self._set_worker_cuda()

        (
            (b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths),
            (ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths)
        ), indices = self.preprocess(*batch)

        b_idx = b_idx.pin_memory()
        t_idx = t_idx.pin_memory()
        s_idx = s_idx.pin_memory()
        cu_seqlens = cu_seqlens.pin_memory()
        lengths = lengths.pin_memory()

        ge_b_idx = ge_b_idx.pin_memory()
        ge_f_idx = ge_f_idx.pin_memory()
        ge_f_vals = ge_f_vals.pin_memory()
        ge_cu_seqlens = ge_cu_seqlens.pin_memory()
        ge_lengths = ge_lengths.pin_memory()

        return (
            (b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths),
            (ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths)
        ), indices
