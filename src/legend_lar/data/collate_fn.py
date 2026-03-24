from typing import Tuple
import torch
from torch import Tensor
from legend_lar.utils import pack_data, pack_nrec_data

class CollateFn:
    def __init__(
        self,
        num_sipm_chs: int,
        cuda_device: str = "cpu",
        **kwargs
    ):
        self.num_sipm_chs = num_sipm_chs
        self.true_coincidence_label = 1

        self.device = "cpu"
        self.cuda_device = cuda_device
        self._device_set = False

    def _set_worker_cuda(self):
        if not self._device_set:
            torch.cuda.set_device(self.cuda_device)
            self._device_set = True

    @torch.no_grad()
    def preprocess(self, x: Tensor, gE: Tensor, labels: Tensor):
        x = torch.from_numpy(x).to(dtype=torch.float32)
        gE = torch.from_numpy(gE).to(dtype=torch.float32)
        labels = torch.from_numpy(labels).to(dtype=torch.float32)

        if len(gE) != len(x): # For the unconditional branch and BCE branch
            gE_random_pairs = gE.new_zeros((len(x), 2))
            num_tc = int((labels == self.true_coincidence_label).sum().detach().item())
            gE_random_pairs[num_tc:] = gE
            gE_random_pairs[:num_tc] = gE
            g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths = pack_data(x, gE_random_pairs, zero_token_id=self.num_sipm_chs)
            # return g.to(dtype=torch.float32), E.to(dtype=torch.float32), b_idx.to(dtype=torch.float32), t_idx.to(dtype=torch.float32), s_idx.to(dtype=torch.float32), cu_seqlens.to(dtype=torch.float32), int(max_seqlen), lengths.to(dtype=torch.float32), labels
            return gE[:, 0].to(dtype=torch.float32), gE[:, 1].to(dtype=torch.float32), b_idx.to(dtype=torch.float32), t_idx.to(dtype=torch.float32), s_idx.to(dtype=torch.float32), cu_seqlens.to(dtype=torch.float32), int(max_seqlen), lengths.to(dtype=torch.float32), labels
        else: # For the conditional branch
            g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths = pack_data(x, gE, zero_token_id=self.num_sipm_chs)
            return gE[:, 0].to(dtype=torch.float32), gE[:, 1].to(dtype=torch.float32), b_idx.to(dtype=torch.float32), t_idx.to(dtype=torch.float32), s_idx.to(dtype=torch.float32), cu_seqlens.to(dtype=torch.float32), int(max_seqlen), lengths.to(dtype=torch.float32), labels

    def __call__(self, batch: Tuple):
        self._set_worker_cuda()

        g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels = self.preprocess(*batch)
        g = g.pin_memory()
        E = E.pin_memory()
        b_idx = b_idx.pin_memory()
        t_idx = t_idx.pin_memory()
        s_idx = s_idx.pin_memory()
        cu_seqlens = cu_seqlens.pin_memory()
        lengths = lengths.pin_memory()
        labels = labels.pin_memory()
        return g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels

class NRECCollateFn:
    def __init__(
        self,
        num_sipm_chs: int,
        cuda_device: str = "cpu",
        **kwargs
    ):
        self.num_sipm_chs = num_sipm_chs
        self.true_coincidence_label = 1

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

        b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths = pack_data(x, zero_token_id=self.num_sipm_chs)
        return gE.to(dtype=torch.float32), b_idx.to(dtype=torch.float32), t_idx.to(dtype=torch.float32), s_idx.to(dtype=torch.float32), cu_seqlens.to(dtype=torch.float32), int(max_seqlen), lengths.to(dtype=torch.float32), indices

    def __call__(self, batch: Tuple):
        self._set_worker_cuda()

        gE, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, indices = self.preprocess(*batch)
        gE = gE.pin_memory()
        b_idx = b_idx.pin_memory()
        t_idx = t_idx.pin_memory()
        s_idx = s_idx.pin_memory()
        cu_seqlens = cu_seqlens.pin_memory()
        lengths = lengths.pin_memory()
        return gE, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, indices
