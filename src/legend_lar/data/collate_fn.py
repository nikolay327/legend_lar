from typing import Tuple
import torch
from legend_lar.utils import pack_data

class CollateFn:
    def __init__(
        self,
        num_sipm_chs: int,
        true_coincidence_label: int,
        cuda_device: str = "cpu"
    ):
        self.num_sipm_chs = num_sipm_chs
        self.true_coincidence_label = true_coincidence_label

        self.device = "cpu"
        self.cuda_device = cuda_device
        self._device_set = False

    def _set_worker_cuda(self):
        if not self._device_set:
            torch.cuda.set_device(self.cuda_device)
            self._device_set = True

    @torch.no_grad()
    def preprocess(self, batch: Tuple):
        x, gE, labels = batch
        x = torch.from_numpy(x).to(dtype=torch.float32)
        gE = torch.from_numpy(gE).to(dtype=torch.float32)
        labels = torch.from_numpy(labels).to(dtype=torch.float32)

        gE_random_pairs = gE.new_zeros((len(x), 2))
        num_tc = int((labels == self.true_coincidence_label).sum().detach().item())
        gE_random_pairs[num_tc:] = gE
        gE_random_pairs[:num_tc] = gE

        g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths = pack_data(x, gE_random_pairs, zero_token_id=self.num_sipm_chs)

        return g.to(dtype=torch.float32), E.to(dtype=torch.float32), b_idx.to(dtype=torch.float32), t_idx.to(dtype=torch.float32), s_idx.to(dtype=torch.float32), cu_seqlens.to(dtype=torch.float32), int(max_seqlen), lengths.to(dtype=torch.float32), labels

    def __call__(self, batch: Tuple):
        self._set_worker_cuda()

        g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels = self.preprocess(batch)
        g = g.pin_memory()
        E = E.pin_memory()
        b_idx = b_idx.pin_memory()
        t_idx = t_idx.pin_memory()
        s_idx = s_idx.pin_memory()
        cu_seqlens = cu_seqlens.pin_memory()
        lengths = lengths.pin_memory()
        labels = labels.pin_memory()
        return g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels
