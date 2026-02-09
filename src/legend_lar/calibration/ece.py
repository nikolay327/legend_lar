import torch
from torch import Tensor
import torch.nn.functional as F


class NRETestMetrics:
    def __init__(self, ece_bins: int, n_classes: int = None, device = None):
        # NOTE: need to adjust many device arguments for mixed-precision training. Currently, this is not implemented
        # NOTE: doesn't yet support DDP

        self.n_bins = ece_bins
        self.n_classes = n_classes
        self.device = device

        self.bin_edges = torch.linspace(0, 1, self.n_bins+1, device=self.device)

    def _create_buffer(self, shape):
        return torch.zeros(shape, dtype=torch.float32, device=self.device)
        
    def reset_buffers(self):
        self.w = self._create_buffer(self.n_classes-1)
        self.w_sq = self._create_buffer(self.n_classes-1)

        self.bin_cts_perclass = self._create_buffer((self.n_classes, self.n_bins))
        self.conf_sums_perclass = self._create_buffer((self.n_classes, self.n_bins))
        self.corr_sums_perclass = self._create_buffer((self.n_classes, self.n_bins))

        self.bin_cts_top = self._create_buffer(self.n_bins)
        self.conf_sums_top = self._create_buffer(self.n_bins)
        self.corr_sums_top = self._create_buffer(self.n_bins)

        self.total_samples = 0
        self.acc_sum = self._create_buffer(1)
    
    @torch.no_grad()
    def update(self, logits: Tensor, labels: Tensor):
        posteriors = F.softmax(logits, dim=-1)
        B, K = posteriors.shape
        if self.n_classes is None:
            self.n_classes = K
            self.reset_buffers()
        self.total_samples += B

        # accuracy
        self.acc_sum += (posteriors.argmax(dim=-1, keepdim=False) == labels).to(dtype=torch.float32).sum()

        # Effective Sample Size (ESS)
        mask = labels == 0
        w = logits[mask][:, 1:].clone()
        w = (w - logits[mask][:, :1]).exp()
        w_sq = w**2

        self.w += w.sum(dim=0, keepdim=False)
        self.w_sq += w_sq.sum(dim=0, keepdim=False)

        # --------- Per-class ECE stats --------- #
        one_hot = torch.zeros_like(posteriors, device=self.device).scatter_(1, labels.unsqueeze(1), 1) # (B, K)

        bin_ids = torch.bucketize(posteriors, self.bin_edges, right=False) - 1 # (B, K)
        bin_ids = torch.clamp(bin_ids, 0, self.n_bins-1)

        # Encode class dimension into bin index
        bin_ids = bin_ids + (torch.arange(K, device=self.device) * self.n_bins).reshape(1, -1) # (n, K)
        bin_ids = bin_ids.flatten()

        flat_post = posteriors.flatten()
        one_hot = one_hot.flatten()

        self.bin_cts_perclass += torch.bincount(bin_ids, minlength=self.n_bins*K).to(dtype=torch.float32).reshape(K, self.n_bins) # (K*n_bins,) -> (K, n_bins)
        self.conf_sums_perclass += torch.bincount(bin_ids, weights=flat_post, minlength=self.n_bins*K).to(dtype=torch.float32).reshape(K, self.n_bins)
        self.corr_sums_perclass += torch.bincount(bin_ids, weights=one_hot, minlength=self.n_bins*K).to(dtype=torch.float32).reshape(K, self.n_bins)

        # --------- Top-class ECE stats --------- #
        confs, preds = posteriors.max(dim=1) # (B,)
        correct = (preds == labels) # (B,)

        bin_ids = torch.bucketize(confs, self.bin_edges, right=False) - 1 # (B,)
        bin_ids = torch.clamp(bin_ids, 0, self.n_bins-1)

        self.bin_cts_top += torch.bincount(bin_ids, minlength=self.n_bins).to(dtype=torch.float32) # (n_bins,)
        self.conf_sums_top += torch.bincount(bin_ids, weights=confs, minlength=self.n_bins).to(dtype=torch.float32)
        self.corr_sums_top += torch.bincount(bin_ids, weights=correct, minlength=self.n_bins).to(dtype=torch.float32)

    @torch.no_grad()
    def aggregate(self, out: dict = None):
        # accuracy
        acc = (self.acc_sum / self.total_samples).item()
        # ESS
        ess = (self.w**2 / self.w_sq).to(device="cpu").numpy().tolist()

        # Top class ECE
        inv_bin_cts = 1 / self.bin_cts_top.clamp(min=1)
        conf_top = self.conf_sums_top * inv_bin_cts
        acc_top = self.corr_sums_top * inv_bin_cts

        ece_top = ((self.bin_cts_top / self.total_samples) * (acc_top - conf_top).abs()).sum().item()
        conf_top = conf_top.to(device="cpu").numpy().tolist()
        acc_top = acc_top.to(device="cpu").numpy().tolist()

        # Wilson CIs (vertical error bar)
        k_top = self.corr_sums_top
        n_top = self.bin_cts_top.clamp(min=1)
        p_hat_top = k_top / n_top
        z = 1.0
        den = 1.0 + (z*z)/n_top
        center = p_hat_top + (z*z)/(2.0*n_top)
        rad = z * torch.sqrt((p_hat_top*(1.0 - p_hat_top))/n_top + (z*z)/(4.0*n_top*n_top))
        ci_low_top = ((center - rad)/den).clamp(0.0, 1.0)
        ci_high_top = ((center + rad)/den).clamp(0.0, 1.0)

        # Top ECE CI (vertical bar)
        p_top = (self.corr_sums_top / self.bin_cts_top.clamp(min=1)).clamp(0.0, 1.0)
        ece_top_ci = self._bootstrap_binomial_ci(
            self.bin_cts_top, p_top,
            compute_ece_fn=lambda k: self._ece_top_from_counts(k, self.bin_cts_top, self.conf_sums_top, self.total_samples)
        )
        # horizontal bar
        alpha = 0.32
        n_top = self.bin_cts_top.clamp(min=1)
        # Hoeffding (distribution-free, conservative)
        rad_top = torch.sqrt(torch.log(torch.tensor(2.0/alpha, device=n_top.device)) / (2.0 * n_top))
        conf_ci_top_low = (self.conf_sums_top * (1.0 / n_top) - rad_top).clamp(0.0, 1.0)
        conf_ci_top_high = (self.conf_sums_top * (1.0 / n_top) + rad_top).clamp(0.0, 1.0)

        # Per-class ECE
        inv_bin_cts = 1 / self.bin_cts_perclass.clamp(min=1)
        conf_pc = self.conf_sums_perclass * inv_bin_cts
        acc_pc = self.corr_sums_perclass * inv_bin_cts

        N_per_class = self.bin_cts_perclass.sum(dim=1, keepdim=True).clamp(min=1.0)
        ece_pc = ((self.bin_cts_perclass / N_per_class) * (acc_pc - conf_pc).abs()).sum(dim=1, keepdim=False).to(device="cpu").numpy().tolist()
        conf_pc = conf_pc.to(device="cpu").numpy().tolist()
        acc_pc = acc_pc.to(device="cpu").numpy().tolist()

        # Per-class ECE CI (vertical bar)
        p_pc = (self.corr_sums_perclass / self.bin_cts_perclass.clamp(min=1)).clamp(0.0, 1.0)
        ece_pc_ci = self._bootstrap_binomial_ci(
            self.bin_cts_perclass, p_pc,
            compute_ece_fn=lambda k: self._ece_pc_from_counts(k, self.bin_cts_perclass, self.conf_sums_perclass)
        )
        # horizontal bar
        alpha = 0.32
        n_pc = self.bin_cts_perclass.clamp(min=1)
        rad_pc = torch.sqrt(torch.log(torch.tensor(2.0/alpha, device=n_pc.device)) / (2.0 * n_pc))
        conf_ci_pc_low = (self.conf_sums_perclass * (1.0 / n_pc) - rad_pc).clamp(0.0, 1.0)
        conf_ci_pc_high = (self.conf_sums_perclass * (1.0 / n_pc) + rad_pc).clamp(0.0, 1.0)

        # Wilson CIs (vertical error bar)
        k_pc = self.corr_sums_perclass
        n_pc = self.bin_cts_perclass.clamp(min=1)
        p_hat_pc = k_pc / n_pc
        z = 1.0
        den_pc = 1.0 + (z*z)/n_pc
        center_pc = p_hat_pc + (z*z)/(2.0*n_pc)
        rad_pc = z * torch.sqrt((p_hat_pc*(1.0 - p_hat_pc))/n_pc + (z*z)/(4.0*n_pc*n_pc))
        ci_low_pc = ((center_pc - rad_pc)/den_pc).clamp(0.0, 1.0)
        ci_high_pc = ((center_pc + rad_pc)/den_pc).clamp(0.0, 1.0)

        # Global ECE (vertical error bar)
        bin_cts_glob = self.bin_cts_perclass.sum(dim=0)
        conf_sums_glob = self.conf_sums_perclass.sum(dim=0)
        corr_sums_glob = self.corr_sums_perclass.sum(dim=0)

        inv_bin_cts = 1 / bin_cts_glob.clamp(min=1)
        conf_glob = conf_sums_glob * inv_bin_cts
        acc_glob = corr_sums_glob * inv_bin_cts

        ece_glob = ((bin_cts_glob / (self.total_samples * self.n_classes)) * (acc_glob - conf_glob).abs()).sum().item()
        conf_glob = conf_glob.to(device="cpu").numpy().tolist()
        acc_glob = acc_glob.to(device="cpu").numpy().tolist()

        # Wilson CIs (vertical error bar)
        k_glob = corr_sums_glob
        n_glob = bin_cts_glob.clamp(min=1)
        p_hat_glob = k_glob / n_glob
        z = 1.0
        den_glob = 1.0 + (z*z)/n_glob
        center_glob = p_hat_glob + (z*z)/(2.0*n_glob)
        rad_glob = z * torch.sqrt((p_hat_glob*(1.0 - p_hat_glob))/n_glob + (z*z)/(4.0*n_glob*n_glob))
        ci_low_glob = ((center_glob - rad_glob)/den_glob).clamp(0.0, 1.0)
        ci_high_glob = ((center_glob + rad_glob)/den_glob).clamp(0.0, 1.0)

        # Global ECE CI (veritcal bar)
        p_glob = (corr_sums_glob / bin_cts_glob.clamp(min=1)).clamp(0.0, 1.0)
        ece_glob_ci = self._bootstrap_binomial_ci(
            bin_cts_glob, p_glob,
            compute_ece_fn=lambda k: self._ece_glob_from_counts(k, bin_cts_glob, conf_sums_glob, self.total_samples)
        )
        # horizontal bar
        alpha = 0.32
        n_glob = bin_cts_glob.clamp(min=1)
        rad_glob = torch.sqrt(torch.log(torch.tensor(2.0/alpha, device=n_glob.device)) / (2.0 * n_glob))
        conf_ci_glob_low = (conf_sums_glob * (1.0 / n_glob) - rad_glob).clamp(0.0, 1.0)
        conf_ci_glob_high = (conf_sums_glob * (1.0 / n_glob) + rad_glob).clamp(0.0, 1.0)

        if out is None:
            out = {}
        # write into out
        out["acc"] = acc
        out["ess"] = ess

        out["bin_cts_top"] = self.bin_cts_top.to(device="cpu").numpy().tolist()
        out["ece_top"] = ece_top
        out["conf_top"] = conf_top
        out["acc_top"] = acc_top
        out["acc_ci_top_low"] = ci_low_top.to(device="cpu").numpy().tolist()
        out["acc_ci_top_high"] = ci_high_top.to(device="cpu").numpy().tolist()
        out["conf_ci_top_low"] = conf_ci_top_low.to(device="cpu").numpy().tolist()
        out["conf_ci_top_high"] = conf_ci_top_high.to(device="cpu").numpy().tolist()

        out["bin_cts_perclass"] = self.bin_cts_perclass.to(device="cpu").numpy().tolist()
        out["ece_pc"] = ece_pc
        out["conf_pc"] = conf_pc
        out["acc_pc"] = acc_pc
        out["acc_ci_pc_low"] = ci_low_pc.to(device="cpu").numpy().tolist()
        out["acc_ci_pc_high"] = ci_high_pc.to(device="cpu").numpy().tolist()
        out["conf_ci_pc_low"] = conf_ci_pc_low.to(device="cpu").numpy().tolist()
        out["conf_ci_pc_high"] = conf_ci_pc_high.to(device="cpu").numpy().tolist()

        out["bin_cts_glob"] = bin_cts_glob.to(device="cpu").numpy().tolist()
        out["ece_glob"] = ece_glob
        out["conf_glob"] = conf_glob
        out["acc_glob"] = acc_glob
        out["acc_ci_glob_low"] = ci_low_glob.to(device="cpu").numpy().tolist()
        out["acc_ci_glob_high"] = ci_high_glob.to(device="cpu").numpy().tolist()

        out["ece_top_ci"] = ece_top_ci.to(device="cpu").numpy().tolist() # [low, high]
        out["ece_pc_ci_low"] = ece_pc_ci[0].to(device="cpu").numpy().tolist() # per class
        out["ece_pc_ci_high"] = ece_pc_ci[1].to(device="cpu").numpy().tolist()
        out["ece_glob_ci"] = ece_glob_ci.to(device="cpu").numpy().tolist()
        out["conf_ci_glob_low"] = conf_ci_glob_low.to(device="cpu").numpy().tolist()
        out["conf_ci_glob_high"] = conf_ci_glob_high.to(device="cpu").numpy().tolist()

        return out
    
    # --- helpers for ECE from bin summaries ---
    def _ece_top_from_counts(self, k, n, s, total):
        n_clamped = n.clamp(min=1)
        acc = k / n_clamped
        conf = s / n_clamped
        return ((n / total) * (acc - conf).abs()).sum()

    def _ece_pc_from_counts(self, k_pc, n_pc, s_pc):
        n_pc_cl = n_pc.clamp(min=1)
        acc_pc = k_pc / n_pc_cl
        conf_pc = s_pc / n_pc_cl
        N_per_class = n_pc.sum(dim=1, keepdim=True).clamp(min=1.0)
        return ((n_pc / N_per_class) * (acc_pc - conf_pc).abs()).sum(dim=1, keepdim=False)  # shape [K]

    def _ece_glob_from_counts(self, k_g, n_g, s_g, total):
        n_g_cl = n_g.clamp(min=1)
        acc_g = k_g / n_g_cl
        conf_g = s_g / n_g_cl
        return ((n_g / (total * self.n_classes)) * (acc_g - conf_g).abs()).sum()

    # --- generic parametric bootstrap on binomial counts; supports scalar or vector ECEs ---
    def _bootstrap_binomial_ci(self, n, p, compute_ece_fn, R=500, q_low=0.16, q_high=0.84):
        # n, p: tensors with same shape as bins (broadcast OK). compute_ece_fn: fn(k)->ECE tensor
        n = n.to(dtype=torch.float32)
        p = p.to(dtype=torch.float32)
        samples = []
        for _ in range(R):
            k_samp = torch.binomial(n, p)  # element-wise Binomial draws
            samples.append(compute_ece_fn(k_samp))
        samples = torch.stack(samples, dim=0)  # [R, ...]
        qs = torch.tensor([q_low, q_high], device=samples.device)
        return torch.quantile(samples, qs, dim=0)  # shape [2, ...]
