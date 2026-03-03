import os
import json
import random
import math
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.amp import autocast

import numpy as np

from legend_lar.utils import BootstrappedKFoldConfig, EvalConfig, _initialize_configs, _init_torch
from legend_lar.model import ContrastiveRatioEstimator
from legend_lar.data import BootstrappedKFoldLArListDataset, CollateFn, KFoldBootstrap_worker_init_fn

from lgdo import Table, Array, lh5

class Evaluator:
    def __init__(
        self,
        config: BootstrappedKFoldConfig,
        eval_config: EvalConfig,
        model_dir: str,
        data_dir: Path,
        save_to: str,
        device: str | int
    ):
        self.device = device
        self.config = config
        self.eval_config = eval_config

        self.BASE_SEED = self.config.rng_seed
        self.BASE_RNG = random.Random(self.BASE_SEED)

        self.rng_seed_for_data_plit = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_bootstrap = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_data_sampling = self.BASE_RNG.getrandbits(64)

        self.fid_value = mp.Value("i", 0)

        self.null_anchor_buffer: Tensor = None # stored on GPU
        self.null_val_anchor_buffer: Tensor = None # stored on GPU
        self.model_dir = model_dir
        self.save_to = save_to

        self._init_cal_dataloader()
        self._init_phy_dataloader()

        self.phy_4by4 = np.load(str(data_dir / self.eval_config.phy_4by4_data) + '.npy').astype(bool)
        self.cal_4by4 = np.load(str(data_dir / self.eval_config.fc_4by4_data) + '.npy').astype(bool)

    def _init_phy_dataloader(self):
        test_folds = [np.load(f'{self.model_dir}/fid_{i}/fold_indices.npy').astype(np.float64).tolist() for i in range(self.config.num_folds)]
        self.phy_dataset = BootstrappedKFoldLArListDataset(
            lar_paths=self.config.data_paths,
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.config.local_batch_size,
            labels=self.config.labels,
            prior=self.config.prior,
            hpge_path=self.config.hpge_id_and_energy,
            hpge_energy_mean=self.config.hpge_energy_mean,
            hpge_energy_std=self.config.hpge_energy_std,
            test_folds=test_folds,
            rng_seed_for_split=self.rng_seed_for_data_plit,
            times_of_mixing=self.config.times_of_mixing,
            sg_train_val_cal_test_frac=self.config.sg_train_val_cal_test_frac,
            mode=4,
            fold_id=self.fid_value
        )
        collate_fn = CollateFn(
            num_sipm_chs=self.config.num_sipms,
            cuda_device=self.device
        )
        self.phy_dataloader = DataLoader(
            dataset=self.phy_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            prefetch_factor=4,
            persistent_workers=True,
            worker_init_fn=KFoldBootstrap_worker_init_fn,
            collate_fn=collate_fn
        )

    def _init_cal_dataloader(self):
        self.cal_mode_value = mp.Value("i", 2)
        self.cal_dataset = BootstrappedKFoldLArListDataset(
            lar_paths=self.config.data_paths,
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.config.local_batch_size,
            labels=self.config.labels,
            prior=self.config.prior,
            hpge_path=self.config.hpge_id_and_energy,
            hpge_energy_mean=self.config.hpge_energy_mean,
            hpge_energy_std=self.config.hpge_energy_std,
            rng_seed_for_split=self.rng_seed_for_data_plit,
            times_of_mixing=self.config.times_of_mixing,
            num_folds=self.config.num_folds,
            sg_train_val_cal_test_frac=self.config.sg_train_val_cal_test_frac,
            mode=self.cal_mode_value
        )
        collate_fn = CollateFn(
            num_sipm_chs=self.config.num_sipms,
            cuda_device=self.device
        )
        self.cal_dataloader = DataLoader(
            dataset=self.cal_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            prefetch_factor=4,
            persistent_workers=True,
            worker_init_fn=KFoldBootstrap_worker_init_fn,
            collate_fn=collate_fn
        )

    def clean_state_dict(self, state_dict):
        cleaned_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('_orig_mod.', '')
            cleaned_dict[cleaned_key] = value
        return cleaned_dict

    @torch.no_grad()
    def load_ensemble(self, fid: int):
        self.ensemble = nn.ModuleList()
        for i in range(self.config.num_bootstraps_per_fold):
            model = ContrastiveRatioEstimator(
                config=self.config,
                device=self.device
            ).to(dtype=torch.float32, device=self.device)

            cp = torch.load(f'{self.model_dir}/fid_{fid}/bid_{i}/model.pt', map_location=self.device)
            model.load_state_dict(self.clean_state_dict(cp["model"]), strict=True)
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            self.ensemble.append(model)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _set_null_anchor_buffer(self, mode_id: int):
        self.cal_dataloader.dataset.set_mode(mode_id)
        null_anchor_buffer = []
        is_lar_vetoed = []
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, indices in self.cal_dataloader:
            g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32)
            b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32)
            max_seqlen=int(max_seqlen)
            lengths=lengths.to(device=self.device, non_blocking=True)

            indices = indices.numpy().astype(np.int64)
            is_lar_vetoed.append(self.cal_4by4[indices])

            anchor_ensemble = []
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                for model in self.ensemble:
                    anchors, _ = model(# (B, D)
                        g=g,
                        E=E,
                        b_idx=b_idx,
                        t_idx=t_idx,
                        s_idx=s_idx,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        lengths=lengths
                    )
                    anchor_ensemble.append(anchors.unsqueeze(0))
            anchor_ensemble = torch.cat(anchor_ensemble, dim=0) # (n_ensemble, B, D) keep in bf16 for storage
            null_anchor_buffer.append(anchor_ensemble)
        is_lar_vetoed = np.concatenate(is_lar_vetoed, axis=0)
        return null_anchor_buffer, is_lar_vetoed

    def cache_null_anchors(self):
        # Calibration dataset
        self.null_anchor_buffer, self.null_is_lar_vetoed = self._set_null_anchor_buffer(mode_id=2)
        # Null held-out dataset
        self.null_val_anchor_buffer, self.null_val_is_lar_vetoed = self._set_null_anchor_buffer(mode_id=3)

    def batch_forward(
        self,
        g: Tensor,
        E: Tensor,
        b_idx: Tensor,
        t_idx: Tensor,
        s_idx: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        lengths: Tensor
    ):
        e_hpge = []
        logits = []
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            for model in self.ensemble:
                anchors, e_hpge_ = model(
                    g=g,
                    E=E,
                    b_idx=b_idx,
                    t_idx=t_idx,
                    s_idx=s_idx,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    lengths=lengths
                )
                logits.append(((anchors.to(dtype=torch.float32) * e_hpge_.to(dtype=torch.float32)).sum(dim=-1, keepdim=False) / self.config.temperature).unsqueeze(0)) # (B, )
                e_hpge.append(e_hpge_.reshape(1, -1, self.config.hidden_size)) # (1, B, D)
        e_hpge = torch.cat(e_hpge, dim=0).to(dtype=torch.float32) # (n_ensemble, B, D)

        logits = torch.cat(logits, dim=0) # (n_ensemble, B)
        delta = torch.var(logits, dim=0)
        # Geometric mean
        logits = (logits.sum(dim=0) / self.config.num_bootstraps_per_fold).to(dtype=torch.float32) # (B,)

        return delta, logits, e_hpge

    @torch.no_grad()
    def evaluate_fold(self, fid: int):
        self.phy_dataloader.dataset.set_fold_id(fid)
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, indices in self.phy_dataloader:
            # Calculate the test statistic of each event
            delta, logits, e_hpge = self.batch_forward(# (B,), (n_ensemble, B, D), # (n_ensemble, B, D)
                g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True)
            )

            # Calculate the test statistic under the null for each event
            null_logits = []
            for anchor in self.null_anchor_buffer:
                null_logits_ = e_hpge.unsqueeze(2) * anchor.to(torch.float32).unsqueeze(1) # (n_ensemble, B, 1, D) * (n_ensemble, 1, B_rc, D) = (n_ensemble, B, B_rc, D)
                null_logits_ = null_logits_.sum(dim=-1) / self.config.temperature # (n_ensemble, B, B_rc)
                null_logits.append(null_logits_)
            null_logits_ = None

            null_logits = torch.cat(null_logits, dim=-1) # (n_ensemble, B, N_rc_data)
            null_delta = torch.var(null_logits, dim=0)
            null_logits = torch.sum(null_logits, dim=0) / self.config.num_bootstraps_per_fold # (B, N_rc_data)
            # null_delta = null_delta - null_logits

            # Calculate the test statistic under the heldout null for each event
            heldout_null_logits = []
            for anchor in self.null_val_anchor_buffer:
                null_logits_ = e_hpge.unsqueeze(2) * anchor.to(torch.float32).unsqueeze(1) # (n_ensemble, B, 1, D) * (n_ensemble, 1, B_rc, D) = (n_ensemble, B, B_rc, D)
                null_logits_ = null_logits_.sum(dim=-1) / self.config.temperature # (n_ensemble, B, B_rc)
                heldout_null_logits.append(null_logits_)
            null_logits_ = None

            heldout_null_logits = torch.cat(heldout_null_logits, dim=-1) # (n_ensemble, B, N_rc_data)
            heldout_null_delta = torch.var(heldout_null_logits, dim=0)
            heldout_null_logits = torch.sum(heldout_null_logits, dim=0) / self.config.num_bootstraps_per_fold # (B, N_rc_data)
            # heldout_null_delta = heldout_null_delta - heldout_null_logits

            # Calculate the cut value (for plots)
            cut = torch.quantile(null_logits, 1 - self.eval_config.alpha, dim=-1).cpu().numpy() # (B,)

            N_rc_data = null_logits.shape[1]
            # Evidence p-value
            p_val = ((null_logits >= logits.reshape(-1, 1)).float().sum(dim=-1) + 1) / (N_rc_data + 1) # (B,)
            p_val = p_val.cpu().numpy()
            # Epistemic p-value
            p_val_ep = ((null_delta >= delta.reshape(-1, 1)).float().sum(dim=-1) + 1) / (N_rc_data + 1) # (B,)
            p_val_ep = p_val_ep.cpu().numpy()

            # Calculate the calibrated evidence and epistemic p-val under the heldout null (do it in batches)
            sorted_null = null_logits.sort(dim=-1).values # (B, N_rc_data)
            null_logits = null_logits.cpu().numpy() # want to release

            sorted_null_delta = null_delta.sort(dim=-1).values # (B, N_rc_data)
            null_delta = null_delta.cpu().numpy()

            heldout_null_logits_len = heldout_null_logits.shape[1]
            num_iters = math.ceil(heldout_null_logits_len / self.config.local_batch_size)
    
            null_p_val = []
            null_p_val_ep = []
            for it in range(num_iters):
                # (B, local_batch_size)
                heldout_batch = heldout_null_logits[:, it*self.config.local_batch_size:] if (it + 1)*self.config.local_batch_size > heldout_null_logits_len else heldout_null_logits[:, it*self.config.local_batch_size: (it + 1)*self.config.local_batch_size]
                idx = torch.searchsorted(sorted_null, heldout_batch, right=False) # (B, lb)
                null_p_val_ = ((N_rc_data - idx).float() + 1) / (N_rc_data + 1)
                null_p_val.append(null_p_val_)

                heldout_batch = heldout_null_delta[:, it*self.config.local_batch_size:] if (it + 1)*self.config.local_batch_size > heldout_null_logits_len else heldout_null_delta[:, it*self.config.local_batch_size: (it + 1)*self.config.local_batch_size]
                idx = torch.searchsorted(sorted_null_delta, heldout_batch, right=False) # (B, lb)
                null_p_val_ = ((N_rc_data - idx).float() + 1) / (N_rc_data + 1)
                null_p_val_ep.append(null_p_val_)

            null_p_val_ = None
            idx = None

            null_p_val = torch.cat(null_p_val, dim=-1).cpu().numpy() # (B, heldout_null_logits_len)
            null_p_val_ep = torch.cat(null_p_val_ep, dim=-1).cpu().numpy() # (B, heldout_null_logits_len)

            g_id = g.to(torch.float32).cpu().numpy() # (B,)
            energy = E.to(torch.float32).cpu().numpy() # (B,)

            # Global score calculation
            indices = indices.numpy().astype(np.int64)
            is_lar_vetoed = self.phy_4by4[indices]
            flag_4by4 = (p_val_ep <= self.eval_config.alpha_epistemic) & is_lar_vetoed
            global_score = np.copy(p_val)
            global_score[flag_4by4] = 0. # always reject untrustworthy scores that do not pass the 4x4 classifier

            # Split the heldout null dataset into global calibration dataset and validation
            N_heldout_null = null_p_val.shape[1]
            N_heldout_null_cal = math.ceil(self.eval_config.global_calib_frac * N_heldout_null)
            N_heldout_null_val = N_heldout_null - N_heldout_null_cal

            null_p_val_cal = null_p_val[:, :N_heldout_null_cal]
            null_p_val = null_p_val[:, N_heldout_null_cal:]

            null_p_val_ep_cal = null_p_val_ep[:, :N_heldout_null_cal]
            null_p_val_ep = null_p_val_ep[:, N_heldout_null_cal:]

            # Calculate the null global score
            flag_4by4 = (null_p_val_ep_cal <= self.eval_config.alpha_epistemic) & self.null_val_is_lar_vetoed[:N_heldout_null_cal]
            null_global_score = np.copy(null_p_val_cal)
            null_global_score[flag_4by4] = 0.

            # Calculate the heldout null global score
            flag_4by4 = (null_p_val_ep <= self.eval_config.alpha_epistemic) & self.null_val_is_lar_vetoed[N_heldout_null_cal:]
            heldout_null_global_score = np.copy(null_p_val)
            heldout_null_global_score[flag_4by4] = 0.

            # Global p-value
            p_val_glob = ((null_global_score <= global_score.reshape(-1, 1)).sum(axis=-1) + 1) / (N_heldout_null_cal + 1) # (B,)

            # Global p-value under the held-out null
            sorted_null = torch.tensor(null_global_score).sort(dim=-1).values
            heldout_null_len = heldout_null_global_score.shape[1]
            num_iters = math.ceil(heldout_null_len / self.config.local_batch_size)
            heldout_null_p_val_glob = []
            for it in range(num_iters):
                heldout_batch = heldout_null_global_score[:, it*self.config.local_batch_size:] if (it + 1)*self.config.local_batch_size > heldout_null_len else heldout_null_global_score[:, it*self.config.local_batch_size: (it + 1)*self.config.local_batch_size]
                heldout_batch = torch.tensor(heldout_batch)
                idx = torch.searchsorted(sorted_null, heldout_batch, right=True) # (B, lb)
                null_p_val_ = (idx.float() + 1) / (N_heldout_null_cal + 1)
                heldout_null_p_val_glob.append(null_p_val_.numpy())
            heldout_null_p_val_glob = np.concatenate(heldout_null_p_val_glob, axis=-1)

            table_size = len(energy)
            lgdo_table = Table(
                size=table_size,
                col_dict={
                    "evt_idx": Array(indices.astype(np.float32)),
                    "g_id": Array(g_id, dtype=np.float32),
                    "energy": Array(energy, dtype=np.float32),
                    "is_lar_vetoed": Array(is_lar_vetoed, dtype=np.float32),

                    "t_epistemic": Array(delta.cpu().numpy(), dtype=np.float32),
                    "t_evidence": Array(logits.cpu().numpy(), dtype=np.float32),
                    "t_global": Array(global_score, dtype=np.float32),
                    "cut": Array(cut, dtype=np.float32),

                    "p_evidence": Array(p_val, dtype=np.float32),
                    "p_epistemic": Array(p_val_ep, dtype=np.float32),
                    "p_global": Array(p_val_glob, dtype=np.float32),

                    "null_t_epistemic": Array(null_delta, dtype=np.float32),
                    "null_t_evidence": Array(null_logits, dtype=np.float32),

                    "heldout_null_t_global": Array(heldout_null_global_score, dtype=np.float32),

                    "heldout_null_p_evidence_cal": Array(null_p_val_cal, dtype=np.float32),
                    "heldout_null_p_epistemic_cal": Array(null_p_val_ep_cal, dtype=np.float32),

                    "heldout_null_p_evidence_test": Array(null_p_val, dtype=np.float32),
                    "heldout_null_p_epistemic_test": Array(null_p_val_ep, dtype=np.float32),
                    "heldout_null_p_global_test": Array(heldout_null_p_val_glob, dtype=np.float32),

                    "null_t_global": Array(null_global_score, dtype=np.float32),
                    "t_global": Array(global_score, dtype=np.float32)
                }
            )
            lh5.write(
                lgdo_table,
                "phy/tcal", # NOTE: for now hardcoded
                self.save_to + "/inferred.lh5",
                n_rows=table_size,
                wo_mode="append"
            )
        self.null_anchor_buffer = None
        self.null_val_anchor_buffer = None

    def encode_hpge(self, g: Tensor, E: Tensor):
        e_hpge = []
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            for model in self.ensemble:
                e_hpge_ = model.joint_hpge_emb(g=g, E=E)
                e_hpge_ = F.normalize(e_hpge_.to(dtype=torch.float32), p=2, dim=-1)
                e_hpge.append(e_hpge_.reshape(1, -1, self.config.hidden_size)) # (1, B, D)
        e_hpge = torch.cat(e_hpge, dim=0).to(dtype=torch.float32) # (n_ensemble, B, D)
        return e_hpge

    @torch.no_grad()
    def scan_across_energies(self, fid: int):
        self.load_ensemble(fid)
        self.cache_null_anchors()
        energy_grid = (torch.arange(200, 3000, 1, dtype=torch.float32, device=self.device) - self.config.hpge_energy_mean) / self.config.hpge_energy_std

        B = self.config.local_batch_size // 2
        num_batches = math.ceil(energy_grid.shape[0] / B)
        gedet_indices = torch.arange(self.config.num_hpges, device=self.device, dtype=torch.long)

        for gedet_idx in gedet_indices:
            for batch_idx in range(num_batches):
                energy = energy_grid[batch_idx*B:] if (batch_idx * (B + 1)) > energy_grid.shape[0] else energy_grid[batch_idx*B: (batch_idx + 1)*B]
                e_hpge = self.encode_hpge(g=gedet_idx.unsqueeze(0).repeat_interleave(energy.shape[0], dim=0).reshape(-1), E=energy) # (n_ensemble, B, D)

                # Null
                null_logits = []
                for anchor in self.null_anchor_buffer:
                    null_logits_ = e_hpge.unsqueeze(2) * anchor.to(torch.float32).unsqueeze(1) # (n_ensemble, B, 1, D) * (n_ensemble, 1, B_rc, D) = (n_ensemble, B, B_rc, D)
                    null_logits_ = null_logits_.sum(dim=-1) / self.config.temperature # (n_ensemble, B, B_rc)
                    null_logits.append(null_logits_)
                null_logits_ = None

                null_logits = torch.cat(null_logits, dim=-1) # (n_ensemble, B, N_rc_data)
                null_delta = torch.var(null_logits, dim=0)
                null_logits = torch.sum(null_logits, dim=0) / self.config.num_bootstraps_per_fold # (B, N_rc_data)

                # heldout null
                heldout_null_logits = []
                for anchor in self.null_val_anchor_buffer:
                    null_logits_ = e_hpge.unsqueeze(2) * anchor.to(torch.float32).unsqueeze(1) # (n_ensemble, B, 1, D) * (n_ensemble, 1, B_rc, D) = (n_ensemble, B, B_rc, D)
                    null_logits_ = null_logits_.sum(dim=-1) / self.config.temperature # (n_ensemble, B, B_rc)
                    heldout_null_logits.append(null_logits_)
                null_logits_ = None

                heldout_null_logits = torch.cat(heldout_null_logits, dim=-1) # (n_ensemble, B, N_rc_data)
                heldout_null_delta = torch.var(heldout_null_logits, dim=0)
                heldout_null_logits = torch.sum(heldout_null_logits, dim=0) / self.config.num_bootstraps_per_fold # (B, N_rc_data)

                N_rc_data = null_logits.shape[1]
                sorted_null = null_logits.sort(dim=-1).values # (B, N_rc_data)
                null_logits = null_logits.cpu().numpy() # want to release

                sorted_null_delta = null_delta.sort(dim=-1).values # (B, N_rc_data)
                null_delta = null_delta.cpu().numpy()

                heldout_null_logits_len = heldout_null_logits.shape[1]
                num_iters = math.ceil(heldout_null_logits_len / B)

                null_p_val = []
                null_p_val_ep = []
                for it in range(num_iters):
                    # (B, local_batch_size)
                    heldout_batch = heldout_null_logits[:, it*B:] if (it + 1)*B > heldout_null_logits_len else heldout_null_logits[:, it*B: (it + 1)*B]
                    idx = torch.searchsorted(sorted_null, heldout_batch, right=False) # (B, lb)
                    null_p_val_ = ((N_rc_data - idx).float() + 1) / (N_rc_data + 1)
                    null_p_val.append(null_p_val_)

                    heldout_batch = heldout_null_delta[:, it*B:] if (it + 1)*B > heldout_null_logits_len else heldout_null_delta[:, it*B: (it + 1)*B]
                    idx = torch.searchsorted(sorted_null_delta, heldout_batch, right=False) # (B, lb)
                    null_p_val_ = ((N_rc_data - idx).float() + 1) / (N_rc_data + 1)
                    null_p_val_ep.append(null_p_val_)

                null_p_val_ = None
                idx = None

                null_p_val = torch.cat(null_p_val, dim=-1).cpu().numpy() # (B, heldout_null_logits_len)
                null_p_val_ep = torch.cat(null_p_val_ep, dim=-1).cpu().numpy() # (B, heldout_null_logits_len)
                energy = energy.to(torch.float32).cpu().numpy()

                # Split the heldout null dataset into global calibration dataset and validation
                N_heldout_null = null_p_val.shape[1]
                N_heldout_null_cal = math.ceil(self.eval_config.global_calib_frac * N_heldout_null)
                N_heldout_null_val = N_heldout_null - N_heldout_null_cal

                null_p_val_cal = null_p_val[:, :N_heldout_null_cal]
                null_p_val = null_p_val[:, N_heldout_null_cal:]

                null_p_val_ep_cal = null_p_val_ep[:, :N_heldout_null_cal]
                null_p_val_ep = null_p_val_ep[:, N_heldout_null_cal:]

                # Calculate the null global score
                flag_4by4 = (null_p_val_ep_cal <= self.eval_config.alpha_epistemic) & self.null_val_is_lar_vetoed[:N_heldout_null_cal]
                null_global_score = np.copy(null_p_val_cal)
                null_global_score[flag_4by4] = 0.

                # Calculate the heldout null global score
                flag_4by4 = (null_p_val_ep <= self.eval_config.alpha_epistemic) & self.null_val_is_lar_vetoed[N_heldout_null_cal:]
                heldout_null_global_score = np.copy(null_p_val)
                heldout_null_global_score[flag_4by4] = 0.

                # Global p-value under the held-out null
                sorted_null = torch.tensor(null_global_score).sort(dim=-1).values
                heldout_null_len = heldout_null_global_score.shape[1]
                num_iters = math.ceil(heldout_null_len / B)
                heldout_null_p_val_glob = []
                for it in range(num_iters):
                    heldout_batch = heldout_null_global_score[:, it*B:] if (it + 1)*B > heldout_null_len else heldout_null_global_score[:, it*B: (it + 1)*B]
                    heldout_batch = torch.tensor(heldout_batch)
                    idx = torch.searchsorted(sorted_null, heldout_batch, right=True) # (B, lb)
                    null_p_val_ = (idx.float() + 1) / (N_heldout_null_cal + 1)
                    heldout_null_p_val_glob.append(null_p_val_.numpy())
                heldout_null_p_val_glob = np.concatenate(heldout_null_p_val_glob, axis=-1)

                table_size = len(energy)
                lgdo_table = Table(
                    size=table_size,
                    col_dict={
                        "energy": Array(energy, dtype=np.float32),

                        "null_t_epistemic": Array(null_delta, dtype=np.float32),
                        "null_t_evidence": Array(null_logits, dtype=np.float32),
                        "null_t_global": Array(null_global_score, dtype=np.float32),

                        "heldout_null_t_global": Array(heldout_null_global_score, dtype=np.float32),

                        "heldout_null_p_evidence_cal": Array(null_p_val_cal, dtype=np.float32),
                        "heldout_null_p_epistemic_cal": Array(null_p_val_ep_cal, dtype=np.float32),

                        "heldout_null_p_evidence_test": Array(null_p_val, dtype=np.float32),
                        "heldout_null_p_epistemic_test": Array(null_p_val_ep, dtype=np.float32),
                        "heldout_null_p_global_test": Array(heldout_null_p_val_glob, dtype=np.float32)
                    }
                )
                lh5.write(
                    lgdo_table,
                    "grid_scan/gedet_gid{gid:02d}".format(gid=int(gedet_idx.view(-1).item())),
                    self.save_to + "/grid_scan.lh5",
                    n_rows=table_size,
                    wo_mode="append"
                )
        self.null_anchor_buffer = None
        self.null_val_anchor_buffer = None

    def evaluate(self):
        for fid in range(self.config.num_folds):
            self.load_ensemble(fid)
            self.cache_null_anchors()
            self.evaluate_fold(fid)

            self.null_anchor_buffer = None
            # Null held-out dataset
            self.null_val_anchor_buffer = None

        torch.cuda.empty_cache()

        # for fid in range(self.config.num_folds):
        #     self.scan_across_energies(fid)
        #     self.null_anchor_buffer = None
        #     # Null held-out dataset
        #     self.null_val_anchor_buffer = None
        #     break

def evaluate(experiment: str, model_name: str, version: str, period: str, working_dir: str, data_dir: str):
    local_rank, rank, world_size, device = _init_torch()
    wd = Path(working_dir)
    mmpd = Path(data_dir)

    with open(str(wd / "trained" / experiment / model_name / "eval_config.json"), "r") as f:
        eval_cfg = json.load(f)
    
    eval_config = EvalConfig()
    eval_config.__dict__.update(eval_cfg)

    config, data_config, paths = _initialize_configs(
        config_obj=BootstrappedKFoldConfig(),
        wd=wd,
        experiment=experiment,
        model_name=model_name,
        version=version,
        mmpd=mmpd
    )

    config.data_paths = [str(paths.data_dir / f"{key}.npz") for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.prior = [data_config[key]["prior"] for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.labels = [data_config[key]["label"] for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.hpge_id_and_energy = data_config["hpge_id_and_energy"]
    config.hpge_id_and_energy = str(paths.data_dir / f'{config.hpge_id_and_energy}.npy')

    os.makedirs(wd / "data" / experiment / "tier" / model_name / period, exist_ok=True)
    evaluator = Evaluator(
        config=config,
        eval_config=eval_config,
        model_dir=str(wd / "trained" / experiment / model_name / version / "checkpoints"),
        data_dir=paths.data_dir,
        save_to=str(wd / "data" / experiment / "tier" / model_name / period),
        device=device
    )

    evaluator.evaluate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training started")
    parser.add_argument("experiment", type=str)
    parser.add_argument("model_name", type=str, help="Name of the model being trained")
    parser.add_argument("version", type=str)
    parser.add_argument("working_dir", type=str, help="Top-most dir of the training pipeline")
    parser.add_argument("data_dir", type=str, help="Directory the training data is saved under")
    parser.add_argument("training_config", type=str, help="JSON config file of the training, which contains model and training configurations, data configs, etc.")
    parser.add_argument("cache_dir", type=str, help="Directory to store torch.inductor and triton cache")
    args = parser.parse_args()

    import os
    from pathlib import Path

    BASE = args.cache_dir
    rank = os.environ["RANK"]
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{BASE}/inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{BASE}/triton/rank_{rank}")
    os.environ.setdefault("NUMBA_CACHE_DIR", f"{BASE}/numba/rank_{rank}")

    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    JOB_SHM_DIR = os.environ["JOB_SHMTMPDIR"] if "JOB_SHMTMPDIR" in os.environ else None

    evaluate(args.experiment, args.model_name, args.version, args.working_dir, args.data_dir, args.training_config)
