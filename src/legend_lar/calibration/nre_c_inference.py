import os
import math

from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import numpy as np
from numpy.lib.format import open_memmap
import scipy

from functools import partial

from legend_lar.model import NREC
from legend_lar.utils import NRECConfig, _initialize_configs, decode_geom, _init_torch, FileDB
from legend_lar.data import ParallelBootstrappedKFoldLArListDataset, ParallelKFoldBootstrap_worker_init_fn, NRECCollateFn

from lgdo import Table, Array, lh5

class NRECCalibrator:
    def __init__(
        self,
        file_db: FileDB,
        partition: str,
        model_name: str,
        version: str,
        dataset_version: str,
        batch_size: int,
        device: str | int,
        max_t: int = -1
    ):
        self.device = device

        self.file_db = file_db
        self.partition = partition
        self.model_name = model_name
        self.version = version
        self.dataset_version = dataset_version
        self.batch_size = batch_size
        self.max_t = max_t

        model_cfg, data_config = _initialize_configs(
            config_obj=NRECConfig(),
            config_path=file_db.build_file(
                tier="model_config",
                partition=partition,
                model_name=model_name,
                version=version
            )
        )

        self.config = model_cfg
        self.data_config = data_config

        self.inv_temp = 1 / self.config.temperature
        self.ev_ep_null_buffer: List[Tensor] = None # stored on GPU
        self.glob_null_buffer: List[Tensor] = None # stored on GPU
        self.classical_classifier_glob_buffer = None

        self._init_phy_dataloader()
        self._init_ev_ep_null_dataloader()
        self._init_glob_null_dataloader()

    def _init_phy_dataloader(self):
        test_folds = [
            np.load(
                self.file_db.build_file(
                    tier="fold_ids",
                    partition=self.partition,
                    model_name=self.model_name,
                    version=self.version,
                    fid=fid
                )
            ).astype(np.float64).tolist() for fid in range(self.config.num_folds)
        ]

        hpge_dataset = self.file_db.build_file(
            tier="training",
            partition=self.partition,
            version=self.dataset_version,
            filename=self.data_config["hpge_dataset"]
        )
        # load into RAM
        hpge_dataset = open_memmap(
            filename=hpge_dataset,
            mode="r"
        ).copy()

        lar_datasets = [
            self.file_db.build_file(
            tier="training",
            partition=self.partition,
            version=self.dataset_version,
            filename=self.data_config["lar_datasets"][i]
        ) for i in range(2) if i==1
        ]
        lar_datasets = [scipy.sparse.load_npz(path) for path in lar_datasets]

        self.fid_value = mp.Value("i", 0)
        dataset = ParallelBootstrappedKFoldLArListDataset(
            lar_data_lengths=[dataset.shape[0] for dataset in lar_datasets],
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.batch_size,
            hpge_feats_mean=self.config.hpge_feats_mean,
            hpge_feats_std=self.config.hpge_feats_std,
            sipm_pe_scale=self.config.sipm_pe_scale,
            test_folds=test_folds,
            mode=2,
            fold_id=self.fid_value
        )

        collate_fn = NRECCollateFn(
            cls_placeholder_id=self.config.cls_placeholder_id,
            has_cls=True,#self.config.deep_supervision!=1,
            sipm_unbinned_pe=self.config.sipm_unbinned_pe==1,
            cuda_device=self.device
        )
        worker_init_fn = partial(
            ParallelKFoldBootstrap_worker_init_fn,
            hpge_dataset,
            lar_datasets
        )
        self.phy_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

        path = self.file_db.build_file(
            tier="training",
            partition=self.partition,
            version=self.dataset_version,
            filename="classical_classifier_phy.npy"
        )
        self.classical_classifier_phy = np.load(path).astype(bool)

    def _init_ev_ep_null_dataloader(self):
        lar_dataset = self.file_db.build_file(
            tier="inference_dataset",
            partition=self.partition,
            version=self.dataset_version,
            filename=self.data_config["ev_ep_null"]
        )
        lar_dataset = scipy.sparse.load_npz(lar_dataset)

        dataset = ParallelBootstrappedKFoldLArListDataset(
            lar_data_lengths=[lar_dataset.shape[0]],
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.batch_size,
            sipm_pe_scale=self.config.sipm_pe_scale,
            mode=3
        )
        collate_fn = NRECCollateFn(
            cls_placeholder_id=self.config.cls_placeholder_id,
            sipm_unbinned_pe=self.config.sipm_unbinned_pe==1,
            cuda_device=self.device
        )
        worker_init_fn = partial(
            ParallelKFoldBootstrap_worker_init_fn,
            None,
            [lar_dataset]
        )
        self.ev_ep_null_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

        path = self.file_db.build_file(
            tier="inference_dataset",
            partition=self.partition,
            version=self.dataset_version,
            filename="classical_classifier_rc_ev_ep.npy"
        )
        self.classical_classifier_ev_ep = np.load(path).astype(bool)

    def _init_glob_null_dataloader(self):
        lar_dataset = self.file_db.build_file(
            tier="inference_dataset",
            partition=self.partition,
            version=self.dataset_version,
            filename=self.data_config["glob_null"]
        )
        lar_dataset = scipy.sparse.load_npz(lar_dataset)

        dataset = ParallelBootstrappedKFoldLArListDataset(
            lar_data_lengths=[lar_dataset.shape[0]],
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.batch_size,
            sipm_pe_scale=self.config.sipm_pe_scale,
            mode=3
        )
        collate_fn = NRECCollateFn(
            cls_placeholder_id=self.config.cls_placeholder_id,
            sipm_unbinned_pe=self.config.sipm_unbinned_pe==1,
            cuda_device=self.device
        )
        worker_init_fn = partial(
            ParallelKFoldBootstrap_worker_init_fn,
            None,
            [lar_dataset]
        )
        self.glob_null_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

        path = self.file_db.build_file(
            tier="inference_dataset",
            partition=self.partition,
            version=self.dataset_version,
            filename="classical_classifier_glob.npy"
        )
        self.classical_classifier_glob = np.load(path).astype(bool)

    def clean_state_dict(self, state_dict):
        cleaned_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('_orig_mod.', '')
            cleaned_dict[cleaned_key] = value
        return cleaned_dict

    def load_ensemble(self, fid: int):
        lar_detector_coords, hpge_detector_coords = decode_geom(
            self.file_db.build_file(
                tier="dataset",
                partition=self.partition,
                filename="detector_positions.yaml"
            ), self.config
        )

        self.ensemble = nn.ModuleList()
        for bid in range(self.config.num_bootstraps_per_fold):
            model = NREC(
                lar_detector_coords=lar_detector_coords,
                hpge_detector_coords=hpge_detector_coords,
                config=self.config,
                device=self.device
            ).to(dtype=torch.float32, device=self.device)

            cp = self.file_db.build_file(
                tier="models",
                partition=self.partition,
                model_name=self.model_name,
                version=self.version,
                fid=fid,
                bid=bid
            )
            cp = torch.load(cp, weights_only=True, map_location=self.device)["model"]
            cp = self.clean_state_dict(cp)
            model.load_state_dict(cp, strict=True)
            # model = torch.compile(model, dynamic=True)
            self.ensemble.append(model)
        
        self.ensemble.eval()

    def _select_hpge_embeddings(
        self,
        e_hpge: Tensor, # (N_packed, D)
        ge_b_idx: Tensor, # (N_packed,)
        ge_f_idx: Tensor, # (N_packed,)
        ge_cu_seqlens: Tensor # (B + 1,)
    ):
        sos_pos = ge_cu_seqlens[:-1].to(torch.long) # (B,)
        last_pos = ge_cu_seqlens[1:].to(torch.long) - 1 # (B,)

        if self.max_t < 0:
            selected_pos = last_pos

        elif self.max_t == 0:
            selected_pos = sos_pos

        else:
            # Select the rightmost observed raw HPGe feature with feature index < max_t.
            selected_pos = sos_pos.clone()

            valid_pos = (ge_f_idx < self.max_t).nonzero(as_tuple=True)[0]

            if valid_pos.numel() > 0:
                valid_b = ge_b_idx.index_select(0, valid_pos)
                selected_pos[valid_b] = valid_pos

        selected_t = torch.empty_like(selected_pos)

        is_sos = selected_pos == sos_pos
        selected_t[is_sos] = 0
        selected_t[~is_sos] = ge_f_idx.index_select(0, selected_pos[~is_sos]) + 1

        selected_e = e_hpge.index_select(0, selected_pos)
        return selected_e, selected_t

    def model_forward(self, model: NREC, lar, hpge):
        (_, t_idx, s_idx, v_val, cu_seqlens, max_seqlen, _) = lar
        t_idx = t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        s_idx = s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        v_val = v_val.to(device=self.device, non_blocking=True).to(dtype=torch.float32) if v_val is not None else v_val
        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        max_seqlen = int(max_seqlen)

        (ge_b_idx, f_idx, f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths, _) = hpge
        if f_idx is not None:
            ge_b_idx = ge_b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            f_idx = f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            f_vals = f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32)
            ge_cu_seqlens = ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            ge_lengths = ge_lengths.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            ge_max_seqlen = int(ge_max_seqlen)

        e_lar, e_hpge = model(
            t_idx=t_idx,
            s_idx=s_idx,
            v_val=v_val,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            f_idx=f_idx,
            f_vals=f_vals,
            ge_cu_seqlens=ge_cu_seqlens,
            ge_max_seqlen=ge_max_seqlen
        )

        if e_hpge is None:
            selected_t = None
        else:
            if self.config.deep_supervision == 1:
                e_hpge, selected_t = self._select_hpge_embeddings(
                    e_hpge=e_hpge,
                    ge_b_idx=ge_b_idx,
                    ge_f_idx=f_idx,
                    ge_cu_seqlens=ge_cu_seqlens
                )
            else:
                # For non-deep-supervision, log the rightmost available raw feature id
                last_pos = ge_cu_seqlens[1:].to(torch.long) - 1
                selected_t = f_idx.index_select(0, last_pos)

        return e_lar, e_hpge, selected_t

    def ensemble_forward(self, lar, hpge):
        e_lar = []
        e_hpge = []
        logits = []
        selected_t = None

        for model in self.ensemble:
            e_lar_, e_hpge_, selected_t_ = self.model_forward(model, lar, hpge)

            if e_hpge_ is None:
                e_lar.append(e_lar_.unsqueeze(0))
            else:
                if selected_t is None:
                    selected_t = selected_t_

                logits_ = (e_lar_ * e_hpge_).sum(dim=-1, keepdim=False) * self.inv_temp
                logits.append(logits_.unsqueeze(0))
                e_hpge.append(e_hpge_.unsqueeze(0))

        if len(e_hpge) == 0:
            return torch.cat(e_lar, dim=0) # (n_ensemble, B, D)
        
        logits = torch.cat(logits, dim=0) # (n_ensemble, B)
        dlogits = torch.var(logits, dim=0, correction=0)
        logits = logits.mean(dim=0)

        e_hpge = torch.cat(e_hpge, dim=0)

        return logits, dlogits, e_hpge, selected_t

    def _get_null_buffer(self, dataloader, classical_classifier):
        buffer = []
        classical = []
        for (lar, hpge), indices in dataloader:
            e_lar = self.ensemble_forward(lar, hpge)
            buffer.append(e_lar)
            classical.append(classical_classifier[indices])
        classical = np.concatenate(classical, axis=0)

        return buffer, classical # (n_ensemble, N_null, D), (N_null,)
    
    def _set_null_buffers(self):
        self.ev_ep_null_buffer, _ = self._get_null_buffer(self.ev_ep_null_dataloader, self.classical_classifier_ev_ep)
        self.glob_null_buffer, self.classical_classifier_glob_buffer = self._get_null_buffer(self.glob_null_dataloader, self.classical_classifier_glob)
        self.classical_classifier_glob_buffer = torch.from_numpy(self.classical_classifier_glob_buffer).to(device=self.device, dtype=torch.bool)

    def unpack_hpge_nrec_data(
        self,
        b_all: Tensor,
        f_all: Tensor,
        v_all: Tensor,
        lengths: Tensor
    ):
        b_all = b_all.to(torch.long)
        f_all = f_all.to(torch.long)
        B = lengths.numel()
        x = torch.full((B, self.config.hpge_num_features + 1), float("nan"), device=v_all.device, dtype=v_all.dtype) # NOTE: for now, +1 is hardcoded bcs pid is not included

        mask = (f_all != self.config.cls_placeholder_id) # ignore CLS tokens
        x[b_all[mask], f_all[mask]] = v_all[mask]

        return x

    @torch.no_grad()
    def infer_fold(self, fid: int):
        self.load_ensemble(fid)
        self._set_null_buffers()
        self.phy_dataloader.dataset.set_fold_id(fid)

        for (lar, hpge), indices in self.phy_dataloader:
            # Calculate the test statistic of each event
            logits, dlogits, e_hpge, selected_t = self.ensemble_forward(lar, hpge)  # (B,), (B,), (n_ensemble, B, D), (B,)

            # Calculate the evidence test statistic under the null for each event
            null_logits = []
            for e_null in self.ev_ep_null_buffer:
                null_logits.append(
                    torch.einsum('ebd,erd->ebr', e_hpge, e_null) * self.inv_temp # (n_ensemble, B, B_rc)
                )
            null_logits = torch.cat(null_logits, dim=-1) # (n_ensemble, B, N_null)
            dnull_logits = torch.var(null_logits, dim=0, correction=0) # (B, N_null)
            null_logits = null_logits.mean(dim=0) # (B, N_null)

            # Calculate the evidence test statistic under the global null for each event
            glob_null_logits = []
            for e_null in self.glob_null_buffer:
                glob_null_logits.append(
                    torch.einsum('ebd,erd->ebr', e_hpge, e_null) * self.inv_temp # (n_ensemble, B, B_rc)
                )

            glob_null_logits = torch.cat(glob_null_logits, dim=-1) # (n_ensemble, B, N_glob_null)
            dglob_null_logits = torch.var(glob_null_logits, dim=0, correction=0) # (B, N_glob_null)
            glob_null_logits = glob_null_logits.mean(dim=0) # (B, N_glob_null)

            N_rc_data = null_logits.shape[1]
            # Evidence p-value
            p_val = ((null_logits >= logits.reshape(-1, 1)).float().sum(dim=-1) + 1) / (N_rc_data + 1) # (B,)
            # Epistemic p-value
            p_val_ep = ((dnull_logits >= dlogits.reshape(-1, 1)).float().sum(dim=-1) + 1) / (N_rc_data + 1) # (B,)

            # Calculate the evidence and epistemic p-val under the global null (for t_global calibration)
            sorted_null = null_logits.sort(dim=-1).values # (B, N_null)
            sorted_dnull_logits = dnull_logits.sort(dim=-1).values # (B, N_null)

            # move to cpu (not used anymore for computations)
            null_logits = null_logits.cpu().numpy()
            dnull_logits = dnull_logits.cpu().numpy()

            glob_null_logits_len = glob_null_logits.shape[1]
            num_iters = math.ceil(glob_null_logits_len / self.batch_size)
    
            glob_null_p_val = []
            glob_null_p_val_ep = []
            for it in range(num_iters):
                start = it * self.batch_size
                stop = min((it + 1) * self.batch_size, glob_null_logits_len)

                batch = glob_null_logits[:, start:stop] # (B, local_batch_size)
                idx = torch.searchsorted(sorted_null, batch, right=False)
                glob_null_p_val.append(((N_rc_data - idx).float() + 1) / (N_rc_data + 1))

                batch = dglob_null_logits[:, start:stop]
                idx = torch.searchsorted(sorted_dnull_logits, batch, right=False)
                glob_null_p_val_ep.append(((N_rc_data - idx).float() + 1) / (N_rc_data + 1))

            glob_null_p_val = torch.cat(glob_null_p_val, dim=-1) # (B, N_glob_null)
            glob_null_p_val_ep = torch.cat(glob_null_p_val_ep, dim=-1) # (B, N_glob_null)

            is_lar_vetoed = self.classical_classifier_phy[indices]

            if self.data_config["alpha_epistemic"] > 0:
                # Global score calculation of each event
                is_lar_vetoed_t = torch.from_numpy(is_lar_vetoed).to(device=self.device, dtype=torch.bool)
                flag_classical = (p_val_ep <= self.data_config["alpha_epistemic"]) & is_lar_vetoed_t

                eps = 1e-6
                global_score = p_val.clone()
                global_score[flag_classical] = (
                    -1.0
                    + (1.0 - 2.0 * eps) * (p_val_ep[flag_classical] / self.data_config["alpha_epistemic"])
                    + eps * p_val[flag_classical]
                ) # always reject untrustworthy scores that do not pass the classical classifier

                # Global score calculation of global null
                flag_classical = (glob_null_p_val_ep <= self.data_config["alpha_epistemic"]) & self.classical_classifier_glob_buffer
                null_global_score = glob_null_p_val.clone()
                null_global_score[flag_classical] = (
                    -1.0
                    + (1.0 - 2.0 * eps) * (glob_null_p_val_ep[flag_classical] / self.data_config["alpha_epistemic"])
                    + eps * glob_null_p_val[flag_classical]
                )

                # Global p-value
                N_glob_null = null_global_score.shape[-1]
                p_val_glob = ((null_global_score <= global_score.reshape(-1, 1)).sum(dim=-1) + 1) / (N_glob_null + 1) # (B,)

                # Calculate the global p-val under the global null (for sanity check)
                sorted_null = null_global_score.sort(dim=-1).values # (B, N_glob_null)

                glob_null_score_len = null_global_score.shape[1]
                num_iters = math.ceil(glob_null_score_len / self.batch_size)
        
                glob_null_p_val_glob = []
                for it in range(num_iters):
                    start = it * self.batch_size
                    stop = min((it + 1) * self.batch_size, glob_null_score_len)

                    batch = null_global_score[:, start:stop] # (B, local_batch_size)
                    idx = torch.searchsorted(sorted_null, batch, right=True)
                    glob_null_p_val_glob.append(idx.float() / glob_null_score_len)

                glob_null_p_val_glob = torch.cat(glob_null_p_val_glob, dim=-1) # (B, N_glob_null)

            # retrieve HPGe observables
            (b_idx, f_idx, f_vals, _, _, geds_lengths, _) = hpge
            # NOTE: this part is hardcoded based on the data ordering from utils/create_base_dataset.py
            geds_features = self.unpack_hpge_nrec_data(b_idx, f_idx, f_vals, geds_lengths) # (B, H)
            geds_features = geds_features.cpu().numpy().astype(np.float32)

            gid = geds_features[:, 0]
            gid = gid * self.config.hpge_feats_std[0] + self.config.hpge_feats_mean[0]

            energy = geds_features[:, 1]
            energy = energy * self.config.hpge_feats_std[1] + self.config.hpge_feats_mean[1]

            drift_time = geds_features[:, 2]
            drift_time = drift_time * self.config.hpge_feats_std[2] + self.config.hpge_feats_mean[2]

            aoe = geds_features[:, 3]
            aoe = aoe * self.config.hpge_feats_std[3] + self.config.hpge_feats_mean[3]

            lq = geds_features[:, 4]
            lq = lq * self.config.hpge_feats_std[4] + self.config.hpge_feats_mean[4]

            table_size = len(indices)
            if self.data_config["alpha_epistemic"] > 0:
                lgdo_table = Table(
                    size=table_size,
                    col_dict={
                        "evt_idx": Array(np.asarray(indices).astype(np.float32)),
                        "t_used": Array(selected_t.cpu().numpy().astype(np.float32), dtype=np.float32),
                        "g_id": Array(gid, dtype=np.float32),
                        "energy": Array(energy, dtype=np.float32),
                        "drift_time": Array(drift_time, dtype=np.float32),
                        "aoe": Array(aoe, dtype=np.float32),
                        "lq": Array(lq, dtype=np.float32),

                        "is_lar_vetoed": Array(is_lar_vetoed.astype(np.bool_), dtype=np.bool_),

                        # evidence and epistemic test statistics and p-values
                        "t_epistemic": Array(dlogits.cpu().numpy(), dtype=np.float32),
                        "t_evidence": Array(logits.cpu().numpy(), dtype=np.float32),
                        "p_evidence": Array(p_val.cpu().numpy(), dtype=np.float32),
                        "p_epistemic": Array(p_val_ep.cpu().numpy(), dtype=np.float32),

                        # sanity checks (calibrationg null_t_evidence and null_t_epistemic with itself) --> uniformly distributed in (0, 1]
                        "null_t_epistemic": Array(dnull_logits, dtype=np.float32),
                        "null_t_evidence": Array(null_logits, dtype=np.float32),

                        # global null evidence and epistemic test statistics and p-values (for global test statistic calibration)
                        "glob_null_t_epistemic": Array(dglob_null_logits.cpu().numpy(), dtype=np.float32),
                        "glob_null_t_evidence": Array(glob_null_logits.cpu().numpy(), dtype=np.float32),
                        "glob_null_p_epistemic": Array(glob_null_p_val_ep.cpu().numpy(), dtype=np.float32),
                        "glob_null_p_evidence": Array(glob_null_p_val.cpu().numpy(), dtype=np.float32),

                        # global test statistic and p-value
                        "t_global": Array(global_score.cpu().numpy(), dtype=np.float32),
                        "p_global": Array(p_val_glob.cpu().numpy(), dtype=np.float32),

                        # global null global test statistic and p-value for calibration and sanity check
                        "glob_null_t_global": Array(null_global_score.cpu().numpy(), dtype=np.float32),
                        "glob_null_p_global": Array(glob_null_p_val_glob.cpu().numpy(), dtype=np.float32)
                    }
                )
            else:
                lgdo_table = Table(
                    size=table_size,
                    col_dict={
                        "evt_idx": Array(np.asarray(indices).astype(np.float32)),
                        "t_used": Array(selected_t.cpu().numpy().astype(np.float32), dtype=np.float32),
                        "g_id": Array(gid, dtype=np.float32),
                        "energy": Array(energy, dtype=np.float32),
                        "drift_time": Array(drift_time, dtype=np.float32),
                        "aoe": Array(aoe, dtype=np.float32),
                        "lq": Array(lq, dtype=np.float32),

                        "is_lar_vetoed": Array(is_lar_vetoed.astype(np.bool_), dtype=np.bool_),

                        # evidence and epistemic test statistics and p-values
                        "t_epistemic": Array(dlogits.cpu().numpy(), dtype=np.float32),
                        "t_evidence": Array(logits.cpu().numpy(), dtype=np.float32),
                        "p_evidence": Array(p_val.cpu().numpy(), dtype=np.float32),
                        "p_epistemic": Array(p_val_ep.cpu().numpy(), dtype=np.float32),

                        # sanity checks (calibrationg null_t_evidence and null_t_epistemic with itself) --> uniformly distributed in (0, 1]
                        "null_t_epistemic": Array(dnull_logits, dtype=np.float32),
                        "null_t_evidence": Array(null_logits, dtype=np.float32),
                        "glob_null_p_epistemic": Array(glob_null_p_val_ep.cpu().numpy(), dtype=np.float32),
                        "glob_null_p_evidence": Array(glob_null_p_val.cpu().numpy(), dtype=np.float32),

                        # global test statistic and p-value
                        "p_global": Array(p_val.cpu().numpy(), dtype=np.float32)
                    }
                )

            path = self.file_db.build_file(
                tier="inference",
                partition=self.partition,
                version=self.dataset_version,
                model_name=self.model_name,
                model_version=self.version
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            lh5.write(
                lgdo_table,
                "phy/inf",
                path,
                n_rows=table_size,
                wo_mode="append"
            )

    def infer(self):
        path = self.file_db.build_file(
            tier="inference",
            partition=self.partition,
            version=self.dataset_version,
            model_name=self.model_name,
            model_version=self.version
        )
        if os.path.isfile(path):
            print(f'{path} exists. Deleting...')
            os.remove(path)

        for fid in range(self.config.num_folds):
            self.infer_fold(fid)
            print('Done with fid{fid:02d}'.format(fid=fid))

def main(
    experiment: str,
    partition: str,
    model_name: str,
    version: str,
    train_dataset_version: str,
    dataflow_dir: str,
    batch_size: int,
    cache_dir: str,
    max_t: int = -1
):
    _, _, device = _init_torch(cache_dir)

    file_db = FileDB(
        working_dir=dataflow_dir,
        experiment=experiment
    )

    calibrator = NRECCalibrator(
        file_db=file_db,
        partition=partition,
        model_name=model_name,
        version=version,
        dataset_version=train_dataset_version,
        batch_size=batch_size,
        device=device,
        max_t=max_t
    )

    calibrator.infer()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibration started")
    parser.add_argument("experiment", type=str, help="Name of the experiment")
    parser.add_argument("partition", type=str, help="partition name")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("version", type=str, help="Model version")
    parser.add_argument("train_dataset_version", type=str, help="Traning dataset version")
    parser.add_argument("dataflow_dir", type=str, help="Directory of the dataflow")
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("cache_dir", type=str, help="Directory to store numba, torch.inductor and triton cache")
    parser.add_argument(
        "--max_t",
        type=int,
        default=-1,
        help=(
            "HPGe prefix cutoff for deep-supervision inference. "
            "-1 uses the full/rightmost available prefix; "
            "0 uses the SOS/empty prefix; "
            "k>0 uses the rightmost observed raw HPGe feature with feature index < k."
        ),
    )
    args = parser.parse_args()

    main(
        args.experiment,
        args.partition,
        args.model_name,
        args.version,
        args.train_dataset_version,
        args.dataflow_dir,
        args.batch_size,
        args.cache_dir,
        args.max_t
    )
