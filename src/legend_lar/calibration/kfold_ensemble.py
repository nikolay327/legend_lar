import os
import json
from glob import glob
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.amp import autocast

import numpy as np

from legend_lar.utils import ModelConfig, EvalConfig, _initialize_configs, _init_torch
from legend_lar.model import UnconditionalRatioEstimator, ConditionalRatioEstimator
from legend_lar.data import LArListDataset, CollateFn, worker_init_fn

from lgdo import Table, Array, lh5

class Evaluator:
    def __init__(
        self,
        unconditional_model: UnconditionalRatioEstimator,
        conditional_model: ConditionalRatioEstimator,
        unconditional_config: ModelConfig,
        conditional_config: ModelConfig,
        eval_config: EvalConfig,
        phy_dataloader: DataLoader,
        lar_dataloader: DataLoader,
        save_to: str,
        device: str | int
    ):
        self.device = device
        self.conditional_config = conditional_config
        self.unconditional_config = unconditional_config
        self.eval_config = eval_config

        self.unconditional_model = unconditional_model.to(device=self.device)
        self.conditional_model = conditional_model.to(device=self.device)

        self.phy_dataloader = phy_dataloader
        self.lar_dataloader = lar_dataloader

        self.rc_unconditional_logit_buffer = [] # stored on GPU
        self.rc_conditional_anchor_buffer = [] # stored on GPU

        self.save_to = save_to

    def clean_state_dict(self, state_dict):
        cleaned_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('_orig_mod.', '')
            cleaned_dict[cleaned_key] = value
        return cleaned_dict

    def load_checkpoint(self):
        assert self.eval_config.conditional_cp_id > 0
        cp = torch.load(f'{self.conditional_config.save_to}/checkpoint_{self.eval_config.conditional_cp_id}.pt', map_location=self.device)
        self.conditional_model.load_state_dict(self.clean_state_dict(cp["model"]), strict=True)
        self.conditional_model = torch.compile(self.conditional_model, dynamic=True)

        assert self.eval_config.unconditional_cp_id > 0
        cp = torch.load(f'{self.unconditional_config.save_to}/checkpoint_{self.eval_config.unconditional_cp_id}.pt', map_location=self.device)
        self.unconditional_model.load_state_dict(self.clean_state_dict(cp["model"]), strict=True)
        self.unconditional_model = torch.compile(self.unconditional_model, dynamic=True)

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
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            unconditional_logit = self.unconditional_model(
                b_idx=b_idx,
                t_idx=t_idx,
                s_idx=s_idx,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                lengths=lengths,
            )
            conditional_logit, (anchors, e_hpge) = self.conditional_model(
                g=g,
                E=E,
                b_idx=b_idx,
                t_idx=t_idx,
                s_idx=s_idx,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                lengths=lengths
            )

        return unconditional_logit.squeeze(-1), conditional_logit.squeeze(-1), anchors, e_hpge

    @torch.no_grad()
    def set_rc_buffers(self):
        self.lar_dataloader.dataset.set_epoch(1)
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, _ in self.lar_dataloader:
            unconditional_logit, _, anchors, _ = self.batch_forward(
                g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True)
            )
            unconditional_logit = unconditional_logit.to(torch.float32) # (B_rc,)
            anchors = anchors.to(torch.float32) # (B_rc, D)

            self.rc_unconditional_logit_buffer.append(unconditional_logit)
            self.rc_conditional_anchor_buffer.append(anchors)
        self.rc_unconditional_logit_buffer = torch.cat(self.rc_unconditional_logit_buffer, dim=0) # (B_rc,)

        lgdo_table = Table(
            size=len(self.rc_unconditional_logit_buffer),
            col_dict={
                "null_unconditional_logits": Array(self.rc_unconditional_logit_buffer.cpu().numpy(), dtype=np.float32)
            }
        )
        lh5.write(
            lgdo_table,
            "null/evt", # NOTE: for now hardcoded
            self.save_to,
            n_rows=len(self.rc_unconditional_logit_buffer),
            wo_mode="append"
        )

    @torch.no_grad()
    def evaluate_phys(self):
        self.phy_dataloader.dataset.set_epoch(1)
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, _ in self.phy_dataloader:
            unconditional_logit, conditional_logit, _, e_hpge = self.batch_forward(
                g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True)
            )

            unconditional_logit = unconditional_logit.to(torch.float32) # (B,)
            conditional_logit = conditional_logit.to(torch.float32) # (B,)
            e_hpge = e_hpge.to(torch.float32) # (B, D)

            # Calculate conditional test statistic distribution under the null for each event
            null_conditional_logits = []
            for rc_conditional_anchor in self.rc_conditional_anchor_buffer:
                null_conditional_logits_ = e_hpge.reshape(-1, 1, self.conditional_config.hidden_size) * rc_conditional_anchor.reshape(1, -1, self.conditional_config.hidden_size) # (B, 1, D) * (1, B_rc, D) = (B, B_rc, D)
                null_conditional_logits_ = null_conditional_logits_.sum(dim=-1) / self.conditional_config.temperature # (B, B_rc)
                null_conditional_logits.append(null_conditional_logits_)
            null_conditional_logits = torch.cat(null_conditional_logits, dim=-1) # (B, N_rc_data)

            N_rc_data = null_conditional_logits.shape[1]
            N_tot = N_rc_data + self.eval_config.num_zero_pe_in_lar_ft + self.eval_config.num_high_pe_in_lar_ft
            q = ((1 - self.eval_config.alpha) * N_tot - self.eval_config.num_zero_pe_in_lar_ft) / N_rc_data
            q = float(max(0.0, min(1.0, q)))

            null_logits = null_conditional_logits + self.rc_unconditional_logit_buffer.reshape(1, -1) # (B, N_rc_data)
            logits = unconditional_logit + conditional_logit # (B,)

            cut = torch.quantile(null_logits, q, dim=-1) # (B,)
            is_accepted = (logits <= cut).to(torch.float32)

            # calibrated p-values
            p_val = ((null_logits >= logits.reshape(-1, 1)).float().sum(dim=-1) + self.eval_config.num_high_pe_in_lar_ft + 1) / (N_rc_data + self.eval_config.num_zero_pe_in_lar_ft + 1) # (B,)

            unconditional_logit = unconditional_logit.cpu().numpy() # (B,)
            conditional_logit = conditional_logit.cpu().numpy() # (B,)
            null_conditional_logits = null_conditional_logits.cpu().numpy() # (B, N_rc_data)
            cut = cut.cpu().numpy() # (B,)
            is_accepted = is_accepted.cpu().numpy() # (B,)
            p_val = p_val.cpu().numpy() # (B,)

            g_id = g.to(torch.float32).cpu().numpy() # (B,)
            energy = E.to(torch.float32).cpu().numpy() # (B,)

            table_size = len(energy)
            lgdo_table = Table(
                size=table_size,
                col_dict={
                    "g_id": Array(g_id, dtype=np.float32),
                    "energy": Array(energy, dtype=np.float32),

                    "unconditional_logit": Array(unconditional_logit, dtype=np.float32),
                    "conditional_logit": Array(conditional_logit, dtype=np.float32),

                    "null_conditional_logits": Array(null_conditional_logits, dtype=np.float32),
                    "cut": Array(cut, dtype=np.float32),
                    "is_accepted": Array(is_accepted, dtype=np.float32),
                    "p_val": Array(p_val, dtype=np.float32)
                }
            )
            lh5.write(
                lgdo_table,
                "phy/evt", # NOTE: for now hardcoded
                self.save_to,
                n_rows=table_size,
                wo_mode="append"
            )

    def evaluate(self):
        self.load_checkpoint()
        self.set_rc_buffers()
        self.evaluate_phys()

def initialize_configs_and_dataset(
    config_obj,
    wd,
    experiment,
    model_name,
    version,
    mmpd
):
    config, _, _ = _initialize_configs(
        config_obj=config_obj,
        wd=wd,
        experiment=experiment,
        model_name=model_name,
        version=version,
        mmpd=mmpd
    )

    return config

def evaluate(
    experiment: str,
    model_name: str,
    period: str,
    unconditional_version: str,
    conditional_version: str,
    working_dir: str,
    data_dir: str
):
    local_rank, rank, world_size, device = _init_torch()
    wd = Path(working_dir)
    mmpd = Path(data_dir)

    with open(str(wd / "trained" / experiment / model_name / "eval_config.json"), "r") as f:
        eval_cfg = json.load(f)
    
    eval_config = EvalConfig(
        unconditional_cp_id=eval_cfg["unconditional_cp_id"],
        conditional_cp_id=eval_cfg["conditional_cp_id"],
        local_batch_size=eval_cfg["local_batch_size"],
        calib_dataset_frac=eval_cfg["calib_dataset_frac"],
        num_zero_pe_in_lar_ft=eval_cfg["num_zero_pe_in_lar_ft"],
        num_high_pe_in_lar_ft=eval_cfg["num_high_pe_in_lar_ft"],
        alpha=eval_cfg["alpha"]
    )

    unconditional_config, _, _ = _initialize_configs(
        config_obj=ModelConfig(),
        wd=wd,
        experiment=experiment,
        model_name=model_name,
        version=unconditional_version,
        mmpd=mmpd
    )
    conditional_config, _, _ = _initialize_configs(
        config_obj=ModelConfig(),
        wd=wd,
        experiment=experiment,
        model_name=model_name,
        version=conditional_version,
        mmpd=mmpd
    )

    epoch_value = mp.Value("i", 0)
    phy_dataset = LArListDataset(
        hpge_path=str(mmpd / f'{eval_cfg["hpge_path"]}.npy'),
        lar_paths=[str(mmpd / f'{eval_cfg["phy_lar_path"]}.npz')],
        labels=[1],
        true_coincidence_label=1,
        hpge_energy_mean=conditional_config.hpge_energy_mean,
        hpge_energy_std=conditional_config.hpge_energy_std,
        prior=[1.0],
        train_val_test_fract=conditional_config.train_val_test_fract,
        local_batch_size=eval_config.local_batch_size,
        num_t_bins=conditional_config.num_sipm_t_bins,
        num_sipm_chs=conditional_config.num_sipms,
        rng_seed_for_split=conditional_config.rng_seed_for_split,
        times_of_mixing=conditional_config.times_of_mixing,
        global_rng_seed_for_sampling=conditional_config.global_rng_seed_for_sampling,
        epoch_value=epoch_value,
        shuffle=False,
        calib_mode=None,
        calib_dataset_frac=None
    )
    collate_fn = CollateFn(
        num_sipm_chs=conditional_config.num_sipms,
        true_coincidence_label=1,
        cuda_device=device
    )
    phy_dataset.set_mode(3) # anything > 2 will load the entire data
    phy_dataloader = DataLoader(
        dataset=phy_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    epoch_value_rc = mp.Value("i", 0)
    rc_dataset = LArListDataset(
        hpge_path=str(mmpd / f'{eval_cfg["hpge_path"]}.npy'),
        lar_paths=[str(mmpd / f'{eval_cfg["rc_lar_path"]}.npz')],
        labels=[0],
        true_coincidence_label=1,
        hpge_energy_mean=conditional_config.hpge_energy_mean,
        hpge_energy_std=conditional_config.hpge_energy_std,
        prior=[1.0],
        train_val_test_fract=conditional_config.train_val_test_fract,
        local_batch_size=eval_config.local_batch_size,
        num_t_bins=conditional_config.num_sipm_t_bins,
        num_sipm_chs=conditional_config.num_sipms,
        rng_seed_for_split=conditional_config.rng_seed_for_split,
        times_of_mixing=conditional_config.times_of_mixing,
        global_rng_seed_for_sampling=conditional_config.global_rng_seed_for_sampling,
        epoch_value=epoch_value_rc,
        shuffle=False,
        calib_mode=True,
        calib_dataset_frac=eval_config.calib_dataset_frac
    )
    collate_fn = CollateFn(
        num_sipm_chs=conditional_config.num_sipms,
        true_coincidence_label=1,
        cuda_device=device
    )
    rc_dataset.set_mode(2) # test / calibration mode
    rc_dataloader = DataLoader(
        dataset=rc_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    conditional_model = ConditionalRatioEstimator(config=conditional_config, device=device).to(torch.float32)
    unconditional_model = UnconditionalRatioEstimator(config=unconditional_config, device=device).to(torch.float32)

    os.makedirs(wd / "data" / experiment / "tier" / model_name / period, exist_ok=True)
    evaluator = Evaluator(
        unconditional_model=unconditional_model,
        conditional_model=conditional_model,
        unconditional_config=unconditional_config,
        conditional_config=conditional_config,
        eval_config=eval_config,
        phy_dataloader=phy_dataloader,
        lar_dataloader=rc_dataloader,
        save_to=str(wd / "data" / experiment / "tier" / model_name / period / "calibration.lh5"),
        device=device
    )
    evaluator.evaluate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation started")
    parser.add_argument("experiment", type=str)
    parser.add_argument("model_name", type=str, help="Name of the model being trained")
    parser.add_argument("period", type=str)
    parser.add_argument("unconditional_version", type=str)
    parser.add_argument("conditional_version", type=str)
    parser.add_argument("working_dir", type=str, help="Top-most dir of the training pipeline")
    parser.add_argument("data_dir", type=str, help="Directory the training data is saved under")
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

    evaluate(args.experiment, args.model_name, args.period, args.unconditional_version, args.conditional_version, args.working_dir, args.data_dir)