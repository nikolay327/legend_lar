import os
import math
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import scipy

import torch
import torch._inductor.config as cfg
cfg.autotune_local_cache = False

from torch import Tensor
import torch.nn.functional as F
from apex.optimizers import FusedMixedPrecisionLamb

from legend_lar.model import NREC
from legend_lar.utils import FileDB, NRECConfig, _initialize_configs, _init_torch, decode_geom

from legend_lar.kfold_ensemble.base import TrainerBase

class NRECTrainer(TrainerBase):
    def __init__(
        self,
        file_db: FileDB,
        partition: str,
        model_name: str,
        version: str,
        config: NRECConfig,
        hpge_dataset: np.ndarray,
        lar_datasets: scipy.sparse._csr.csr_matrix,
        lar_detector_coords: Tensor,
        hpge_detector_coords: Tensor,
        rank: int,
        world_size: int,
        device: str | int,
        tmp_dir: str
    ):
        super(NRECTrainer, self).__init__(
            file_db=file_db,
            partition=partition,
            model_name=model_name,
            version=version,
            config=config,
            hpge_dataset=hpge_dataset,
            lar_datasets=lar_datasets,
            rank=rank,
            world_size=world_size,
            device=device,
            tmp_dir=tmp_dir
        )

        self.lar_detector_coords = lar_detector_coords
        self.hpge_detector_coords = hpge_detector_coords

    def _reinit_model(self, fid: int, bid: int):
        self.model = NREC(
            lar_detector_coords=self.lar_detector_coords,
            hpge_detector_coords=self.hpge_detector_coords,
            config=self.config,
            device=self.device
        ).to(dtype=torch.float32, device=self.device)
        self.model_initiator.reinit_(
            self.model,
            self.get_model_reinit_seed(fid, bid)
        )

    def _init_optimizer(self, model_opt_state = None):
        self.model_opt = FusedMixedPrecisionLamb(
            params=self.model.parameters(),
            lr = self.config.lr_model,
            betas=self.config.betas_model,
            weight_decay=self.config.weight_decay
        )
        if model_opt_state is not None:
            self.model_opt.load_state_dict(model_opt_state)

    def reset_model_and_optimizer(self, fid: int, bid: int, start_from_epoch: int = 1):
        self._reinit_model(fid, bid)

        if start_from_epoch == 1:
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
            self._init_optimizer()
            torch.cuda.empty_cache()
            return

        def clean_state_dict(state_dict):
            cleaned_dict = {}
            for key, value in state_dict.items():
                cleaned_key = key.replace('_orig_mod.', '')
                cleaned_dict[cleaned_key] = value
            return cleaned_dict

        last_epoch = start_from_epoch - 1
        save_path = self.file_db.build_file(
            tier="models",
            partition=self.partition,
            model_name=self.model_name,
            version=self.version,
            fid=fid,
            bid=bid
        )

        cp = torch.load(save_path, map_location=self.device)
        epoch = cp["epoch"]
        if epoch != last_epoch:
            raise ValueError(f'Variable last_epoch with value ({last_epoch}) is different from the last saved checkpoint ({epoch})')
        else:
            self.last_saved_epoch = last_epoch

        self.model.load_state_dict(clean_state_dict(cp["model"]), strict=True)
        self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)

        self._init_optimizer(cp["model_opt"])

        self.train_loss = cp["train_loss"]
        self.val_loss = cp["val_loss"]

    def calculate_loss(self, logits: Tensor, K: int):
        logK = math.log(K)
        loggamma = 0.0 if self.config.gamma == 1 else math.log(self.config.gamma)

        logits = torch.cat(
            [
                torch.full((len(logits), 1), logK, device=logits.device),
                logits + loggamma
            ],
            dim=1
        ) # (B, K+1)

        # y = 0
        logits_y0 = logits[:-K]
        target_y0 = torch.zeros(len(logits_y0), dtype=torch.long, device=logits.device)
        loss_y0 = F.cross_entropy(logits_y0, target_y0)

        # y != 0
        logits_y_not0 = logits[-K:]
        target_y_not0 = torch.arange(len(logits_y_not0), device=logits.device) + 1
        loss_y_not0 = F.cross_entropy(logits_y_not0, target_y_not0)

        loss = 1.0/(1.0+self.config.gamma)*loss_y0 + self.config.gamma/(1.0+self.config.gamma)*loss_y_not0
        return loss
    
    def forward_batch(
        self,
        f_idx: Tensor, # (N_valid,)
        f_vals: Tensor, # (N_valid,)
        ge_cu_seqlens: Tensor, # (B/2+1,)
        ge_max_seqlen: int,
        t_idx: Tensor, # (N,)
        s_idx: Tensor, # (N,)
        cu_seqlens: Tensor, # (B+1,)
        max_seqlen: int
    ):
        e_lar, e_hpge = self.model(
            f_idx=f_idx,
            f_vals=f_vals,
            ge_cu_seqlens=ge_cu_seqlens,
            ge_max_seqlen=ge_max_seqlen,
            t_idx=t_idx,
            s_idx=s_idx,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        e_lar = F.normalize(e_lar, p=2, dim=-1) # (B, D)
        e_hpge = F.normalize(e_hpge, p=2, dim=-1) # (B / 2, D)

        K = len(ge_cu_seqlens) - 1
        logits = (e_lar @ e_hpge.t()) / self.config.temperature # (B, B / 2)
        loss = self.calculate_loss(logits, K)
        return loss

    def train_batch(
        self, *args, **kwargs
    ):

        loss = self.forward_batch(*args, **kwargs)
        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        return loss.detach().item()

    def train_epoch(self):
        loss = 0.
        n_step = 0
        for lar, hpge, _ in self.dataloader:
            (_, t_idx, s_idx, cu_seqlens, max_seqlen, _) = lar
            (_, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, _) = hpge

            loss_ = self.train_batch(
                f_idx=ge_f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                f_vals=ge_f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                ge_cu_seqlens=ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                ge_max_seqlen=int(ge_max_seqlen),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                max_seqlen=int(max_seqlen)
            )

            loss += loss_
            n_step += 1

        n_step = 1 / n_step
        self.train_loss.append(loss * n_step)

    def val_batch(self):
        return

    @torch.no_grad()
    def val_epoch(self):
        loss = 0.
        n_step = 0
        for lar, hpge, _ in self.dataloader:
            (_, t_idx, s_idx, cu_seqlens, max_seqlen, _) = lar
            (_, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, _) = hpge

            loss_ = self.forward_batch(
                f_idx=ge_f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                f_vals=ge_f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                ge_cu_seqlens=ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                ge_max_seqlen=int(ge_max_seqlen),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                max_seqlen=int(max_seqlen)
            )

            loss += loss_
            n_step += 1

        n_step = 1 / n_step
        self.val_loss.append(loss * n_step)


def train(
    experiment: str,
    partition: str,
    model_name: str,
    version: str,
    dataflow_dir: str,
    base_cfg_name: str,
    tmp_dir: str
):
    local_rank, rank, world_size, device = _init_torch()

    file_db = FileDB(
        working_dir=dataflow_dir,
        experiment=experiment
    )

    base_cfg = file_db.build_file(
        tier="base_configs",
        filename=base_cfg_name
    )

    model_cfg = file_db.build_file(
        tier="model_config",
        partition=partition,
        model_name=model_name,
        version=version
    )

    model_cfg, data_config = _initialize_configs(
        config_obj=NRECConfig(),
        config_path=model_cfg,
        base_config=base_cfg
    )

    hpge_dataset = file_db.build_file(
        tier="training",
        partition=partition,
        filename=data_config["hpge_dataset"]
    )
    # load into RAM
    hpge_dataset = open_memmap(
        filename=hpge_dataset,
        mode="r"
    ).copy()

    lar_datasets = [
        file_db.build_file(
        tier="training",
        partition=partition,
        filename=data_config["lar_datasets"][i]
    ) for i in range(2)
    ]
    lar_datasets = [scipy.sparse.load_npz(path) for path in lar_datasets]

    # decode detector positions
    det_geom_and_subpart = file_db.build_file(
        tier="dataset",
        partition=partition,
        filename="detector_positions.yaml"
    )
    lar_detector_coords, hpge_detector_coords = decode_geom(det_geom_and_subpart, model_cfg)

    trainer = NRECTrainer(
        file_db=file_db,
        partition=partition,
        model_name=model_name,
        version=version,
        config=model_cfg,
        hpge_dataset=hpge_dataset,
        lar_datasets=lar_datasets,
        lar_detector_coords=lar_detector_coords,
        hpge_detector_coords=hpge_detector_coords,
        rank=rank,
        world_size=world_size,
        device=device,
        tmp_dir=tmp_dir
    )

    to_be_trained = np.ones((model_cfg.num_folds, model_cfg.num_bootstraps_per_fold))
    to_be_trained = np.stack(to_be_trained.nonzero()).T
    start_from_epoch = np.ones(len(to_be_trained)).reshape(-1, 1)
    to_be_trained = np.concatenate((to_be_trained, start_from_epoch), axis=-1)

    remove_id = []
    for path in os.listdir(tmp_dir):
        meta = path.split("_")
        if len(meta) == 3:
            meta = np.array(meta).astype(int)
            global_id = model_cfg.num_folds * meta[0] + meta[1]
            if meta[-1] != 1:
                to_be_trained[global_id, -1] = meta[-1]
        elif len(meta) == 4:
            meta = np.array(meta[:3]).astype(int)
            global_id = model_cfg.num_folds * meta[0] + meta[1]
            remove_id.append(remove_id)

    to_be_trained = np.delete(to_be_trained, remove_id, axis=0)

    shard_size = len(to_be_trained) // world_size
    if rank == (world_size - 1):
        to_be_trained = to_be_trained[shard_size * rank:]
    else:
        to_be_trained = to_be_trained[shard_size * rank: shard_size * (rank + 1)]

    trainer.train(to_be_trained)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training started")
    parser.add_argument("experiment", type=str, help="Name of the experiment")
    parser.add_argument("partition", type=str, help="partition name")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("version", type=str, help="Model version")
    parser.add_argument("dataflow_dir", type=str, help="Directory of the dataflow")
    parser.add_argument("base_cfg_name", type=str, help="Name of the base json config")
    parser.add_argument("tmp_dir", type=str, help="Directory for storing temporary files")
    parser.add_argument("cache_dir", type=str, help="Directory to store numba, torch.inductor and triton cache")
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

    train(
        args.experiment,
        args.partition,
        args.model_name,
        args.version,
        args.dataflow_dir,
        args.base_cfg_name,
        args.tmp_dir
    )
