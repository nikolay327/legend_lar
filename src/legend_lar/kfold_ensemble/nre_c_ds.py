import os
import math

import numpy as np
from numpy.lib.format import open_memmap
import scipy

import torch
import torch._inductor.config as cfg
cfg.autotune_local_cache = False

from torch import Tensor
import torch.nn.functional as F

try:
    from apex.optimizers import FusedMixedPrecisionLamb
except ImportError:
    from bitsandbytes.optim import LAMB as FusedMixedPrecisionLamb

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

        self.train_prefix_losses = []
        self.val_prefix_losses = []

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
            # self.model = torch.compile(self.model)
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
        # self.model = torch.compile(self.model)

        self._init_optimizer(cp["model_opt"])

        self.train_loss = cp["train_loss"]
        self.val_loss = cp["val_loss"]
        self.train_prefix_losses = cp.get("train_prefix_losses", [])
        self.val_prefix_losses = cp.get("val_prefix_losses", [])
        self.best_val_loss = cp.get("best_val_loss", cp["val_loss"][-1])

    def calculate_loss(self, logits: Tensor, K: int):
        G = logits.shape[0]

        logits = torch.cat(
            [
                torch.full((G, 2 * K, 1), math.log(K), device=logits.device, dtype=logits.dtype),
                logits + (0.0 if self.config.gamma == 1 else math.log(self.config.gamma))
            ],
            dim=-1
        ) # (G, 2K, K+1)

        loss_y0 = F.cross_entropy(
            logits[:, :K].reshape(G * K, K + 1),
            torch.zeros(G * K, dtype=torch.long, device=logits.device)
        )

        loss_y_not0 = F.cross_entropy(
            logits[:, K:].reshape(G * K, K + 1),
            torch.arange(K, device=logits.device).unsqueeze(0).expand(G, K).reshape(G * K) + 1
        )

        return (
            1.0 / (1.0 + self.config.gamma) * loss_y0
            + self.config.gamma / (1.0 + self.config.gamma) * loss_y_not0
        )
    def calculate_deep_supervision_loss(
        self,
        e_lar: Tensor, # (2 * B_hpge, D)
        e_hpge: Tensor, # (N_packed, D)
        ge_group_ex_idx: Tensor, # (Tprefix, Gmax, K)
        ge_group_hpge_pos: Tensor, # (Tprefix, Gmax, K)
        ge_group_valid: Tensor # (Tprefix, Gmax)
    ):
        K = self.config.K
        B_hpge = e_lar.shape[0] // 2
        D = e_lar.shape[-1]

        Tprefix, Gmax, _ = ge_group_ex_idx.shape

        ge_group_ex_idx = ge_group_ex_idx.to(torch.long)
        ge_group_hpge_pos = ge_group_hpge_pos.to(torch.long)
        ge_group_valid = ge_group_valid.to(torch.bool)

        alpha_t = torch.as_tensor(
            self.config.alpha_t[:Tprefix],
            device=e_lar.device,
            dtype=e_lar.dtype
        )  # (Tprefix,)

        active_prefix = ge_group_valid.any(dim=1) & (alpha_t > 0)
        if not active_prefix.any():
            raise RuntimeError("No valid deep-supervision prefix groups were found in this batch.")

        # Gather into fixed shapes
        hpge_all = e_hpge[ge_group_hpge_pos] # (Tprefix, Gmax, K, D)
        lar_y0_all = e_lar[ge_group_ex_idx] # (Tprefix, Gmax, K, D)
        lar_y1_all = e_lar[ge_group_ex_idx + B_hpge] # (Tprefix, Gmax, K, D)

        # Flatten prefix and group dims for big batched matmuls
        TG = Tprefix * Gmax
        hpge_all = hpge_all.view(TG, K, D)
        lar_y0_all = lar_y0_all.view(TG, K, D)
        lar_y1_all = lar_y1_all.view(TG, K, D)

        logits_y0 = torch.bmm(lar_y0_all, hpge_all.transpose(1, 2)) / self.config.temperature
        logits_y1 = torch.bmm(lar_y1_all, hpge_all.transpose(1, 2)) / self.config.temperature

        logK = math.log(K)
        loggamma = 0.0 if self.config.gamma == 1 else math.log(self.config.gamma)

        null_col = torch.full((TG, K, 1), logK, device=e_lar.device, dtype=e_lar.dtype)
        logits_y0 = torch.cat([null_col, logits_y0 + loggamma], dim=-1) # (TG, K, K+1)
        logits_y1 = torch.cat([null_col, logits_y1 + loggamma], dim=-1) # (TG, K, K+1)

        target_y0 = torch.zeros(TG * K, dtype=torch.long, device=e_lar.device)
        target_y1 = torch.arange(K, device=e_lar.device).unsqueeze(0).expand(TG, K).reshape(TG * K) + 1

        loss_y0_rows = F.cross_entropy(
            logits_y0.reshape(TG * K, K + 1),
            target_y0,
            reduction="none"
        ).view(Tprefix, Gmax, K)

        loss_y1_rows = F.cross_entropy(
            logits_y1.reshape(TG * K, K + 1),
            target_y1,
            reduction="none"
        ).view(Tprefix, Gmax, K)

        valid_f = ge_group_valid.to(e_lar.dtype) # (Tprefix, Gmax)
        valid_rows = valid_f.unsqueeze(-1) # (Tprefix, Gmax, 1)

        coeff_y0 = 1.0 / (1.0 + self.config.gamma)
        coeff_y1 = self.config.gamma / (1.0 + self.config.gamma)

        row_count = ge_group_valid.sum(dim=1).to(e_lar.dtype) * K # (Tprefix,)

        prefix_losses_t = torch.full(
            (Tprefix,),
            float("nan"),
            device=e_lar.device,
            dtype=e_lar.dtype
        )

        sum_y0 = (loss_y0_rows * valid_rows).sum(dim=(1, 2)) # (Tprefix,)
        sum_y1 = (loss_y1_rows * valid_rows).sum(dim=(1, 2)) # (Tprefix,)

        prefix_losses_t[active_prefix] = (
            coeff_y0 * (sum_y0[active_prefix] / row_count[active_prefix])
            + coeff_y1 * (sum_y1[active_prefix] / row_count[active_prefix])
        )

        total_loss = (
            alpha_t[active_prefix] * prefix_losses_t[active_prefix]
        ).sum() / alpha_t[active_prefix].sum()

        prefix_losses = prefix_losses_t.detach().cpu().tolist()
        return total_loss, prefix_losses

    def forward_batch(
        self,
        f_idx: Tensor,
        f_vals: Tensor,
        ge_cu_seqlens: Tensor,
        ge_max_seqlen: int,
        ge_group_ex_idx: Tensor,
        ge_group_hpge_pos: Tensor,
        ge_group_valid: Tensor,
        t_idx: Tensor,
        s_idx: Tensor,
        v_val: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int
    ):
        e_lar, e_hpge = self.model(
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

        e_lar = F.normalize(e_lar, p=2, dim=-1)
        e_hpge = F.normalize(e_hpge, p=2, dim=-1)

        K = self.config.K
        if self.config.deep_supervision == 0:
            G = (len(ge_cu_seqlens) - 1) // K
            D = e_lar.shape[-1]

            logits = torch.cat(
                [
                    torch.bmm(
                        e_lar[:G * K].reshape(G, K, D),
                        e_hpge.reshape(G, K, D).transpose(1, 2)
                    ),
                    torch.bmm(
                        e_lar[G * K:].reshape(G, K, D),
                        e_hpge.reshape(G, K, D).transpose(1, 2)
                    )
                ],
                dim=1
            ) / self.config.temperature

            return self.calculate_loss(logits, K), None

        return self.calculate_deep_supervision_loss(
            e_lar=e_lar,
            e_hpge=e_hpge,
            ge_group_ex_idx=ge_group_ex_idx,
            ge_group_hpge_pos=ge_group_hpge_pos,
            ge_group_valid=ge_group_valid
        )

    def train_batch(
        self, *args, **kwargs
    ):

        loss, prefix_losses = self.forward_batch(*args, **kwargs)
        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        return loss.detach().item(), prefix_losses

    def train_epoch(self):
        loss = 0.
        n_step = 0

        prefix_loss_sum = [0.0] * (self.config.hpge_num_features + (2 if self.config.subpartition_hpge_feats == 1 else 1))
        prefix_loss_count = [0] * (self.config.hpge_num_features + (2 if self.config.subpartition_hpge_feats == 1 else 1))

        for (lar, hpge), _ in self.dataloader:
            (_, t_idx, s_idx, v_val, cu_seqlens, max_seqlen, _) = lar
            (_, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, _, ge_group_meta) = hpge

            if ge_group_meta is None:
                ge_group_ex_idx = None
                ge_group_hpge_pos = None
                ge_group_valid = None
            else:
                ge_group_ex_idx, ge_group_hpge_pos, ge_group_valid = ge_group_meta

            loss_, prefix_losses_ = self.train_batch(
                f_idx=ge_f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                f_vals=ge_f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                ge_cu_seqlens=ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                ge_max_seqlen=int(ge_max_seqlen),
                ge_group_ex_idx=ge_group_ex_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long) if ge_group_ex_idx is not None else None,
                ge_group_hpge_pos=ge_group_hpge_pos.to(device=self.device, non_blocking=True).to(dtype=torch.long) if ge_group_hpge_pos is not None else None,
                ge_group_valid=ge_group_valid.to(device=self.device, non_blocking=True).to(dtype=torch.bool) if ge_group_valid is not None else None,
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                v_val=v_val.to(device=self.device, non_blocking=True).to(dtype=torch.float32) if v_val is not None else v_val,
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                max_seqlen=int(max_seqlen)
            )

            loss += loss_
            n_step += 1

            if prefix_losses_ is not None:
                for t, v in enumerate(prefix_losses_):
                    if not math.isnan(v):
                        prefix_loss_sum[t] += v
                        prefix_loss_count[t] += 1

        n_step = 1 / n_step
        self.train_loss.append(loss * n_step)

        if self.config.deep_supervision == 1:
            self.train_prefix_losses.append([
                prefix_loss_sum[t] / prefix_loss_count[t] if prefix_loss_count[t] > 0 else math.nan
                for t in range(len(prefix_loss_sum))
            ])

    def val_batch(self):
        return

    @torch.no_grad()
    def val_epoch(self):
        loss = 0.
        n_step = 0

        prefix_loss_sum = [0.0] * (self.config.hpge_num_features + (2 if self.config.subpartition_hpge_feats == 1 else 1))
        prefix_loss_count = [0] * (self.config.hpge_num_features + (2 if self.config.subpartition_hpge_feats == 1 else 1))

        for (lar, hpge), _ in self.dataloader:
            (_, t_idx, s_idx, v_val, cu_seqlens, max_seqlen, _) = lar
            (_, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, _, ge_group_meta) = hpge

            if ge_group_meta is None:
                ge_group_ex_idx = None
                ge_group_hpge_pos = None
                ge_group_valid = None
            else:
                ge_group_ex_idx, ge_group_hpge_pos, ge_group_valid = ge_group_meta

            loss_, prefix_losses_ = self.forward_batch(
                f_idx=ge_f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                f_vals=ge_f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                ge_cu_seqlens=ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                ge_max_seqlen=int(ge_max_seqlen),
                ge_group_ex_idx=ge_group_ex_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long) if ge_group_ex_idx is not None else None,
                ge_group_hpge_pos=ge_group_hpge_pos.to(device=self.device, non_blocking=True).to(dtype=torch.long) if ge_group_hpge_pos is not None else None,
                ge_group_valid=ge_group_valid.to(device=self.device, non_blocking=True).to(dtype=torch.bool) if ge_group_valid is not None else None,
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                v_val=v_val.to(device=self.device, non_blocking=True).to(dtype=torch.float32) if v_val is not None else v_val,
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                max_seqlen=int(max_seqlen)
            )

            loss += loss_.detach().item()
            n_step += 1

            if prefix_losses_ is not None:
                for t, v in enumerate(prefix_losses_):
                    if not math.isnan(v):
                        prefix_loss_sum[t] += v
                        prefix_loss_count[t] += 1

        n_step = 1 / n_step
        self.val_loss.append(loss * n_step)

        if self.config.deep_supervision == 1:
            self.val_prefix_losses.append([
                prefix_loss_sum[t] / prefix_loss_count[t] if prefix_loss_count[t] > 0 else math.nan
                for t in range(len(prefix_loss_sum))
            ])

def train(
    experiment: str,
    partition: str,
    model_name: str,
    version: str,
    train_dataset_version: str,
    dataflow_dir: str,
    base_cfg_name: str,
    tmp_dir: str,
    cache_dir: str
):
    rank, world_size, device = _init_torch(cache_dir)

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
        version=train_dataset_version,
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
        version=train_dataset_version,
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
            global_id = model_cfg.num_bootstraps_per_fold * meta[0] + meta[1]
            to_be_trained[global_id, -1] = meta[-1] + 1
        elif len(meta) == 4:
            meta = np.array(meta[:3]).astype(int)
            global_id = model_cfg.num_bootstraps_per_fold * meta[0] + meta[1]
            remove_id.append(global_id)

    to_be_trained = np.delete(to_be_trained, remove_id, axis=0)
    to_be_trained = to_be_trained.astype(int).tolist()

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
    parser.add_argument("train_dataset_version", type=str, help="Traning dataset version")
    parser.add_argument("dataflow_dir", type=str, help="Directory of the dataflow")
    parser.add_argument("base_cfg_name", type=str, help="Name of the base json config")
    parser.add_argument("tmp_dir", type=str, help="Directory for storing temporary files")
    parser.add_argument("cache_dir", type=str, help="Directory to store numba, torch.inductor and triton cache")
    args = parser.parse_args()

    train(
        args.experiment,
        args.partition,
        args.model_name,
        args.version,
        args.train_dataset_version,
        args.dataflow_dir,
        args.base_cfg_name,
        args.tmp_dir,
        args.cache_dir
    )
