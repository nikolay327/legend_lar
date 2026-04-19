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
        ge_cu_seqlens: Tensor # (B_hpge + 1,)
    ):
        K = self.config.K
        B_hpge = ge_cu_seqlens.numel() - 1
        G_full = B_hpge // K
        D = e_lar.shape[-1]

        seq_starts = ge_cu_seqlens[:-1].to(torch.long)
        seq_lengths = (ge_cu_seqlens[1:] - ge_cu_seqlens[:-1]).view(G_full, K)
        k_idx = torch.arange(K, device=e_lar.device)

        total_loss = e_lar.new_zeros(())
        total_weight = 0.0

        prefix_losses = [math.nan] * (self.config.hpge_num_features + 1)
        for t in range(self.config.hpge_num_features + 1): #NOTE: +1 hardcoded: no partition id
            alpha = float(self.config.alpha_t[t])
            if alpha == 0.0:
                continue

            g_idx = ((seq_lengths > t).all(dim=1)).nonzero(as_tuple=True)[0]
            if g_idx.numel() == 0:
                continue

            G_t = g_idx.numel()
            seq_idx = (g_idx[:, None] * K + k_idx[None, :]).reshape(-1)

            hpge_t = e_hpge[seq_starts[seq_idx] + t].view(G_t, K, D)
            lar_y0 = e_lar[seq_idx].view(G_t, K, D)
            lar_y1 = e_lar[seq_idx + B_hpge].view(G_t, K, D)

            logits_t = torch.cat(
                [
                    torch.bmm(lar_y0, hpge_t.transpose(1, 2)),
                    torch.bmm(lar_y1, hpge_t.transpose(1, 2)),
                ],
                dim=1
            ) / self.config.temperature

            loss_t = self.calculate_loss(logits_t, K)
            prefix_losses[t] = loss_t.detach().item()

            total_loss = total_loss + alpha * loss_t
            total_weight += alpha

        return total_loss / total_weight, prefix_losses

    def forward_batch(
        self,
        f_idx: Tensor,  # (N_valid,)
        f_vals: Tensor,  # (N_valid,)
        ge_cu_seqlens: Tensor,  # (B/2+1,)
        ge_max_seqlen: int,
        t_idx: Tensor,  # (N,)
        s_idx: Tensor,  # (N,)
        cu_seqlens: Tensor,  # (B+1,)
        max_seqlen: int
    ):
        e_lar, e_hpge = self.model(
            t_idx=t_idx,
            s_idx=s_idx,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            f_idx=f_idx,
            f_vals=f_vals,
            ge_cu_seqlens=ge_cu_seqlens,
            ge_max_seqlen=ge_max_seqlen
        )

        e_lar = F.normalize(e_lar, p=2, dim=-1) # (B, D)
        e_hpge = F.normalize(e_hpge, p=2, dim=-1) # (B/2, D)

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
            ) / self.config.temperature # (G, 2K, K)

            return self.calculate_loss(logits, K), None

        return self.calculate_deep_supervision_loss(
            e_lar=e_lar,
            e_hpge=e_hpge,
            ge_cu_seqlens=ge_cu_seqlens
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

        prefix_loss_sum = [0.0] * (self.config.hpge_num_features + 1) #NOTE: +1 hardcoded: no partition id
        prefix_loss_count = [0] * (self.config.hpge_num_features + 1) #NOTE: +1 hardcoded: no partition id

        for (lar, hpge), _ in self.dataloader:
            (_, t_idx, s_idx, cu_seqlens, max_seqlen, _) = lar
            (_, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, _) = hpge

            loss_, prefix_losses_ = self.train_batch(
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
                for t in range(self.config.hpge_num_features + 1)
            ])

    def val_batch(self):
        return

    @torch.no_grad()
    def val_epoch(self):
        loss = 0.
        n_step = 0

        prefix_loss_sum = [0.0] * (self.config.hpge_num_features + 1) #NOTE: +1 hardcoded: no partition id
        prefix_loss_count = [0] * (self.config.hpge_num_features + 1) #NOTE: +1 hardcoded: no partition id

        for (lar, hpge), _ in self.dataloader:
            (_, t_idx, s_idx, cu_seqlens, max_seqlen, _) = lar
            (_, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, _) = hpge

            loss_, prefix_losses_ = self.forward_batch(
                f_idx=ge_f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                f_vals=ge_f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                ge_cu_seqlens=ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                ge_max_seqlen=int(ge_max_seqlen),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
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
                for t in range(self.config.hpge_num_features + 1)
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
