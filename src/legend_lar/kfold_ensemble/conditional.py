import os
from pathlib import Path

import torch
import torch._inductor.config as cfg
cfg.autotune_local_cache = False

from torch import Tensor
import torch.nn.functional as F
from torch.amp import autocast
from bitsandbytes.optim import LAMB

from legend_lar.model import ConditionalRatioEstimator
from legend_lar.utils import BootstrappedKFoldConfig, _initialize_configs, _init_torch, InitRNG

from legend_lar.kfold_ensemble.base import TrainerBase

class ConditionalTrainer(TrainerBase):
    def __init__(
        self,
        config: BootstrappedKFoldConfig,
        device: str | int
    ):
        super(ConditionalTrainer, self).__init__(config, device)

    def _set_model_initializer(self):
        self.rng_seed_for_unconditional_model_init = self.BASE_RNG.getrandbits(64) # This is here for consistency
        self.rng_seed_for_conditional_model_init = self.BASE_RNG.getrandbits(64)

        self.model_initiator = InitRNG(
            seed=self.rng_seed_for_conditional_model_init, device=self.device
        )

    def reset_model_and_optimizer(self):
        self.model = ConditionalRatioEstimator(
            config=self.config,
            device=self.device
        ).to(dtype=torch.float32, device=self.device)
        self.model_initiator.reinit_(self.model)
        self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)

        self.model_opt = LAMB(
            params=self.model.parameters(),
            lr = self.config.lr_model,
            betas=self.config.betas_model,
            weight_decay=self.config.weight_decay
        )
        torch.cuda.empty_cache()

    def train_batch(
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
            logits, labels = self.model.training_forward(
                g=g,
                E=E,
                b_idx=b_idx,
                t_idx=t_idx,
                s_idx=s_idx,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                lengths=lengths
            )
        logits = logits.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction="mean")

        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        return loss.detach().cpu().item()

    def train_epoch(self):
        loss = 0.
        n_step = 0
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels in self.dataloader:
            labels=labels.to(device=self.device, non_blocking=True)
            loss_ = self.train_batch(
                g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True)
            )
            loss += loss_
            n_step += 1

        n_step = 1 / n_step
        self.train_loss.append(loss * n_step)

    @torch.no_grad()
    def val_batch(
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
            logits, labels = self.model.training_forward(
                g=g,
                E=E,
                b_idx=b_idx,
                t_idx=t_idx,
                s_idx=s_idx,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                lengths=lengths
            )
        logits = logits.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction="mean")

        return loss.cpu().item()

    def val_epoch(self):
        loss = 0.
        n_step = 0
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels in self.dataloader:
            labels=labels.to(device=self.device, non_blocking=True)
            loss_ = self.val_batch(
                g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True)
            )
            loss += loss_
            n_step += 1

        n_step = 1 / n_step
        self.val_loss.append(loss * n_step)

def train_conditional(experiment: str, model_name: str, version: str, working_dir: str, data_dir: str, training_config: str):
    local_rank, rank, world_size, device = _init_torch()
    wd = Path(working_dir)
    mmpd = Path(data_dir)

    config, data_config, paths = _initialize_configs(
        config_obj=BootstrappedKFoldConfig(),
        wd=wd,
        experiment=experiment,
        model_name=model_name,
        version=version,
        mmpd=mmpd,
        training_config=training_config
    )

    config.data_paths = [str(paths.data_dir / f"{key}.npz") for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.prior = [data_config[key]["prior"] for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.labels = [data_config[key]["label"] for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.hpge_id_and_energy = data_config["hpge_id_and_energy"]
    config.hpge_id_and_energy = str(paths.data_dir / f'{config.hpge_id_and_energy}.npy')

    trainer = ConditionalTrainer(
        config=config,
        device=device
    )

    trainer.train_folds()

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

    train_conditional(args.experiment, args.model_name, args.version, args.working_dir, args.data_dir, args.training_config)
