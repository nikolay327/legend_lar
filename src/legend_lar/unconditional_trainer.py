import os
from pathlib import Path

import torch
import torch._inductor.config as cfg
cfg.autotune_local_cache = False

from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.amp import autocast
from bitsandbytes.optim import LAMB

from legend_lar.model import UnconditionalRatioEstimator
from legend_lar.utils import ModelConfig, _initialize_configs, _init_torch
from legend_lar.data import LArListDataset, CollateFn, worker_init_fn
from legend_lar.calibration import NRETestMetrics


class Trainer:
    def __init__(
        self,
        model: UnconditionalRatioEstimator,
        config: ModelConfig,
        dataloader: DataLoader,
        val_dataloader: DataLoader,
        nre_tester: NRETestMetrics,
        device: str | int
    ):
        self.device = device
        self.config = config

        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        self.model = model.to(device=self.device)

        self._init_loss_store()
        self.nre_tester = nre_tester

    def _init_loss_store(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []

    def clean_state_dict(self, state_dict):
        cleaned_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('_orig_mod.', '')
            cleaned_dict[cleaned_key] = value
        return cleaned_dict

    def load_checkpoint(self, starting_epoch: int):
        assert starting_epoch > 0
        checkpoint_id = starting_epoch - 1

        if checkpoint_id > 0:
            cp = torch.load(f'{self.config.save_to}/checkpoint_{checkpoint_id}.pt', map_location=self.device)
            self.model.load_state_dict(self.clean_state_dict(cp["model"]), strict=True)

        self.model = torch.compile(self.model, dynamic=True)

        if checkpoint_id == 0:
            return None

        self.train_loss = cp["train_loss"]
        self.val_loss = cp["val_loss"]
        self.train_acc = cp["train_acc"]
        model_opt_state = cp["model_opt"]

        return model_opt_state

    def save_checkpoint(self, epoch: int, nre_test_result: dict):
        torch.save({
            "epoch": epoch,
            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),

            "model_opt": self.model_opt.state_dict(),

            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "calibration_metrics": nre_test_result
        }, f'{self.config.save_to}/checkpoint_{epoch}.pt')
        print(f"Epoch {epoch} | Train Loss: {self.train_loss[-1]:.6f}, Val Loss: {self.val_loss[-1]:.6f}")

    def init_optimizer(self, model_opt_state):
        self.model_opt = LAMB(
            params=self.model.parameters(),
            lr = self.config.lr_model,
            betas=self.config.betas_model,
            weight_decay=self.config.weight_decay
        )
        if model_opt_state is not None:
            self.model_opt.load_state_dict(model_opt_state)
        torch.cuda.empty_cache()

    def train_batch(
        self,
        b_idx: Tensor,
        t_idx: Tensor,
        s_idx: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        lengths: Tensor,
        labels: Tensor
    ):
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(
                b_idx=b_idx,
                t_idx=t_idx,
                s_idx=s_idx,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                lengths=lengths
            ).squeeze(-1)
        logits = logits.to(dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        with torch.no_grad():
            acc = (F.sigmoid(logits[labels == 1]) > 0.5).to(torch.float32).sum()
            acc += (F.sigmoid(logits[labels == 0]) <= 0.5).to(torch.float32).sum()
            acc = acc / labels.shape[0]

        return loss.detach().cpu().item(), acc.detach().cpu().item()

    def train_epoch(self, epoch: int):
        self.dataloader.dataset.set_epoch(epoch)
        acc = 0.
        loss = 0.
        n_step = 0

        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels in self.dataloader:
            g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long),
            E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32),
            loss_, acc_ = self.train_batch(
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True),
                labels=labels.to(device=self.device, non_blocking=True)
            )
            acc += acc_
            loss += loss_
            n_step += 1

        n_step = 1 / n_step
        self.train_acc.append(acc * n_step)
        self.train_loss.append(loss * n_step)

    @torch.no_grad()
    def val_batch(
        self,
        b_idx: Tensor,
        t_idx: Tensor,
        s_idx: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        lengths: Tensor,
        labels: Tensor
    ):
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(
                b_idx=b_idx,
                t_idx=t_idx,
                s_idx=s_idx,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                lengths=lengths
            ).squeeze(-1)
        logits = logits.to(dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss.detach().cpu().item(), logits.detach(), labels.detach()
    
    def val_epoch(self, epoch: int, global_rng_seed_offset: int):
        self.val_dataloader.dataset.set_epoch(global_rng_seed_offset + epoch)
        with torch.no_grad():
            self.nre_tester.reset_buffers()

        loss = 0.
        n_step = 0
        for g, E, b_idx, t_idx, s_idx, cu_seqlens, max_seqlen, lengths, labels in self.val_dataloader:
            g=g.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            E=E.to(device=self.device, non_blocking=True).to(dtype=torch.float32)
            loss_, logits, label = self.val_batch(
                b_idx=b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                t_idx=t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                s_idx=s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long),
                cu_seqlens=cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.int32),
                max_seqlen=int(max_seqlen),
                lengths=lengths.to(device=self.device, non_blocking=True),
                labels=labels.to(device=self.device, non_blocking=True)
            )
            loss += loss_
            n_step += 1

            with torch.no_grad():
                logits = -logits.unsqueeze(-1).repeat(1, 2)
                logits[:, -1] = 0.
            self.nre_tester.update(logits, label.to(dtype=torch.int64))

        n_step = 1 / n_step
        self.val_loss.append(loss * n_step)

        return self.nre_tester.aggregate()
 
    def train(self, starting_epoch: int, final_epoch: int):
        model_opt_state = self.load_checkpoint(starting_epoch)
        self.init_optimizer(model_opt_state)

        for epoch in range(starting_epoch, final_epoch+1):
            self.model.train()
            self.train_epoch(epoch)

            self.model.eval()
            nre_test_result = self.val_epoch(epoch, final_epoch - starting_epoch + 1)

            self.save_checkpoint(epoch, nre_test_result)

def _prepare_model(model_config: ModelConfig, device: str | int):
    model = UnconditionalRatioEstimator(
        config=model_config,
        device=device
    ).to(dtype=torch.float32)

    return model

def train_unconditional(starting_epoch: int, final_epoch: int, experiment: str, model_name: str, version: str, working_dir: str, data_dir: str, training_config: str):
    local_rank, rank, world_size, device = _init_torch()
    wd = Path(working_dir)
    mmpd = Path(data_dir)

    config, data_config, paths = _initialize_configs(
        config_obj=ModelConfig(),
        wd=wd,
        experiment=experiment,
        model_name=model_name,
        version=version,
        mmpd=mmpd,
        training_config=training_config
    )

    model = _prepare_model(
        model_config=config,
        device=device
    )

    config.data_paths = [str(paths.data_dir / f"{key}.npz") for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.prior = [data_config[key]["prior"] for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.labels = [data_config[key]["label"] for key in data_config.keys() if str(key) != "hpge_id_and_energy"]
    config.hpge_id_and_energy = data_config["hpge_id_and_energy"]
    config.hpge_id_and_energy = str(paths.data_dir / f'{config.hpge_id_and_energy}.npy')

    epoch_value = mp.Value("i", starting_epoch)
    dataset = LArListDataset(
        hpge_path=config.hpge_id_and_energy,
        lar_paths=config.data_paths,
        labels=config.labels,
        true_coincidence_label=config.true_coincidence_label,
        hpge_energy_mean=config.hpge_energy_mean,
        hpge_energy_std=config.hpge_energy_std,
        prior=config.prior,
        train_val_test_fract=config.train_val_test_fract,
        local_batch_size=config.local_batch_size,
        num_t_bins=config.num_sipm_t_bins,
        num_sipm_chs=config.num_sipms,
        rng_seed_for_split=config.rng_seed_for_split,
        times_of_mixing=config.times_of_mixing,
        global_rng_seed_for_sampling=config.global_rng_seed_for_sampling,
        epoch_value=epoch_value
    )
    collate_fn = CollateFn(
        num_sipm_chs=config.num_sipms,
        true_coincidence_label=config.true_coincidence_label,
        cuda_device=device
    )
    dataset.set_mode(0)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    val_dataset = LArListDataset(
        hpge_path=config.hpge_id_and_energy,
        lar_paths=config.data_paths,
        labels=config.labels,
        true_coincidence_label=config.true_coincidence_label,
        hpge_energy_mean=config.hpge_energy_mean,
        hpge_energy_std=config.hpge_energy_std,
        prior=config.prior,
        train_val_test_fract=config.train_val_test_fract,
        local_batch_size=config.local_batch_size,
        num_t_bins=config.num_sipm_t_bins,
        num_sipm_chs=config.num_sipms,
        rng_seed_for_split=config.rng_seed_for_split,
        times_of_mixing=config.times_of_mixing,
        global_rng_seed_for_sampling=config.global_rng_seed_for_sampling,
        epoch_value=epoch_value
    )
    val_dataset.set_mode(1)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    nre_tester = NRETestMetrics(
        ece_bins=config.ece_bins,
        n_classes=config.n_classes,
        device=device
    )
    trainer = Trainer(
        model=model,
        config=config,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        nre_tester=nre_tester,
        device=device
    )

    trainer.train(starting_epoch, final_epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training started")
    parser.add_argument("starting_epoch", type=int, help="Epoch to start the training. If 1, train from scratch, else, load checkpoint")
    parser.add_argument("final_epoch", type=int, help="Training ended at this epoch")
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

    train_unconditional(args.starting_epoch, args.final_epoch, args.experiment, args.model_name, args.version, args.working_dir, args.data_dir, args.training_config)
