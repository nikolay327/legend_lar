from abc import ABC, abstractmethod
import os

import random

import numpy as np

import torch
import torch._inductor.config as cfg
cfg.autotune_local_cache = False

from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from legend_lar.utils import BootstrappedKFoldConfig
from legend_lar.data import BootstrappedKFoldLArListDataset, CollateFn, KFoldBootstrap_worker_init_fn

class TrainerBase(ABC):
    def __init__(
        self,
        config: BootstrappedKFoldConfig,
        device: str | int,
        load_both_indices: bool = False
    ):
        self.device = device
        self.config = config
        self.load_both_indices = load_both_indices
        self._init_loss_store()

        self.BASE_SEED = self.config.rng_seed
        self.BASE_RNG = random.Random(self.BASE_SEED)

        self.rng_seed_for_data_plit = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_bootstrap = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_data_sampling = self.BASE_RNG.getrandbits(64)
        self._set_model_initializer()

        self._init_dataloader()

        self.best_val_loss = 9999.
        self.patience = 0
        self.current_bid = -1

    @abstractmethod
    def _set_model_initializer(self):
        pass

    def _init_loss_store(self):
        self.train_loss = []
        self.val_loss = []

    def _init_dataloader(self):
        self.mode_value = mp.Value("i", 0)
        self.fid_value = mp.Value("i", 0)
        self.change_bootstrap_id_value = mp.Value("i", 1)

        self.dataset = BootstrappedKFoldLArListDataset(
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
            bootstrap_rng_seed=self.rng_seed_for_bootstrap,
            global_rng_seed_for_sampling=self.rng_seed_for_data_sampling,
            num_folds=self.config.num_folds,
            num_bootstraps_per_fold=self.config.num_bootstraps_per_fold,
            sg_train_val_cal_test_frac=self.config.sg_train_val_cal_test_frac,
            mode=self.mode_value,
            fold_id=self.fid_value,
            change_bootstrap_id=self.change_bootstrap_id_value,
            load_both_indices=self.load_both_indices
        )
        self.collate_fn = CollateFn(
            num_sipm_chs=self.config.num_sipms,
            cuda_device=self.device
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            prefetch_factor=4,
            persistent_workers=True,
            worker_init_fn=KFoldBootstrap_worker_init_fn,
            collate_fn=self.collate_fn
        )

        self.k_fold = self.dataset.indices["bg"]["test_folds"]

    @abstractmethod
    def reset_model_and_optimizer(self):
        pass

    def save_checkpoint(self, epoch: int):
        save_dir = f'{self.config.save_to}/fid_{self.fid_value.value}/bid_{self.current_bid}'
        os.makedirs(save_dir, exist_ok=True)
        try:
            os.remove(f'{save_dir}/model.pt')
        except:
            pass
        torch.save({
            "fid": self.fid_value.value,
            "bid": self.current_bid,
            "epoch": epoch,
            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),

            "train_loss": self.train_loss,
            "val_loss": self.val_loss
        }, f'{save_dir}/model.pt')

    @abstractmethod
    def train_batch(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def val_batch(self):
        pass

    @abstractmethod
    def val_epoch(self):
        pass

    def train_one_bootstrap(self):
        self.dataloader.dataset.set_bid_flag(1) # instruct the workers to create a fresh bootstrap of the current fold
        self.current_bid += 1

        self.reset_model_and_optimizer()
        print(f'fid_{self.fid_value.value}, bid_{self.current_bid}')
        for epoch in range(1, self.config.max_epochs+1):
            self.dataloader.dataset.set_mode(0) # training mode
            self.model.train()
            self.train_epoch()
            self.dataloader.dataset.set_bid_flag(0) # instruct the workers to not create a fresh bootstrap of the current fold anymore

            self.dataloader.dataset.set_mode(1) # validation mode
            self.model.eval()
            self.val_epoch()

            print(f"Epoch {epoch} | Train Loss: {self.train_loss[-1]:.6f}, Val Loss: {self.val_loss[-1]:.6f}")
            delta_loss = self.best_val_loss - self.val_loss[-1]
            min_delta = self.config.rel_tolerance * self.best_val_loss
            if delta_loss > min_delta:
                self.best_val_loss = self.val_loss[-1]
                self.save_checkpoint(epoch)
                self.patience = 0
            else:
                self.patience += 1

            if self.patience == self.config.patience:
                break
        self.best_val_loss = 9999.
        self.patience = 0
        self._init_loss_store()

    def train_folds(self):
        for fid in range(self.config.num_folds):
            self.dataloader.dataset.set_fold_id(fid)

            fold_indices = np.array(self.dataloader.dataset.indices["bg"]["test_folds"]['fold_{i}'.format(i=int(self.fid_value.value))], dtype=np.int64)
            fold_dir = f'{self.config.save_to}/fid_{self.fid_value.value}'
            os.makedirs(fold_dir, exist_ok=True)
            np.save(f'{fold_dir}/fold_indices.npy', fold_indices)

            self.current_bid = -1
            for _ in range(self.config.num_bootstraps_per_fold):
                self.train_one_bootstrap()
