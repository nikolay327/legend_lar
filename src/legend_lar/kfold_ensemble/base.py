from abc import ABC, abstractmethod
import os
from typing import List

import random

import numpy as np
import scipy

import torch
import torch._inductor.config as cfg
cfg.autotune_local_cache = False

from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from legend_lar.utils import FileDB, NRECConfig, InitRNG
from legend_lar.data import ParallelBootstrappedKFoldLArListDataset, NRECCollateFn, ParallelKFoldBootstrap_worker_init_fn

from functools import partial

class TrainerBase(ABC):
    def __init__(
        self,
        file_db: FileDB,
        partition: str,
        model_name: str,
        version: str,
        config: NRECConfig,
        hpge_dataset: np.ndarray,
        lar_datasets: scipy.sparse._csr.csr_matrix,
        rank: int,
        world_size: int,
        device: str | int,
        tmp_dir: str
    ):
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.file_db = file_db
        self.partition = partition
        self.model_name = model_name
        self.version = version

        self.device = device
        self.config = config

        self.rank = rank
        self.world_size = world_size

        self._init_loss_store()

        self.BASE_SEED = self.config.rng_seed
        self.BASE_RNG = random.Random(self.BASE_SEED)

        self.rng_seed_for_data_plit = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_bootstrap = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_data_sampling = self.BASE_RNG.getrandbits(64)
        self.rng_seed_for_model_reinit = self.BASE_RNG.getrandbits(64)
        self._set_model_initializer()

        self._init_dataloader(hpge_dataset, lar_datasets)

        self.best_val_loss = 9999.
        self.patience = 0
        self.last_saved_epoch = None

        self.current_fid = None
        self.current_bid = None

    def _init_loss_store(self):
        self.train_loss = []
        self.val_loss = []

    def _init_dataloader(self, hpge_dataset, lar_datasets):
        self.mode_value = mp.Value("i", 0)
        self.fid_value = mp.Value("i", 0)
        self.bid_value = mp.Value("i", 0)
        self.epoch_value = mp.Value("i", 0)

        self.dataset = ParallelBootstrappedKFoldLArListDataset(
            lar_data_lengths=[dataset.shape[0] for dataset in lar_datasets],
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.config.local_batch_size,
            hpge_feats_mean=self.config.hpge_feats_mean,
            hpge_feats_std=self.config.hpge_feats_std,
            rng_seed_for_split=self.rng_seed_for_data_plit,
            times_of_mixing=self.config.times_of_mixing,
            bootstrap_rng_seed=self.rng_seed_for_bootstrap,
            global_rng_seed_for_sampling=self.rng_seed_for_data_sampling,
            num_folds=self.config.num_folds,
            mode=self.mode_value,
            fold_id=self.fid_value,
            bootstrap_id=self.bid_value,
            epoch_id=self.epoch_value
        )
        self.collate_fn = NRECCollateFn(
            cls_placeholder_id=self.config.cls_placeholder_id,
            cuda_device=self.device
        )
        worker_init_fn = partial(
            ParallelKFoldBootstrap_worker_init_fn,
            hpge_dataset,
            lar_datasets
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=4,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn
        )

        if self.rank == 0:
            k_fold = self.dataset.indices["bg"]["test_folds"]
            for fid in range(self.config.num_folds):
                fold_indices = np.array(k_fold['fold_{i}'.format(i=fid)], dtype=np.int64)
                fold_path = self.file_db.build_file(
                    tier="fold_ids",
                    partition=self.partition,
                    model_name=self.model_name,
                    version=self.version,
                    fid=fid
                )
                os.makedirs(os.path.dirname(fold_path), exist_ok=True)
                np.save(fold_path, fold_indices)

        if self.world_size > 1:
            torch.distributed.barrier()

    def _set_model_initializer(self):
        self.model_initiator = InitRNG(
            device=self.device
        )

    def get_model_reinit_seed(self, fid: int, bid: int):
        return self.rng_seed_for_model_reinit + self.config.num_folds * fid + bid

    @abstractmethod
    def reset_model_and_optimizer(self, fid: int, bid: int, start_from_epoch: int = 1):
        pass

    def save_checkpoint(self, fid: int, bid: int, epoch: int):
        save_path = self.file_db.build_file(
            tier="models",
            partition=self.partition,
            model_name=self.model_name,
            version=self.version,
            fid=fid,
            bid=bid
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.isfile(save_path):
            os.remove(save_path)

        torch.save({
            "fid": fid,
            "bid": bid,
            "epoch": epoch,
            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "model_opt": self.model_opt.state_dict(),

            "train_loss": self.train_loss,
            "val_loss": self.val_loss
        }, save_path)

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

    def train_and_val_one_epoch(self, fid: int, bid: int, epoch: int):
        self.dataloader.dataset.set_epoch(epoch)

        # training
        self.dataloader.dataset.set_mode(0)
        self.model.train()
        self.train_epoch()

        # validation
        self.dataloader.dataset.set_mode(1)
        self.model.eval()
        self.val_epoch()

        print(f"fid {fid}, bid {bid}, epoch {epoch} | Train Loss: {self.train_loss[-1]:.6f}, Val Loss: {self.val_loss[-1]:.6f}")

        delta_loss = self.best_val_loss - self.val_loss[-1]
        min_delta = self.config.rel_tolerance * self.best_val_loss
        if delta_loss > min_delta:
            self.best_val_loss = self.val_loss[-1]
            self.save_checkpoint(fid, bid, epoch)
            self.patience = 0

            path = f'{self.tmp_dir}/{self.current_fid}_{self.current_bid}_{self.last_saved_epoch}'
            if os.path.isdir(path):
                os.rmdir(path)

            self.last_saved_epoch = epoch
            os.makedirs(f'{self.tmp_dir}/{self.current_fid}_{self.current_bid}_{self.last_saved_epoch}')
        else:
            self.patience += 1

    def train_one_model(self, fid: int, bid: int, start_from_epoch: int = 1):
        self.current_fid = fid
        self.current_bid = bid

        self.dataloader.dataset.set_fold_id(bid)
        self.dataloader.dataset.set_bootstrap_id(bid)
        self.reset_model_and_optimizer(fid, bid, start_from_epoch)

        if start_from_epoch == 1:
            print(f'fid_{fid}, bid_{bid} training started')
        else:
            self.last_saved_epoch = start_from_epoch - 1
            print(f'fid_{fid}, bid_{bid} training continued')

        for epoch in range(start_from_epoch, self.config.max_epochs+1):
            self.train_and_val_one_epoch(fid, bid, epoch)

            if self.patience == self.config.patience:
                break

        path = f'{self.tmp_dir}/{self.current_fid}_{self.current_bid}_{self.last_saved_epoch}'
        os.rmdir(path)
        os.makedirs(f'{path}_done')

        self.best_val_loss = 99999.
        self.patience = 0
        self._init_loss_store()

    def train(self, to_be_trained: List[List[int]]):
        for fid, bid, start_from_epoch in to_be_trained:
            self.train_one_model(fid, bid, start_from_epoch)
