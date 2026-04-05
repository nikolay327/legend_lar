import math
import multiprocessing as mp

from torch.utils.data import IterableDataset, get_worker_info

import numpy as np
import scipy


class LArDatasetBase:
    def __init__(self, dataset: scipy.sparse._csr.csr_matrix):
        self.dataset = dataset

    def __getitem__(self, indices):
        indices = indices[~np.isnan(indices)].astype(np.int64)
        return self.dataset[indices], indices

class ParallelBootstrappedKFoldLArListDataset(IterableDataset):
    """
        NOTE: the RC dataset is not chunked into K-folds here. The physics data is chunked.
              by default, the physics data is the one with label 1.

              But the RC training and validation dataset is bootstraped.

              The data loaded in the evaluation mode is from test_folds arg, while in the calibration mode, the data is shuffled,
              and then partitioned using calib_partitioning_nums arg.
              This means:
                - use eval mode for loading phys data during calibration
                - use calib mode for loading FT data used for calibration
                - use eval mode for loading FT data used for coverage test
    """
    def __init__(
        self,
        lar_data_lengths: list[int],
        num_t_bins: int,
        num_sipm_chs: int,
        batch_size: int,
        hpge_feats_mean: list[float] = None,
        hpge_feats_std: list[float] = None,
        calib_partitioning_nums: list[int] = None,
        test_folds: list[list[int]] = None,
        rng_seed_for_split: int = None,
        times_of_mixing: int = 5,
        bootstrap_rng_seed: int = None,
        global_rng_seed_for_sampling: int = None,
        num_folds: int = None,
        num_bootstraps_per_fold: int = None,
        mode: mp.Value = None, # can be an int when mode == 3 (evaluation mode)
        fold_id: mp.Value = None,
        bootstrap_id: mp.Value = None,
        epoch_id: mp.Value = None
    ):
        """
            self.mixed_indices is treated as a first global shuffle before the data is K-folded
        """
        super(ParallelBootstrappedKFoldLArListDataset, self).__init__()

        self.num_folds = num_folds # == K
        self.num_bootstraps_per_fold = num_bootstraps_per_fold
        
        self.rng_seed_for_split = rng_seed_for_split
        self.times_of_mixing = times_of_mixing
        self.bootstrap_rng_seed = bootstrap_rng_seed

        self.hpge_dataset = None
        self.lar_datasets = None
        self.hpge_feats_mean = hpge_feats_mean
        self.hpge_feats_std = hpge_feats_std

        self.test_folds = test_folds # needed in evaluation mode
        self.calib_partitioning_nums = calib_partitioning_nums # needed in calibration mode

        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.stratified_batch_sizes = [int(self.batch_size // 2), int(self.batch_size // 2)]

        self.num_t_bins = num_t_bins
        self.num_sipm_chs = num_sipm_chs
        self.global_rng_seed_for_sampling = global_rng_seed_for_sampling
        self.sampling_rng = None

        self.mode = mode # train (0), val (1), calibration (2), evaluation (3)
        # NOTE: in evaluation mode (3), data is not chunked and is not bootstrapped
        self.fold_id = fold_id
        self.bootstrap_id = bootstrap_id
        self.bootstrap_id_cache = None
        self.epoch_id = epoch_id

        self.data_lengths = lar_data_lengths

        self._set_mixed_indices()
        self._set_folds_indices()

    def _load_datasets(self, hpge_dataset, lar_datasets):
        assert get_worker_info() is not None, "Can only be called by each worker to avoid pickling errors"

        self.hpge_dataset = hpge_dataset
        self.lar_datasets = lar_datasets

    def set_mode(self, mode: int):
        """To be called by the main process before iteration. mode value is stored in a multiprocessing.Value."""
        mode_value = self.mode if isinstance(self.mode, int) else self.mode.value
        if mode_value == 3:
            if mode != 3:
                raise ValueError("cannot change from evaluation mode to other modes")
        elif mode_value == 2:
            if mode != 2:
                raise ValueError("cannot change from calibration mode to other modes")
        else:
            if mode in (2, 3):
                raise ValueError("cannot change from other modes to calibration/evaluation mode")

        with self.mode.get_lock():
            self.mode.value = mode

    def set_fold_id(self, fold_id: int):
        """Called inside main process"""
        with self.fold_id.get_lock():
            self.fold_id.value = fold_id

    def set_bootstrap_id(self, bid: int):
        """Called inside main process"""
        with self.bootstrap_id.get_lock():
            self.bootstrap_id.value = bid
    
    def set_epoch(self, epoch: int):
        """Called inside main process"""
        with self.epoch_id.get_lock():
            self.epoch_id.value = epoch

    def _set_mixed_indices(self):
        # Monotonicly increasing default indices
        mixed_indices = [np.arange(mixing_idx).astype(np.int64) for mixing_idx in self.data_lengths]

        rng = np.random.default_rng(self.rng_seed_for_split)
        for i in range(len(self.data_lengths)):
            # Shuffle the default indices
            permuted_idx = mixed_indices[i]
            for _ in range(self.times_of_mixing):
                permuted_idx = permuted_idx[rng.permutation(len(permuted_idx))]
            mixed_indices[i] = permuted_idx.tolist()
        self.mixed_indices = mixed_indices

    def _get_data_chunks_cumsum(self, data_len: int, chunk_fractions: list[float] = None):
        if chunk_fractions is None:
            return np.array(0, data_len, dtype=np.int64)
        chunk_lens = [0]
        for i, chunk_fraction in enumerate(chunk_fractions):
            if i == len(chunk_fractions) - 1:
                chunk_lens.append(data_len - sum(chunk_lens))
            else:
                chunk_lens.append(math.ceil(data_len * chunk_fraction))
        cumsum = np.cumsum(chunk_lens).astype(np.int64)
        return cumsum

    def _set_folds_indices(self):
        mode_value = self.mode if isinstance(self.mode, int) else self.mode.value

        # train and val mode
        if mode_value in (0, 1):
            assert len(self.data_lengths) == 2
            # label 0 data (sg / RC dataset) is treated to have 1 fold
            sg_data_cumsum = self._get_data_chunks_cumsum(int(self.data_lengths[0]), [1.0, 0.0])
            # label 1 data with k folds (bg / physics)
            permuted_indices = np.array(self.mixed_indices[1], dtype=np.int64)
            num_data_within_folds = len(permuted_indices) // self.num_folds
            is_incomplete_last_fold = num_data_within_folds * self.num_folds != len(permuted_indices)
            test_folds = permuted_indices[:num_data_within_folds * self.num_folds]
            test_folds = test_folds.reshape(self.num_folds, -1).tolist()
            if is_incomplete_last_fold:
                last_fold = np.concatenate([test_folds[-1], permuted_indices[num_data_within_folds * self.num_folds:]], axis=0)
                test_folds[-1] = last_fold.tolist()

            bg_train_and_val_folds = []
            for test_fold in test_folds:
                mask = np.ones(len(permuted_indices)).astype(np.bool_)
                mask[test_fold] = False
                bg_train_and_val_folds.append(
                    permuted_indices[mask].tolist()
                )

            self.indices = {
                "sg": {
                    "train_val": self.mixed_indices[0][:sg_data_cumsum[-1]]
                },
                "bg": {
                    "test_folds": {
                        'fold_{k}'.format(k=k): test_folds[k] for k in range(self.num_folds)
                    },
                    "train_val": {
                        'fold_{k}'.format(k=k): bg_train_and_val_folds[k] for k in range(self.num_folds)
                    }
                }
            }
        elif mode_value == 2:
            self.indices = self.test_folds[int(self.fold_id.value)]
        else:
            self.indices = np.arange(int(np.array(self.data_lengths, dtype=np.int64)[0]), dtype=np.int64).tolist()

    def _set_sg_and_bg_datasets(self):
        assert get_worker_info() is not None, "Can only be called by each worker inside worker_init"
        if self.hpge_dataset is not None:
            mean = np.array(self.hpge_feats_mean).reshape(1, -1)
            std = np.array(self.hpge_feats_std).reshape(1, -1)
            self.hpge_dataset = (self.hpge_dataset - mean) / std

        self.dataset = [
            LArDatasetBase(self.lar_datasets[i]) for i in range(len(self.lar_datasets))
        ] # sg, bg

    def _worker_init(self, hpge_dataset, lar_datasets):
        worker_info = get_worker_info()
        self.worker_id = worker_info.id
        self.num_workers = worker_info.num_workers

        self._load_datasets(hpge_dataset, lar_datasets)

        self._set_sg_and_bg_datasets()

    def _set_current_bootstrap_indices(self):
        assert get_worker_info() is not None, "Can only be called by each worker"
        mode_value = self.mode if isinstance(self.mode, int) else self.mode.value

        if mode_value != 0:
            return

        if self.bootstrap_id.value == self.bootstrap_id_cache:
            return
        else:
            self.bootstrap_id_cache = self.bootstrap_id.value

        bootstrap_rng = np.random.default_rng(self.bootstrap_rng_seed + self.num_folds*self.fold_id.value + self.bootstrap_id.value) # bootstrap rng is the same for each worker because randomness is controlled globally

        sg_fold = np.array(self.indices["sg"]["train_val"])
        n_sg = len(sg_fold)
        sg_boot_idx = bootstrap_rng.choice(n_sg, size=n_sg, replace=True)
        sg_inbag_unique = np.unique(sg_boot_idx)
        sg_oob_mask = np.ones(n_sg, dtype=np.bool_)
        sg_oob_mask[sg_inbag_unique] = False
        sg_oob_idx = np.flatnonzero(sg_oob_mask)

        self.current_sg_train = sg_fold[sg_boot_idx]
        self.current_sg_val = sg_fold[sg_oob_idx]

        bg_fold = np.array(self.indices["bg"]["train_val"]['fold_{i}'.format(i=int(self.fold_id.value))])
        n_bg = len(bg_fold)
        bg_boot_idx = bootstrap_rng.choice(n_bg, size=n_bg, replace=True)
        bg_inbag_unique = np.unique(bg_boot_idx)
        bg_oob_mask = np.ones(n_bg, dtype=np.bool_)
        bg_oob_mask[bg_inbag_unique] = False
        bg_oob_idx = np.flatnonzero(bg_oob_mask)

        self.current_bg_train = bg_fold[bg_boot_idx]
        self.current_bg_val = bg_fold[bg_oob_idx]

        self.current_bg_test_fold = np.array(self.indices["bg"]["test_folds"]['fold_{i}'.format(i=int(self.fold_id.value))])

    def _shuffle(self):
        """
            To be called by each worker, using a global rng seed.
            NOTE: both the sg and bg datasets need to always be loaded and bootstrapped together for consistency
        """

        mode_value = self.mode if isinstance(self.mode, int) else self.mode.value
        if mode_value == 0:
            seed = self.global_rng_seed_for_sampling + self.num_folds*self.fold_id.value + self.bootstrap_id.value
            seed = seed + self.epoch_id.value
            self.sampling_rng = np.random.default_rng(seed)

        if mode_value == 0:
            self.current_indices = [
                self.current_sg_train[self.sampling_rng.permutation(len(self.current_sg_train))].astype(np.float32),
                self.current_bg_train[self.sampling_rng.permutation(len(self.current_bg_train))].astype(np.float32)
            ]
        elif mode_value == 1:
            self.current_indices = [
                self.current_sg_val[self.sampling_rng.permutation(len(self.current_sg_val))].astype(np.float32),
                self.current_bg_val[self.sampling_rng.permutation(len(self.current_bg_val))].astype(np.float32)
            ]
        else:
            self.current_indices = [
                np.array(self.indices).astype(np.float32)
            ]

        current_indices = []
        for i in range(len(self.current_indices)):
            batch_size = self.stratified_batch_sizes[i]
            indices = self.current_indices[i]
            last_batch = None
            if mode_value not in (0, 1) : # no drop-last during calibration, and evaluation modes
                if len(indices) % batch_size != 0:
                    last_batch_len = len(indices) - batch_size * (len(indices) // batch_size)
                    # pad the last incomplete batch with nans and append to indices
                    last_batch = np.zeros(batch_size)
                    last_batch[:last_batch_len] = indices[-last_batch_len:]
                    last_batch[last_batch_len:] = np.nan
                    # the nan entries will be handled in the collate_fn

            indices = indices[:batch_size * (len(indices) // batch_size)] # drop the last incomplete batch
            if last_batch is not None:
                indices = np.concatenate((indices, last_batch), axis=0) # pad the last incomplete batch with nans and append to indices
            indices = indices.reshape(-1, batch_size)
            indices = indices[self.worker_id::self.num_workers]
            current_indices.append(indices)
        self.current_indices = current_indices

    def __len__(self):
        return min([len(indices) for indices in self.current_indices])
    
    def __iter__(self):
        self._set_current_bootstrap_indices()
        self._shuffle()
        mode_value = self.mode if isinstance(self.mode, int) else self.mode.value
        for idx in range(len(self)):
            batch = []
            indices = []
            for i in range(len(self.current_indices)):
                indices_shard = self.current_indices[i][idx]
                partial_batch, indices_shard = self.dataset[i][indices_shard]
                batch.append(partial_batch.toarray().reshape(-1, self.num_t_bins, self.num_sipm_chs))
                indices.append(indices_shard)

            batch = np.concatenate(batch, axis=0) if len(batch) > 1 else batch[0]
            if mode_value in (0, 1):
                gE = self.hpge_dataset[indices[1]]
            else:
                if self.hpge_dataset is not None:
                    gE = self.hpge_dataset[indices[0]]
                else:
                    gE = np.zeros((len(batch), len(self.hpge_feats_mean) + 2), dtype=np.float32)
            indices = np.concatenate(indices, axis=0) if len(indices) > 1 else indices[0]
            yield batch, gE, indices


def ParallelKFoldBootstrap_worker_init_fn(hpge_dataset, lar_datasets, worker_id: int):
    """
        Has to be wrapped using functools.partial to avoid self.hpge_dataset and self.lar_datasets being pickled
    """
    worker_info = get_worker_info()
    dataset: ParallelBootstrappedKFoldLArListDataset = worker_info.dataset
    dataset._worker_init(hpge_dataset, lar_datasets)
