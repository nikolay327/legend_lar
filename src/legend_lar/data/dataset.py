import math
import multiprocessing as mp
import traceback

from torch.utils.data import IterableDataset, get_worker_info

from scipy.sparse import load_npz
from numpy.lib.format import open_memmap
import numpy as np

class LArDataset:
    def __init__(
        self,
        lar_path: str,
        label: int,
        shuffled_indices: np.ndarray | list[int],
        train_val_test_cumsum: np.ndarray | list[int],
        batch_size: int,
        calib_mode: bool = None,
        calib_dataset_frac: float = None
    ):
        """
            This dataset is initialized with an array of shuffled_indices, which is a permutation of the data indices.
            Depending on self.mode_id (train, val, test, calib, or eval mode), different chunks of the data can be iterated upon.
            Which chunk belongs to train/val/test/calib/eval depends on train_val_test_cumsum and calib_dataset_frac.
            Calibration mode takes the first calib_dataset_frac fraction of the test cumsum indices.

            Which chunk is currently visible by the dataset can be changed by calling self.set_mode:
                self.set_mode updates self.indices_buffer, which stores the indices of self.shuffled_indices that belongs to the chunk.

            How data is fetched on each iterator call:
                - If the shuffler argument is fed into a self.shuffle method call, self.indices is updated to store a permuted version of
                  the indices of the current chunk.
                - self.indices is then reshaped into (-1, batch_size) with the last incomplete batch being dropped.
                - If the shuffler argument is not fed, the last incomplete batch is padded with nans and is kept.
                - These batches are then "distributed" to the workers. Each worker always load unique batch indices.
                - The calls from upstream __iter__ method then fetch data based on these pre-fetched batch indices (see self.__getitem__).

            NOTE: shuffled_indices argument does not have to be a permutation of data indices (without replacements). It can also be indices,
                  sampled with replacements.
        """
        self.lar_path = lar_path
        self.label = label
        self.shuffled_indices = shuffled_indices
        self.train_val_test_cumsum = train_val_test_cumsum
        self.batch_size = batch_size
        self.calib_mode = calib_mode
        self.calib_dataset_frac = calib_dataset_frac

        self.mode_idx = None # train (0), val (1), or test (2)
        self.indices_buffer = None
        self.indices = None

        self.data = None

        # worker specifics
        self.worker_id = None
        self.num_workers = None

    def set_mode(self, mode_idx: int):
        """To be called before DataLoader is intilialized."""
        self.mode_idx = mode_idx

        if mode_idx in (0, 1, 2):
            self.indices_buffer = np.arange(self.train_val_test_cumsum[mode_idx], self.train_val_test_cumsum[mode_idx+1])
        else:
            self.indices_buffer = np.arange(self.train_val_test_cumsum[0], self.train_val_test_cumsum[-1])

        if mode_idx != 2:
            assert self.calib_mode is None
        else:
            assert self.calib_mode is not None

            calib_data_size = int(self.calib_dataset_frac * len(self.indices_buffer))
            if self.calib_mode:
                self.indices_buffer = self.indices_buffer[calib_data_size:]
            else:
                self.indices_buffer = self.indices_buffer[:calib_data_size]

    def shuffle(self, shuffler: np.ndarray):
        """To be called by each worker, using a global rng seed."""
        if shuffler is None:
            if self.mode_idx in (0, 1, 2):
                self.indices = np.array(self.shuffled_indices)[self.indices_buffer.astype(np.int64)].astype(np.float32)
            else:
                self.indices = self.indices_buffer.astype(np.float32)
        else:
            assert len(self.indices_buffer) == len(shuffler)
            self.indices = np.array(self.shuffled_indices)[self.indices_buffer.astype(np.int64)][shuffler].astype(np.float32)

        last_batch = None
        if self.mode_idx not in (0, 1) : # no drop-last during test and calibration mode, and an unshuffled mode
            if len(self.indices) % self.batch_size != 0:
                last_batch_len = len(self.indices) - self.batch_size * (len(self.indices) // self.batch_size)
                # pad the last incomplete batch with nans and append to self.indices
                last_batch = np.zeros(self.batch_size)
                last_batch[:last_batch_len] = self.indices[-last_batch_len:]
                last_batch[last_batch_len:] = np.nan
                # the nan entries will be handled in the collate_fn

        self.indices = self.indices[:self.batch_size * (len(self.indices) // self.batch_size)] # drop the last incomplete batch
        if last_batch is not None:
            self.indices = np.concatenate((self.indices, last_batch), axis=0)
        self.indices = self.indices.reshape(-1, self.batch_size)
        self.indices = self.indices[self.worker_id::self.num_workers]

    def __len__(self):
        if self.indices is None:
            return
        return len(self.indices)

    def _set_worker_id(self, id: int, num_workers: int):
        """To be called inside worker_init_fn"""
        self.worker_id = id
        self.num_workers = num_workers

    def _read_data_inside_worker(self):
        """To be called inside worker_init_fn"""
        assert get_worker_info() is not None
        self.data = load_npz(self.lar_path) # load everything into ram for each worker since this data is super small

    def __getitem__(self, idx: int):
        ids = self.indices[idx]
        ids = ids[~np.isnan(ids)].astype(np.int64)
        return self.data[ids.tolist()], np.ones(len(ids), dtype=np.float32) * self.label, ids

class LArListDataset(IterableDataset):
    def __init__(
        self,
        hpge_path: str,
        lar_paths: list[str],
        prior: list[float],
        labels: list[int],
        true_coincidence_label: int,
        hpge_energy_mean: float,
        hpge_energy_std: float,
        train_val_test_fract: list[float],
        local_batch_size: int,
        num_t_bins: int,
        num_sipm_chs: int,
        rng_seed_for_split: int,
        times_of_mixing: int,
        global_rng_seed_for_sampling: int,
        epoch_value: mp.Value = None,
        shuffle: bool = True,
        calib_mode: bool = None,
        calib_dataset_frac: float = None
    ):
        super(LArListDataset, self).__init__()
        assert len(lar_paths) == len(prior)
        assert sum(prior) == 1.0

        self.train_val_test_fract = train_val_test_fract
        self.rng_seed_for_split = rng_seed_for_split
        self.times_of_mixing = times_of_mixing

        self.hpge_path = hpge_path
        self.lar_paths = lar_paths
        self.labels = labels
        self.true_coincidence_label = true_coincidence_label
        self.hpge_energy_mean = hpge_energy_mean
        self.hpge_energy_std = hpge_energy_std

        self.prior = prior
        self.batch_size = local_batch_size
        self.num_t_bins = num_t_bins
        self.num_sipm_chs = num_sipm_chs
        self.global_rng_seed_for_sampling = global_rng_seed_for_sampling
        self.epoch_value = epoch_value
        self.shuffle = shuffle

        self.stratified_batch_sizes = None
        self.mixed_indices = None
        self.train_val_test_nums = None
        self.train_val_test_cumsums = None
        self.mode = None  # train (0), val (1), or test (2)

        self.calib_mode = calib_mode
        self.calib_dataset_frac = calib_dataset_frac

        self._set_stratified_batch_sizes()
        self._set_mixed_indices()
        self._set_train_val_test_nums()
        self._set_train_val_test_cumsums()

    def _set_stratified_batch_sizes(self):
        """
            Get the stratified batch sizes according to the prior and the total batch size using np.ceil.
            By convention, the last class will take the remainder if the batch sizes do not sum to the total batch size.
        """
        stratified_batch_sizes = np.ceil(np.array(self.prior) * self.batch_size)
        if stratified_batch_sizes.sum() != self.batch_size:
            stratified_batch_sizes[-1] = stratified_batch_sizes[-1] + self.batch_size - stratified_batch_sizes.sum()
        self.stratified_batch_sizes = stratified_batch_sizes.astype(np.int64).tolist()
        self.prior = (self.stratified_batch_sizes / np.sum(self.stratified_batch_sizes)).tolist()

    def _init_dataset(self):
        self.datasets = []
        for i in range(len(self.lar_paths)):
            lar_dataset = LArDataset(
                lar_path=self.lar_paths[i],
                label=self.labels[i],
                shuffled_indices=self.mixed_indices[i],
                train_val_test_cumsum=self.train_val_test_cumsums[i],
                batch_size=self.stratified_batch_sizes[i],
                calib_mode=self.calib_mode,
                calib_dataset_frac=self.calib_dataset_frac
            )
            self.datasets.append(lar_dataset)

    def _set_mixed_indices(self):
        # Monotonicly increasing default indices
        self.data_lengths = [
            load_npz(path).shape[0] for path in self.lar_paths
        ]
        mixed_indices = [np.arange(mixing_idx).astype(np.int64) for mixing_idx in self.data_lengths]

        rng = np.random.default_rng(self.rng_seed_for_split)
        for i in range(len(self.lar_paths)):
            # Shuffle the default indices
            permuted_idx = mixed_indices[i]
            for _ in range(self.times_of_mixing):
                permuted_idx = permuted_idx[rng.permutation(len(permuted_idx))]
            mixed_indices[i] = permuted_idx.tolist()
        self.mixed_indices = mixed_indices

    def _set_train_val_test_nums(self):
        self.train_val_test_nums = []
        for i in range(len(self.data_lengths)):
            train = int(math.ceil(self.data_lengths[i] * self.train_val_test_fract[0]))
            test = int(math.ceil(self.data_lengths[i] * self.train_val_test_fract[-1]))
            val = int(self.data_lengths[i] - train - test)
            assert val > self.stratified_batch_sizes[i] if val != 0 else True # validation dataset cannot be smaller than the stratified batch size. val = 0 means we are evaluating the entire dataset

            self.train_val_test_nums.append([train, val, test])

    def _set_train_val_test_cumsums(self):
        train_val_test_cumsums = [
            np.cumsum(
                np.concatenate([[0], self.train_val_test_nums[i]], axis=0).astype(np.int64)
            ).tolist() for i in range(len(self.train_val_test_nums))
        ]
        self.train_val_test_cumsums = train_val_test_cumsums

    def _read_hpge_dataset(self):
        """To be called inside each worker since the hpge data is only < 1MB"""
        self.hpge_dataset = open_memmap(self.hpge_path, mode="r").copy()
        self.hpge_dataset[:, 1] = (self.hpge_dataset[:, 1] - self.hpge_energy_mean) / self.hpge_energy_std # energy is transformed to have zero mean and unit variance

    def __len__(self):
        if self.datasets is None:
            return
        return min([len(dataset) for dataset in self.datasets])

    def _worker_init(self):
        """To be called inside worker_init_fn"""
        worker_info = get_worker_info()
        self._init_dataset()
        for dataset in self.datasets:
            dataset._set_worker_id(worker_info.id, worker_info.num_workers)
            dataset._read_data_inside_worker()
            dataset.set_mode(self.mode)
        self._read_hpge_dataset()

    def set_mode(self, mode_idx: int):
        """To be called before DataLoader is intilialized in the main process."""
        self.mode = mode_idx

    def set_epoch(self, epoch: int):
        """To be called by the main process at the beginning of each epoch. Epoch value is stored in a multiprocessing.Value."""
        with self.epoch_value.get_lock():
            self.epoch_value.value = epoch

    def _shuffle(self):
        """To be called by each worker inside __iter__(), using a global rng seed."""
        rng = np.random.default_rng(self.global_rng_seed_for_sampling + self.epoch_value.value)
        for dataset in self.datasets:
            if self.shuffle:
                shuffler = rng.permutation(len(dataset.indices_buffer))
            else:
                shuffler = None
            dataset.shuffle(shuffler)

    def _close_worker_resources(self):
        try:
            for dataset in self.datasets:
                dataset.data.close()
        except Exception:
            pass

    def __iter__(self):
        try:
            assert self.datasets is not None
            self._shuffle()
            for idx in range(len(self)):
                batch = []
                labels = []
                indices = None
                break_ = False
                for i, dataset in enumerate(self.datasets):
                    partial_batch, label, indices_shard = dataset[idx]
                    batch.append(partial_batch.toarray().reshape(-1, self.num_t_bins, self.num_sipm_chs))
                    labels.append(label)
                    if label[i] == self.true_coincidence_label:
                        indices = indices_shard
                    if partial_batch is None:
                        break_ = True
                if break_:
                    break

                batch = np.concatenate(batch, axis=0) if len(batch) > 1 else batch[0]
                labels = np.concatenate(labels, axis=0) if len(labels) > 0 else labels[0]
                indices = np.array(indices, dtype=np.int64) if indices is not None else None
                yield batch, self.hpge_dataset[indices] if indices is not None else np.zeros((len(batch), 2), dtype=np.float32), labels # the indices == None case is for LAr FT data loading without physics data
        except Exception as e:
            self._close_worker_resources()
            traceback.print_exc()
            print(repr(e))
            raise

def worker_init_fn(worker_id: int):
    worker_info = get_worker_info()
    dataset: LArListDataset = worker_info.dataset
    dataset._worker_init()


class LArDatasetBase:
    def __init__(self, path: str):
        self.path = path
        self.dataset = load_npz(self.path)

    def __getitem__(self, indices):
        indices = indices[~np.isnan(indices)].astype(np.int64)
        return self.dataset[indices], indices

class BootstrappedKFoldLArListDataset(IterableDataset):
    """
        NOTE: the RC dataset is not chunked into K-folds here. The physics data is chunked.
              by default, the physics data is the one with label 1.

              But the RC training and validation dataset is bootstraped.
    """
    def __init__(
        self,
        lar_paths: list[str],
        num_t_bins: int,
        num_sipm_chs: int,
        batch_size: int,
        labels: list[int] = [0],
        prior: list[float] = [1.0],
        hpge_path: str = None,
        hpge_energy_mean: float = None,
        hpge_energy_std: float = None,
        test_folds: list[list[int]] = None,
        rng_seed_for_split: int = None,
        times_of_mixing: int = 5,
        bootstrap_rng_seed: int = None,
        global_rng_seed_for_sampling: int = None,
        num_folds: int = None,
        num_bootstraps_per_fold: int = None,
        sg_train_val_cal_test_frac: list[float] = None,
        mode: mp.Value = None, # can be an int when mode == 4 (evaluation mode)
        fold_id: mp.Value = None,
        change_bootstrap_id: mp.Value = None
    ):
        """
            self.mixed_indices is treated as a first global shuffle before the data is K-folded
        """
        super(BootstrappedKFoldLArListDataset, self).__init__()
        assert len(lar_paths) == len(prior) == len(labels)
        assert sum(prior) == 1.0

        self.num_folds = num_folds # == K
        self.num_bootstraps_per_fold = num_bootstraps_per_fold
        
        self.sg_train_val_cal_test_frac = sg_train_val_cal_test_frac
        self.rng_seed_for_split = rng_seed_for_split
        self.times_of_mixing = times_of_mixing
        self.bootstrap_rng_seed = bootstrap_rng_seed

        self.hpge_path = hpge_path
        self.lar_paths = lar_paths
        self.labels = labels
        self.hpge_energy_mean = hpge_energy_mean
        self.hpge_energy_std = hpge_energy_std

        self.test_folds = test_folds # needed in evaluation mode

        self.prior = prior
        self.batch_size = batch_size
        self.num_t_bins = num_t_bins
        self.num_sipm_chs = num_sipm_chs
        self.global_rng_seed_for_sampling = global_rng_seed_for_sampling

        self.mode = mode # train (0), val (1), calibration (2), test (3), evaluation (4)
        # NOTE: in evaluation mode (4), data is not chunked and is not bootstrapped
        self.fold_id = fold_id
        self.change_bootstrap_id = change_bootstrap_id

        self._set_stratified_batch_sizes()
        self._set_mixed_indices()
        self._set_folds_indices()

        self.bootstrap_rng: np.Generator = None
        self.sampling_rng: np.Generator = None

    def set_mode(self, mode: int):
        """To be called by the main process before iteration. mode value is stored in a multiprocessing.Value."""
        if self.mode.value == 4:
            if mode != 4:
                raise ValueError("cannot change from evaluation mode to other modes")
        else:
            if mode == 4:
                raise ValueError("cannot change from other modes to evaluation mode")

        with self.mode.get_lock():
            self.mode.value = mode

    def set_fold_id(self, fold_id: int):
        """Called inside main process"""
        with self.fold_id.get_lock():
            self.fold_id.value = fold_id

    def set_bid_flag(self, bid_flag: int):
        """Called inside main process"""
        with self.change_bootstrap_id.get_lock():
            self.change_bootstrap_id.value = bid_flag

    def _set_stratified_batch_sizes(self):
        """
            Get the stratified batch sizes according to the prior and the total batch size using np.ceil.
            By convention, the last class will take the remainder if the batch sizes do not sum to the total batch size.
        """
        stratified_batch_sizes = np.ceil(np.array(self.prior) * self.batch_size)
        if stratified_batch_sizes.sum() != self.batch_size:
            stratified_batch_sizes[-1] = stratified_batch_sizes[-1] + self.batch_size - stratified_batch_sizes.sum()
        self.prior = (stratified_batch_sizes / np.sum(stratified_batch_sizes)).tolist()
        self.stratified_batch_sizes = stratified_batch_sizes.astype(np.int64).tolist()

    def _set_mixed_indices(self):
        # Monotonicly increasing default indices
        self.data_lengths = [
            load_npz(path).shape[0] for path in self.lar_paths
        ]
        mixed_indices = [np.arange(mixing_idx).astype(np.int64) for mixing_idx in self.data_lengths]

        rng = np.random.default_rng(self.rng_seed_for_split)
        for i in range(len(self.lar_paths)):
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
        self.labels = np.array(self.labels)
        if self.mode == 4:
            self.indices = np.arange(int(np.array(self.data_lengths, dtype=np.int64)[self.labels == 0][0]), dtype=np.int64).tolist()
            return

        # label 0 data (sg / RC dataset) is treated to have 1 fold
        sg_data_cumsum = self._get_data_chunks_cumsum(int(np.array(self.data_lengths, dtype=np.int64)[self.labels == 0][0]), self.sg_train_val_cal_test_frac)

        if 1 in self.labels:
            # label 1 data with k folds (bg / physics)
            permuted_indices = np.array(self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 1][0]], dtype=np.int64)
            num_data_within_folds = len(permuted_indices) // self.num_folds
            is_incomplete_last_fold = num_data_within_folds * self.num_folds != len(permuted_indices)
            test_folds = permuted_indices[:num_data_within_folds * self.num_folds]
            test_folds = test_folds.reshape(self.num_folds, -1).tolist()
            if is_incomplete_last_fold:
                test_folds.append(permuted_indices[num_data_within_folds * self.num_folds:].tolist())

            bg_train_and_val_folds = []
            for test_fold in test_folds:
                mask = np.ones(len(permuted_indices)).astype(np.bool_)
                mask[test_fold] = False
                bg_train_and_val_folds.append(
                    permuted_indices[mask].tolist()
                )

            self.indices = {
                "sg": {
                    "train_val": self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]][:sg_data_cumsum[2]],
                    "calib": self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]][sg_data_cumsum[2]:sg_data_cumsum[3]],
                    "test": self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]][sg_data_cumsum[3]:sg_data_cumsum[4]]
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
        else:
            self.indices = {
                "sg": {
                    "train_val": self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]][:sg_data_cumsum[2]],
                    "calib": self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]][sg_data_cumsum[2]:sg_data_cumsum[3]],
                    "test": self.mixed_indices[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]][sg_data_cumsum[3]:sg_data_cumsum[4]]
                }
            }
        self.labels = self.labels.tolist()

    def _set_sg_and_bg_datasets(self):
        assert get_worker_info() is not None, "Can only be called by each worker inside worker_init"
        self.labels = np.array(self.labels)
        if self.hpge_path is not None:
            self.hpge_dataset = open_memmap(self.hpge_path, mode="r").copy()
            self.hpge_dataset[:, 1] = (self.hpge_dataset[:, 1] - self.hpge_energy_mean) / self.hpge_energy_std # energy is transformed to have zero mean and unit variance
        else:
            self.hpge_dataset = None

        self.dataset = [
            LArDatasetBase(self.lar_paths[np.array([0, 1], dtype=np.int64)[self.labels == 0][0]]) # signal
        ]
        if 1 in self.labels:
            self.dataset.append(
                LArDatasetBase(self.lar_paths[np.array([0, 1], dtype=np.int64)[self.labels == 1][0]]) # background
            )

    def _worker_init(self):
        worker_info = get_worker_info()
        self.worker_id = worker_info.id
        self.num_workers = worker_info.num_workers
        self.bootstrap_rng = np.random.default_rng(self.bootstrap_rng_seed) # bootstrap rng is the same for each worker because randomness is controlled globally
        self.sampling_rng = np.random.default_rng(self.global_rng_seed_for_sampling)
        self._set_sg_and_bg_datasets()

    def _set_current_bootstrap_indices(self):
        assert get_worker_info() is not None, "Can only be called by each worker"

        if self.mode == 4:
            return

        if self.mode.value != 0:
            return

        if self.change_bootstrap_id.value != 1:
            return

        sg_fold = np.array(self.indices["sg"]["train_val"])
        n_sg = len(sg_fold)
        sg_boot_idx = self.bootstrap_rng.choice(n_sg, size=n_sg, replace=True)
        sg_inbag_unique = np.unique(sg_boot_idx)
        sg_oob_mask = np.ones(n_sg, dtype=np.bool_)
        sg_oob_mask[sg_inbag_unique] = False
        sg_oob_idx = np.flatnonzero(sg_oob_mask)

        self.current_sg_train = sg_fold[sg_boot_idx]
        self.current_sg_val = sg_fold[sg_oob_idx]

        bg_fold = np.array(self.indices["bg"]["train_val"]['fold_{i}'.format(i=int(self.fold_id.value))])
        n_bg = len(bg_fold)
        bg_boot_idx = self.bootstrap_rng.choice(n_bg, size=n_bg, replace=True)
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

        if self.mode.value == 0:
            self.current_indices = [
                self.current_sg_train[self.sampling_rng.permutation(len(self.current_sg_train))].astype(np.float32),
                self.current_bg_train[self.sampling_rng.permutation(len(self.current_bg_train))].astype(np.float32)
            ]
        elif self.mode.value == 1:
            self.current_indices = [
                self.current_sg_val[self.sampling_rng.permutation(len(self.current_sg_val))].astype(np.float32),
                self.current_bg_val[self.sampling_rng.permutation(len(self.current_bg_val))].astype(np.float32)
            ]
        elif self.mode.value == 2:
            self.current_indices = [
                np.array(self.indices["sg"]["calib"]).astype(np.float32)
            ]
        elif self.mode.value == 3:
            self.current_indices = [
                np.array(self.indices["sg"]["test"]).astype(np.float32)
            ]
        elif self.mode == 4:
            self.current_indices = [
                np.array(self.test_folds[int(self.fold_id.value)]).astype(np.float32)
            ]

        current_indices = []
        for i in range(len(self.current_indices)):
            batch_size = self.stratified_batch_sizes[i]
            indices = self.current_indices[i]
            last_batch = None
            if self.mode.value not in (0, 1) : # no drop-last during test, calibration, and evaluation modes
                if len(indices) % batch_size != 0:
                    last_batch_len = len(indices) - batch_size * (len(indices) // batch_size)
                    # pad the last incomplete batch with nans and append to indices
                    last_batch = np.zeros(batch_size)
                    last_batch[:last_batch_len] = indices[-last_batch_len:]
                    last_batch[last_batch_len:] = np.nan
                    # the nan entries will be handled in the collate_fn

            indices = indices[:batch_size * (len(indices) // batch_size)] # drop the last incomplete batch
            if last_batch is not None:
                indices = np.concatenate((indices, last_batch), axis=0)
            indices = indices.reshape(-1, batch_size)
            indices = indices[self.worker_id::self.num_workers]
            current_indices.append(indices)
        self.current_indices = current_indices

    def __len__(self):
        return min([len(indices) for indices in self.current_indices])

    def __iter__(self):
        self._set_current_bootstrap_indices()
        self._shuffle()
        for idx in range(len(self)):
            batch = []
            labels = []
            indices = []
            for i in range(len(self.labels)):
                indices_shard = self.current_indices[i][idx]
                partial_batch, indices_shard = self.dataset[i][indices_shard]
                partial_label = np.ones(len(indices_shard), dtype=np.float32) * self.labels[i]
                batch.append(partial_batch.toarray().reshape(-1, self.num_t_bins, self.num_sipm_chs))
                labels.append(partial_label)
                indices.append(indices_shard)
            
            if self.hpge_dataset is not None:
                index = np.array([0, 1], dtype=np.int64)[self.labels == 0][0]
                batch = batch[index]
                gE = self.hpge_dataset[indices[index]]
                labels = labels[index]
            else:
                index = np.array([0, 1], dtype=np.int64)[self.labels == 0][0]
                gE = np.zeros((len(batch[index]), 2), dtype=np.float32)
                batch = np.concatenate(batch, axis=0) if len(batch) > 1 else batch[0]
                labels = np.concatenate(labels, axis=0) if len(labels) > 1 else labels[0]
            yield batch, gE, labels

def KFoldBootstrap_worker_init_fn(worker_id: int):
    worker_info = get_worker_info()
    dataset: BootstrappedKFoldLArListDataset = worker_info.dataset
    dataset._worker_init()
