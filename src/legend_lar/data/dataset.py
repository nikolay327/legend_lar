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
        batch_size: int
    ):
        self.lar_path = lar_path
        self.label = label
        self.shuffled_indices = shuffled_indices
        self.train_val_test_cumsum = train_val_test_cumsum
        self.batch_size = batch_size

        self.mode_idx = None # train (0), val (1), or test (2)
        self.indices_buffer = None
        self.indices = None

        self.data = None

        # worker specifics
        self.worker_id = None
        self.num_workers = None

    def set_mode(self, mode_idx: int):
        """To be called before DataLoader is intilialized."""
        assert mode_idx in (0, 1, 2)
        self.indices_buffer = np.arange(self.train_val_test_cumsum[mode_idx], self.train_val_test_cumsum[mode_idx+1])

    def shuffle(self, shuffler: np.ndarray):
        """To be called by each worker, using a global rng seed."""
        assert len(self.indices_buffer) == len(shuffler)
        self.indices = np.array(self.indices_buffer, dtype=np.int64)[shuffler]

        self.indices = self.indices[:self.batch_size * (len(self.indices) // self.batch_size)]  # drop the last incomplete batch
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
        return self.data[ids.tolist()], np.ones(self.batch_size, dtype=np.float32) * self.label, ids

class LArListDataset(IterableDataset):
    def __init__(
        self,
        hpge_path: str,
        lar_paths: str,
        prior: list[float],
        labels: list[int],
        true_coincidence_label: int,
        train_val_test_fract: list[float],
        local_batch_size: int,
        rng_seed_for_split: int,
        times_of_mixing: int,
        global_rng_seed_for_sampling: int,
        epoch_value: mp.Value = None
    ):
        super(LArListDataset, self).__init__()
        assert len(lar_paths) == len(prior)
        assert sum(prior) == 1.0
        assert prior[0] == prior[1] == 0.5, "Only supports 50:50 prior for now"

        self.train_val_test_fract = train_val_test_fract
        self.rng_seed_for_split = rng_seed_for_split
        self.times_of_mixing = times_of_mixing

        self.hpge_path = hpge_path
        self.lar_paths = lar_paths
        self.labels = labels
        self.true_coincidence_label = true_coincidence_label
        self.prior = prior
        self.batch_size = local_batch_size
        self.global_rng_seed_for_sampling = global_rng_seed_for_sampling
        self.epoch_value = epoch_value

        self.stratified_batch_sizes = None
        self.mixed_indices = None
        self.train_val_test_nums = None
        self.train_val_test_cumsums = None
        self.mode = None  # train (0), val (1), or test (2)

        self._set_stratified_batch_sizes()

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
                mmap_path=self.lar_paths[i],
                label=self.labels[i],
                shuffled_indices=self.mixed_indices[i],
                train_val_test_cumsum=self.train_val_test_cumsums[i],
                batch_size=self.stratified_batch_sizes[i]
            )
            self.datasets.append(lar_dataset)
    
    def _set_mixed_indices(self):
        # Monotonicly increasing default indices
        self.data_lengths = [
            dataset.data.shape[0] for dataset in self.datasets
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
            assert val > self.stratified_batch_sizes[i] # validation dataset cannot be smaller than the stratified batch size

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
        self._read_hpge_dataset()

        self._set_mixed_indices()
        self._set_train_val_test_nums()
        self._set_train_val_test_cumsums()
    
    def set_mode(self, mode_idx: int):
        """To be called before DataLoader is intilialized in the main process."""
        assert mode_idx in (0, 1, 2)
        self.mode = mode_idx
        for dataset in self.datasets:
            dataset.set_mode(mode_idx)

    def set_epoch(self, epoch: int):
        """To be called by the main process at the beginning of each epoch. Epoch value is stored in a multiprocessing.Value."""
        with self.epoch_value.get_lock():
            self.epoch_value.value = epoch

    def _shuffle(self):
        """To be called by each worker inside __iter__(), using a global rng seed."""
        rng = np.random.default_rng(self.global_rng_seed_for_sampling + self.epoch_value.value)
        for dataset in self.datasets:
            shuffler = rng.permutation(len(dataset.indices_buffer))
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
                indices = []
                break_ = False
                for dataset in self.datasets:
                    partial_batch, label, indices_shard = dataset[idx]
                    batch.append(partial_batch)
                    labels.append(label)
                    indices.append(indices_shard)
                    if partial_batch is None:
                        break_ = True
                if break_:
                    break

                batch = np.concatenate(batch, axis=0)
                labels = np.concatenate(labels, axis=0)
                indices = np.array(indices[self.true_coincidence_label], dtype=np.int64)
                yield batch, self.hpge_dataset[indices], labels
        except Exception as e:
            self._close_worker_resources()
            traceback.print_exc()
            print(repr(e))
            raise

def worker_init_fn(worker_id: int):
    worker_info = get_worker_info()
    dataset: LArListDataset = worker_info.dataset
    dataset._worker_init()
