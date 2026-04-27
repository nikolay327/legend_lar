from functools import partial
from typing import Sequence

import numpy as np
import scipy

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from legend_lar.model import NREC
from legend_lar.utils import NRECConfig, _initialize_configs, decode_geom, FileDB, pack_hpge_nrec_data
from legend_lar.data import ParallelBootstrappedKFoldLArListDataset, ParallelKFoldBootstrap_worker_init_fn, NRECCollateFn


class NRECPrefixScanEngine:
    """
    Fold-specific latent-bank engine for 1D HPGe feature scans.

    Assumes config.deep_supervision == 1 so the HPGe encoder returns all causal prefixes.

    Expected usage pattern:

        engine = NRECPrefixScanEngine(
            file_db=file_db,
            partition=partition,
            model_name=model_name,
            version=version,
            dataset_version=dataset_version,
            batch_size=512,
            device=device,
            fold_id=0
        )

        engine.store_rc_latent_bank()

        engine.set_scan_spec(
            template=[gid, energy, drift_time, aoe, lq],
            scan_feature_idx=2, # vary drift_time
            grid_values=np.linspace(100, 1500, 201),
            normalized=False
        )

        engine.store_hpge_scan_bank()

        # Score at any prefix by lookup
        scores_t2 = engine.compute_scores(prefix_idx=2, mode="evidence")
        scores_t4 = engine.compute_scores(prefix_idx=4, mode="evidence")

        # Optional per-member scores
        scores_t4, member_scores_t4 = engine.compute_scores(
            prefix_idx=4,
            mode="evidence",
            return_member_scores=True
        )

        # Change selected bids on the fly
        engine.set_selected_bids([0, 2, 5])

        # Change fold id on the fly
        engine.set_fold_id(1)

        # Same RC bank API, same scan API
        engine.store_rc_latent_bank()
        engine.set_scan_spec(template=my_template, scan_feature_idx=3, grid_values=my_grid, normalized=False)
        engine.store_hpge_scan_bank()
    """

    def __init__(
        self,
        file_db: FileDB,
        partition: str,
        model_name: str,
        version: str,
        dataset_version: str,
        batch_size: int,
        device: str | int,
        fold_id: int,
        selected_bids: Sequence[int] | None = None
    ):
        self.file_db = file_db
        self.partition = partition
        self.model_name = model_name
        self.version = version
        self.dataset_version = dataset_version
        self.batch_size = int(batch_size)
        self.device = device

        model_cfg, data_config = _initialize_configs(
            config_obj=NRECConfig(),
            config_path=file_db.build_file(
                tier="model_config",
                partition=partition,
                model_name=model_name,
                version=version,
            ),
        )
        self.config = model_cfg
        self.data_config = data_config

        if self.config.deep_supervision != 1:
            raise ValueError("NRECPrefixScanEngine requires config.deep_supervision == 1.")

        self.inv_temp = 1.0 / self.config.temperature
        self._n_raw_hpge_feats = self.config.hpge_num_features + (
            2 if self.config.subpartition_hpge_feats == 1 else 1
        )
        self._n_hpge_prefixes = self._n_raw_hpge_feats + 1

        lar_detector_coords, hpge_detector_coords = decode_geom(
            self.file_db.build_file(
                tier="dataset",
                partition=self.partition,
                filename="detector_positions.yaml",
            ),
            self.config,
        )
        self.lar_detector_coords = lar_detector_coords
        self.hpge_detector_coords = hpge_detector_coords

        # Active ensemble/model state
        self.fold_id: int | None = None
        self.selected_bids: list[int] = []
        self._models_by_bid: dict[int, nn.Module] = {}

        # RC bank: bid -> (N_rc, D)
        self._rc_bank_by_bid: dict[int, Tensor] = {}
        self._rc_indices: np.ndarray | None = None

        # Scan state
        self._scan_feature_idx: int | None = None
        self._scan_template: Tensor | None = None
        self._scan_grid_values: Tensor | None = None
        self._scan_pack: tuple[Tensor, Tensor, Tensor, Tensor, int, Tensor] | None = None

        # HPGe all-prefix bank: bid -> (N_grid, P, D)
        self._scan_bank_by_bid: dict[int, Tensor] = {}
        self._scan_valid_mask: Tensor | None = None # (N_grid, P), shared across bids

        self._init_rc_dataloader()
        self.set_fold_id(fold_id, selected_bids=selected_bids)

    def _init_rc_dataloader(self):
        lar_dataset = self.file_db.build_file(
            tier="inference_dataset",
            partition=self.partition,
            version=self.dataset_version,
            filename=self.data_config["ev_ep_null"]
        )
        lar_dataset = scipy.sparse.load_npz(lar_dataset)

        dataset = ParallelBootstrappedKFoldLArListDataset(
            lar_data_lengths=[lar_dataset.shape[0]],
            num_t_bins=self.config.num_sipm_t_bins,
            num_sipm_chs=self.config.num_sipms,
            batch_size=self.batch_size,
            sipm_pe_scale=self.config.sipm_pe_scale,
            mode=3
        )

        collate_fn = NRECCollateFn(
            cls_placeholder_id=self.config.cls_placeholder_id,
            has_cls=True,
            sipm_unbinned_pe=self.config.sipm_unbinned_pe == 1,
            cuda_device=self.device
        )

        worker_init_fn = partial(
            ParallelKFoldBootstrap_worker_init_fn,
            None,
            [lar_dataset]
        )

        self.rc_dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    def _clean_state_dict(self, state_dict):
        cleaned_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace("_orig_mod.", "")
            cleaned_dict[cleaned_key] = value
        return cleaned_dict

    def _normalize_bid_list(self, bids: Sequence[int] | None) -> list[int]:
        if bids is None:
            bids = list(range(self.config.num_bootstraps_per_fold))

        bids = [int(b) for b in bids]
        if len(bids) == 0:
            raise ValueError("selected_bids cannot be empty.")

        out = []
        seen = set()
        for bid in bids:
            if bid < 0 or bid >= self.config.num_bootstraps_per_fold:
                raise ValueError(f"Invalid bid={bid}.")
            if bid not in seen:
                out.append(bid)
                seen.add(bid)
        return out

    def _load_models_for_bids(self, bids: Sequence[int]):
        for bid in bids:
            if bid in self._models_by_bid:
                continue

            model = NREC(
                lar_detector_coords=self.lar_detector_coords,
                hpge_detector_coords=self.hpge_detector_coords,
                config=self.config,
                device=self.device
            ).to(dtype=torch.float32, device=self.device)

            cp = self.file_db.build_file(
                tier="models",
                partition=self.partition,
                model_name=self.model_name,
                version=self.version,
                fid=self.fold_id,
                bid=bid
            )
            cp = torch.load(cp, weights_only=True, map_location=self.device)["model"]
            cp = self._clean_state_dict(cp)
            model.load_state_dict(cp, strict=True)
            model = torch.compile(model, dynamic=True)
            model.eval()

            self._models_by_bid[bid] = model

    @torch.no_grad()
    def _encode_lar_batch(self, model: NREC, lar) -> Tensor:
        (_, t_idx, s_idx, v_val, cu_seqlens, max_seqlen, _) = lar

        t_idx = t_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        s_idx = s_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        v_val = (
            v_val.to(device=self.device, non_blocking=True).to(dtype=torch.float32)
            if v_val is not None
            else None
        )
        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        max_seqlen = int(max_seqlen)

        e_lar = model.lar_encoder(
            t_idx=t_idx,
            s_idx=s_idx,
            v_val=v_val,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            geom_tokenizer=model.geom_tokenizer
        )
        e_lar = F.normalize(e_lar, p=2, dim=-1)
        return e_lar

    @torch.no_grad()
    def _encode_all_prefix_hpge_bank_for_model(self, model: NREC) -> tuple[Tensor, Tensor]:
        if self._scan_pack is None:
            raise RuntimeError("No scan specification has been set.")

        ge_b_idx, f_idx, f_vals, ge_cu_seqlens, ge_max_seqlen, _ = self._scan_pack

        ge_b_idx = ge_b_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        f_idx = f_idx.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        f_vals = f_vals.to(device=self.device, non_blocking=True).to(dtype=torch.float32)
        ge_cu_seqlens = ge_cu_seqlens.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        ge_max_seqlen = int(ge_max_seqlen)

        e_hpge = model.hpge_encoder(
            f_idx=f_idx,
            f_vals=f_vals,
            cu_seqlens=ge_cu_seqlens,
            max_seqlen=ge_max_seqlen,
            geom_tokenizer=model.geom_tokenizer
        )
        e_hpge = F.normalize(e_hpge, p=2, dim=-1)

        n_grid = int(ge_cu_seqlens.numel() - 1)
        D = e_hpge.shape[-1]
        P = self._n_hpge_prefixes

        bank = torch.zeros((n_grid, P, D), device=self.device, dtype=e_hpge.dtype)
        valid_mask = torch.zeros((n_grid, P), device=self.device, dtype=torch.bool)

        cls_mask = f_idx == self.config.cls_placeholder_id
        non_cls_mask = ~cls_mask

        # prefix 0 = SOS
        bank[ge_b_idx[cls_mask], 0] = e_hpge[cls_mask]
        valid_mask[ge_b_idx[cls_mask], 0] = True

        # prefix p>0 corresponds to raw feature id (p-1)
        bank[ge_b_idx[non_cls_mask], f_idx[non_cls_mask] + 1] = e_hpge[non_cls_mask]
        valid_mask[ge_b_idx[non_cls_mask], f_idx[non_cls_mask] + 1] = True

        return bank, valid_mask

    def _normalize_hpge_inputs(
        self,
        x: Tensor,
        feature_ids: Tensor | None = None
    ) -> Tensor:
        mean = torch.as_tensor(self.config.hpge_feats_mean, dtype=torch.float32)
        std = torch.as_tensor(self.config.hpge_feats_std, dtype=torch.float32)

        x = x.to(torch.float32).clone()

        if feature_ids is None:
            if x.numel() != self._n_raw_hpge_feats:
                raise ValueError(
                    f"Expected template length {self._n_raw_hpge_feats}, got {x.numel()}."
                )
            return (x - mean[: self._n_raw_hpge_feats]) / std[: self._n_raw_hpge_feats]

        feature_ids = feature_ids.to(torch.long)
        return (x - mean.index_select(0, feature_ids)) / std.index_select(0, feature_ids)

    def set_fold_id(self, fold_id: int, selected_bids: Sequence[int] | None = None):
        fold_id = int(fold_id)
        if fold_id < 0 or fold_id >= self.config.num_folds:
            raise ValueError(f"Invalid fold_id={fold_id}.")

        if self.fold_id == fold_id:
            if selected_bids is not None:
                self.set_selected_bids(selected_bids)
            return

        self.fold_id = fold_id

        self._models_by_bid.clear()
        self._rc_bank_by_bid.clear()
        self._scan_bank_by_bid.clear()
        self._scan_valid_mask = None
        self._rc_indices = None

        self.selected_bids = []
        self.set_selected_bids(selected_bids)

    def set_selected_bids(self, selected_bids: Sequence[int] | None):
        bids = self._normalize_bid_list(selected_bids)

        had_rc_cache = len(self._rc_bank_by_bid) > 0
        had_scan_cache = len(self._scan_bank_by_bid) > 0 and self._scan_pack is not None

        self._load_models_for_bids(bids)
        self.selected_bids = bids

        # If banks already exist, only compute missing bids.
        if had_rc_cache:
            self.store_rc_latent_bank()

        if had_scan_cache:
            self.store_hpge_scan_bank()

    def set_scan_spec(
        self,
        template: Sequence[float] | np.ndarray | Tensor,
        scan_feature_idx: int,
        grid_values: Sequence[float] | np.ndarray | Tensor,
        normalized: bool = False
    ):
        """
        Build a scan bank specification.

        The full HPGe vector is kept for each grid point:
        - all features are copied from the fixed template
        - feature scan_feature_idx is replaced by the grid value

        Then the causal HPGe encoder outputs all available prefixes, which are
        banked for lookup later.
        """
        scan_feature_idx = int(scan_feature_idx)
        if scan_feature_idx < 0 or scan_feature_idx >= self._n_raw_hpge_feats:
            raise ValueError(f"Invalid scan_feature_idx={scan_feature_idx}.")

        template = torch.as_tensor(template, dtype=torch.float32).flatten()
        if template.numel() != self._n_raw_hpge_feats:
            raise ValueError(
                f"Expected template length {self._n_raw_hpge_feats}, got {template.numel()}."
            )

        grid_values = torch.as_tensor(grid_values, dtype=torch.float32).flatten()
        if grid_values.numel() == 0:
            raise ValueError("grid_values cannot be empty.")

        if not normalized:
            template = self._normalize_hpge_inputs(template)
            grid_values = self._normalize_hpge_inputs(
                grid_values,
                feature_ids=torch.full((grid_values.numel(),), scan_feature_idx, dtype=torch.long)
            )

        x_grid = template.unsqueeze(0).expand(grid_values.numel(), -1).clone()
        x_grid[:, scan_feature_idx] = grid_values

        ge_b_idx, ge_f_idx, ge_f_vals, ge_cu_seqlens, ge_max_seqlen, ge_lengths = pack_hpge_nrec_data(
            x_grid,
            cls_placeholder_id=self.config.cls_placeholder_id,
            has_cls=True
        )

        self._scan_feature_idx = scan_feature_idx
        self._scan_template = template
        self._scan_grid_values = grid_values
        self._scan_pack = (
            ge_b_idx,
            ge_f_idx,
            ge_f_vals,
            ge_cu_seqlens,
            ge_max_seqlen,
            ge_lengths
        )

        # Reset only the HPGe scan bank
        self._scan_bank_by_bid.clear()
        self._scan_valid_mask = None

    @torch.no_grad()
    def store_rc_latent_bank(self):
        missing_bids = [bid for bid in self.selected_bids if bid not in self._rc_bank_by_bid]
        if len(missing_bids) == 0:
            return

        tmp = {bid: [] for bid in missing_bids}
        all_indices = [] if self._rc_indices is None else None

        for (lar, _), indices in self.rc_dataloader:
            if all_indices is not None:
                all_indices.append(np.asarray(indices).astype(np.int64))

            for bid in missing_bids:
                e_lar = self._encode_lar_batch(self._models_by_bid[bid], lar)
                tmp[bid].append(e_lar)

        for bid in missing_bids:
            self._rc_bank_by_bid[bid] = torch.cat(tmp[bid], dim=0).contiguous()

        if all_indices is not None:
            self._rc_indices = np.concatenate(all_indices, axis=0)

    @torch.no_grad()
    def store_hpge_scan_bank(self):
        if self._scan_pack is None:
            raise RuntimeError("No scan specification has been set.")

        missing_bids = [bid for bid in self.selected_bids if bid not in self._scan_bank_by_bid]
        if len(missing_bids) == 0:
            return

        for bid in missing_bids:
            bank, valid_mask = self._encode_all_prefix_hpge_bank_for_model(self._models_by_bid[bid])
            self._scan_bank_by_bid[bid] = bank

            if self._scan_valid_mask is None:
                self._scan_valid_mask = valid_mask
            else:
                # same scan spec should imply same validity pattern across models
                if not torch.equal(self._scan_valid_mask, valid_mask):
                    raise RuntimeError("HPGe prefix-validity mask mismatch across ensemble members.")

    def store_latent_banks(self, store_rc: bool = True, store_scan: bool = True):
        if store_rc:
            self.store_rc_latent_bank()
        if store_scan:
            self.store_hpge_scan_bank()

    def _stack_active_rc_bank(self) -> Tensor:
        missing = [bid for bid in self.selected_bids if bid not in self._rc_bank_by_bid]
        if len(missing) > 0:
            self.store_rc_latent_bank()
        return torch.stack([self._rc_bank_by_bid[bid] for bid in self.selected_bids], dim=0)

    def _stack_active_scan_bank(self) -> Tensor:
        if self._scan_pack is None:
            raise RuntimeError("No scan specification has been set.")
        missing = [bid for bid in self.selected_bids if bid not in self._scan_bank_by_bid]
        if len(missing) > 0:
            self.store_hpge_scan_bank()
        return torch.stack([self._scan_bank_by_bid[bid] for bid in self.selected_bids], dim=0)

    def compute_scores(
        self,
        prefix_idx: int,
        mode: str = "evidence",
        return_member_scores: bool = False
    ):
        """
        Returns aggregate score tensor of shape (N_grid, N_rc).

        If return_member_scores=True, also returns the per-member score tensor
        of shape (n_selected_bids, N_grid, N_rc).
        """
        prefix_idx = int(prefix_idx)
        if prefix_idx < 0 or prefix_idx >= self._n_hpge_prefixes:
            raise ValueError(f"Invalid prefix_idx={prefix_idx}.")

        mode = str(mode).lower()
        if mode not in ("evidence", "epistemic"):
            raise ValueError("mode must be either 'evidence' or 'epistemic'.")

        rc_bank = self._stack_active_rc_bank() # (E, N_rc, D)
        scan_bank = self._stack_active_scan_bank() # (E, N_grid, P, D)

        hpge_prefix = scan_bank[:, :, prefix_idx, :] # (E, N_grid, D)
        member_scores = torch.einsum("egd,erd->egr", hpge_prefix, rc_bank) * self.inv_temp # (E, N_grid, N_rc)

        if self._scan_valid_mask is None:
            raise RuntimeError("Scan validity mask is not available.")
        valid = self._scan_valid_mask[:, prefix_idx] # (N_grid,)

        if not valid.all():
            member_scores = member_scores.clone()
            member_scores[:, ~valid, :] = float("nan")

        if mode == "evidence":
            if member_scores.shape[0] == 1:
                agg = member_scores[0]
            else:
                agg = member_scores.mean(dim=0)
        else:
            if member_scores.shape[0] < 2:
                raise ValueError("Epistemic mode requires at least 2 selected ensemble members.")
            agg = torch.var(member_scores, dim=0, correction=0)

        if return_member_scores:
            return agg, member_scores
        return agg

    @property
    def rc_indices(self) -> np.ndarray | None:
        return self._rc_indices

    @property
    def scan_feature_idx(self) -> int | None:
        return self._scan_feature_idx

    @property
    def scan_grid_values(self) -> Tensor | None:
        return self._scan_grid_values

    @property
    def scan_valid_mask(self) -> Tensor | None:
        return self._scan_valid_mask
    
    @property
    def num_hpge_prefixes(self) -> int:
        return self._n_hpge_prefixes
