import os
import shutil
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path

from typing import Tuple, Dict

import json

@dataclass
class NRECConfig:
    rng_seed: int = None
    rel_tolerance: float = None
    patience: int = None
    max_epochs: int = None

    num_folds: int = None
    num_bootstraps_per_fold: int = None

    # NRE-C toggle
    gamma: int = None
    temperature: float = None

    # Transformer general config
    hidden_size: int = None
    intermediate_size: int = None
    num_attention_heads: int = None
    block_resid_dropout1: float = None
    block_resid_dropout2: float = None
    attn_dropout: float = None
    causal: int = None

    # LAr encoder
    num_sipms: int = None
    num_sipm_t_bins: int = None
    sipm_num_rz_bands: int = None
    sipm_max_freq_log2_rz: float = None
    sipm_num_phi_harmonics: int = None
    sipm_cls_placeholder_id: int = None

    sipm_num_layers: int = None

    # HPGe encoder
    hpge_global_partitioning_size: int = None

    num_hpges: int = None
    hpge_num_rz_bands: int = None
    hpge_max_freq_log2_rz: float = None
    hpge_num_phi_harmonics: int = None

    hpge_num_features: int = None
    hpge_num_feat_bands: int = None
    hpge_feat_max_freq_log2: float = None
    hpge_cls_placeholder_id: int = None

    hpge_num_layers: int = None

    # Training data
    lar_paths: list[str] = None
    hpge_path: str = None
    hpge_feats_mean: list[float] = None
    hpge_feats_std: list[float] = None

    # Training hyperparams
    local_batch_size: int = None
    sg_train_val: list[float] = None
    times_of_mixing: int = None

    lr_model: float = None
    betas_model: Tuple[float, float] = None
    weight_decay: float = None

    save_to: str = None

@dataclass(frozen=True)
class Paths:
    root: Path # working dir
    trained: Path # trained models dir
    db_conf: Path # database configs
    data_dir: Path # dir containing training data

    def make_checkpoint_dir(self, experiment: str, model_name: str, version: str):
        if self.data_dir is None:
            raise ValueError("data_dir cannot be None")
        (self.trained / experiment / model_name / version / "checkpoints").mkdir(parents=True, exist_ok=True)

def load_config(config_json: str, working_dir: Path, data_dir: Path) -> Tuple[Dict, Dict, Paths]:
    with open(config_json, "r") as f:
        config_json = json.load(f)
    model_config = config_json["model_config"]
    data_config = {key: config_json[key] for key in config_json if key != "model_config"}

    paths = Paths(
        root = working_dir,
        trained= working_dir / "trained",
        db_conf = working_dir / "meta",
        data_dir = data_dir
    )

    return model_config, data_config, paths

def init_config(paths: Paths, experiment: str, model_name: str, version: str, model_config: dict, config_obj = None):
    config = NRECConfig() if config_obj is None else config_obj
    config.__dict__.update(model_config)

    config.save_to = str(paths.trained / experiment / model_name / version / "checkpoints")
    return config

def _initialize_configs(
    config_obj: NRECConfig,
    wd: Path,
    experiment: str,
    model_name: str,
    version: str,
    mmpd: Path,
    training_config: str = None
) -> Tuple[NRECConfig, dict, Paths]:
    config_json = wd / "trained" / experiment / model_name / version / f"{model_name}_{version}.json"
    os.makedirs(os.path.dirname(str(config_json)), exist_ok=True)
    if training_config is not None and not config_json.exists():
        shutil.copy(training_config, config_json)

    model_config, data_config, paths = load_config(config_json, wd, mmpd)
    paths.make_checkpoint_dir(experiment, model_name, version)

    config = init_config(
        paths=paths,
        experiment=experiment,
        model_name=model_name,
        version=version,
        model_config=model_config,
        config_obj=config_obj
    )

    return config, data_config, paths

@dataclass
class EvalConfig:
    unconditional_cp_id: int = None
    conditional_cp_id: int = None

    local_batch_size: int = None
    calib_dataset_frac: float = None

    num_zero_pe_in_lar_ft: int = None
    num_high_pe_in_lar_ft: int = None

    alpha: float = None
    alpha_epistemic: float = None

    global_calib_frac: float = None

    phy_4by4_data: str = None
    fc_4by4_data: str = None
