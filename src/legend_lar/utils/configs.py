import os
import shutil
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path

from typing import Tuple, Dict

import json

@dataclass
class ModelConfig:
    num_hpges: int = None
    num_hpge_features: int = None
    attn_num_hpge_emb_layers: int = None
    global_partitioning_size: int = None
    num_sipms: int = None
    num_sipm_t_bins: int = None

    hidden_size: int = None
    intermediate_size: int = None
    num_attention_heads: int = None

    norm_gate_tanh_scale: float = None
    norm_zero_init: int = None
    
    block_resid_dropout1: float = None
    block_resid_dropout2: float = None

    num_layers: int = None
    causal: int = None

    temperature: float = None
    hpge_energy_mean: float = None
    hpge_energy_std: float = None

    hpge_id_and_energy: str = None
    data_paths: list[str] = None
    prior: list[float] = None
    labels: list[int] = None
    true_coincidence_label: int = None

    # Training hyperparams
    local_batch_size: int = None
    train_val_test_fract: list[float] = None
    rng_seed_for_split: int = None
    times_of_mixing: int = None
    global_rng_seed_for_sampling: int = None

    lr_model: float = None
    betas_model: Tuple[float, float] = None
    weight_decay: float = None

    # calibration check
    ece_bins: int = None
    n_classes: int = None

    save_to: str = None

@dataclass
class BootstrappedKFoldConfig:
    rng_seed: int = None
    rel_tolerance: float = None
    patience: int = None
    max_epochs: int = None

    num_folds: int = None
    num_bootstraps_per_fold: int = None

    # NRE-C toggle
    gamma: int = None

    num_hpges: int = None
    num_sipms: int = None
    num_sipm_t_bins: int = None

    hidden_size: int = None
    intermediate_size: int = None
    num_attention_heads: int = None

    norm_gate_tanh_scale: float = None
    norm_zero_init: int = None
    
    block_resid_dropout1: float = None
    block_resid_dropout2: float = None

    num_layers: int = None
    causal: int = None

    temperature: float = None
    hpge_energy_mean: float = None
    hpge_energy_std: float = None

    hpge_id_and_energy: str = None
    data_paths: list[str] = None
    prior: list[float] = None
    labels: list[int] = None

    # Training hyperparams
    local_batch_size: int = None
    sg_train_val_cal_test_frac: list[float] = None
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
    config = ModelConfig() if config_obj is None else config_obj
    config.__dict__.update(model_config)

    config.save_to = str(paths.trained / experiment / model_name / version / "checkpoints")
    return config

def _initialize_configs(
    config_obj: ModelConfig | BootstrappedKFoldConfig,
    wd: Path,
    experiment: str,
    model_name: str,
    version: str,
    mmpd: Path,
    training_config: str = None
) -> Tuple[ModelConfig | BootstrappedKFoldConfig, dict, Paths]:
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
