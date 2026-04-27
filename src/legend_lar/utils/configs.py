import os
import json
import shutil
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class NRECConfig:
    rng_seed: int = None
    rel_tolerance: float = None
    patience: int = None
    max_epochs: int = None

    num_folds: int = None
    num_bootstraps_per_fold: int = None

    # NRE-C toggle
    K: int = None
    gamma: int = None
    temperature: float = None
    deep_supervision: int = None
    alpha_t: list[float] = None
    lambda_aux: list[float] = None

    # Transformer general config
    hidden_size: int = None
    intermediate_size: int = None
    num_attention_heads: int = None
    block_resid_dropout1: float = None
    block_resid_dropout2: float = None
    attn_dropout: float = None
    cls_placeholder_id: int = None

    # Geometry table general config
    num_rz_bands: int = None
    max_freq_log2_rz: float = None
    num_phi_harmonics: int = None
    r_shift: float = None
    r_inv_scale: float = None
    z_shift: float = None
    z_inv_scale: float = None

    # LAr encoder
    num_sipms: int = None
    num_sipm_t_bins: int = None
    sipm_unbinned_pe: int = None
    sipm_num_feat_bands: int = None
    sipm_feat_max_freq_log2: float = None

    sipm_num_layers: int = None

    # HPGe encoder
    hpge_global_partitioning_size: int = None

    num_hpges: int = None
    hpge_num_features: int = None
    hpge_num_feat_bands: int = None
    hpge_feat_max_freq_log2: float = None

    hpge_num_layers: int = None

    # Training data
    lar_paths: list[str] = None
    hpge_path: str = None
    hpge_feats_mean: list[float] = None
    hpge_feats_std: list[float] = None
    subpartition_hpge_feats: int = None

    sipm_pe_scale: float = 1.0

    # Training hyperparams
    local_batch_size: int = None
    times_of_mixing: int = None

    lr_model: float = None
    betas_model: Tuple[float, float] = None
    weight_decay: float = None


def _load_meta_from_config(config_json: str) -> Tuple[Dict, Dict]:
    with open(config_json, "r") as f:
        config_json = json.load(f)
    model_config = config_json["model_config"]
    data_config = {key: config_json[key] for key in config_json if key != "model_config"}

    return model_config, data_config

def _parse_meta_to_config(model_config: dict, config_obj = None):
    config = NRECConfig() if config_obj is None else config_obj
    config.__dict__.update(model_config)

    return config

def _initialize_configs(
    config_obj: NRECConfig,
    config_path: str,
    base_config: str = None
) -> Tuple[NRECConfig, dict]:
    os.makedirs(os.path.dirname(str(config_path)), exist_ok=True)

    if not os.path.isfile(config_path):
        if base_config is None:
            raise ValueError(f'variable base_config with {base_config} cannot be None when arg config_path is not a file')
        else:
            shutil.copy(base_config, config_path)

    model_config, data_config = _load_meta_from_config(config_path)
    config = _parse_meta_to_config(model_config, config_obj)

    return config, data_config
