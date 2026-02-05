from dataclasses import dataclass
from pathlib import Path

from typing import Tuple, Dict

import json

@dataclass
class ModelConfig:
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

    # Training hyperparams
    local_batch_size: int = None
    lr_model: float = None
    betas_model: Tuple[float, float] = None
    weight_decay: float = None

    data_sampling_seed: int = None
    data_sampling_cycle: int = None
    collate_fn_seed: int = None

    # LR search
    start_lr: float = None
    end_lr: float = None
    lr_sweep_steps: int = None
    constant_lr_steps: int = None
    use_wandb: bool = None

    # Paths
    save_to: str = None

@dataclass(frozen=True)
class Paths:
    root: Path # working dir
    trained: Path # trained models dir
    db_conf: Path # database configs
    mmap: Path # dir to save mmap files

    def make_checkpoint_dir(self, experiment: str, model_name: str, version: str):
        if self.mmap is None:
            raise ValueError("mmap path cannot be None")
        (self.trained / experiment / model_name / version / "checkpoints").mkdir(parents=True, exist_ok=True)

def load_config(config_json: str, working_dir: Path, mmap_dir: Path) -> Tuple[Dict, Dict, Paths]:
    with open(config_json, "r") as f:
        config_json = json.load(f)
    model_config = config_json["model_config"]
    data_config = {key: config_json[key] for key in config_json if key != "model_config"}

    paths = Paths(
        root = working_dir,
        trained= working_dir / "trained",
        db_conf = working_dir / "meta",
        mmap = mmap_dir
    )

    return model_config, data_config, paths

def init_config(paths: Paths, experiment: str, model_name: str, version: str, model_config: dict, config_obj = None):
    config = ModelConfig() if config_obj is None else config_obj
    config.__dict__.update(model_config)

    for attr in ("use_flash_attn", "fused_mlp", "fused_bias_fc", "fused_dropout_add_ln"):
        if not hasattr(config, attr):
            continue
        setattr(config, attr, bool(getattr(config, attr)))

    config.save_to = str(paths.trained / experiment / model_name / version / "checkpoints")
    return config
